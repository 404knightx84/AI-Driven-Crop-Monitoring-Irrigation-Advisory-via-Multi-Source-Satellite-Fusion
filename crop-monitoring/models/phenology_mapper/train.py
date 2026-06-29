"""
models/phenology_mapper/train.py
Training script for the temporal Transformer phenology stage classifier.

Dataset: per-pixel NDVI/EVI time-series with stage labels.
Each sample:
  x: (T, 2)   float32 — NDVI + EVI over T = 23 time steps
  y: int       — phenology stage label (0–5)
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from loguru import logger

from models.phenology_mapper.transformer import PhenologyTransformer, build_model


class PhenologyDataset(Dataset):
    """
    Loads (T, 2) NDVI/EVI sequences and corresponding stage labels.
    Expects:
      sequences_path: *.npy of shape (N, T, 2)
      labels_path:    *.npy of shape (N,)  — int stage labels
    """

    def __init__(self, sequences_path: str, labels_path: str):
        self.X = np.load(sequences_path).astype(np.float32)   # (N, T, 2)
        self.y = np.load(labels_path).astype(np.int64)        # (N,)
        assert len(self.X) == len(self.y)
        logger.info(f"[PhenologyDataset] {len(self.X)} samples, T={self.X.shape[1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class PhenologyTrainer:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = build_model(cfg).to(self.device)

        # Class weights to handle imbalance (germination / harvest underrepresented)
        n_stages = cfg.get("n_stages", 6)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 2e-4),
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.get("lr", 2e-4),
            epochs=cfg.get("epochs", 60),
            steps_per_epoch=1,          # updated after dataset is known
        )

        ckpt_dir = Path(cfg.get("checkpoint_dir", "models/weights"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir  = ckpt_dir
        self.n_stages  = n_stages

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)          # (B, n_stages)
                loss   = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / max(len(loader), 1), macro_f1

    def train(self, sequences_path: str, labels_path: str):
        dataset    = PhenologyDataset(sequences_path, labels_path)
        val_size   = max(1, int(0.15 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        bs = self.cfg.get("batch_size", 256)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)

        # Re-build scheduler now that steps_per_epoch is known
        epochs = self.cfg.get("epochs", 60)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.cfg.get("lr", 2e-4),
            epochs=epochs, steps_per_epoch=len(train_loader),
        )

        best_f1 = 0.0
        logger.info(f"[PhenologyTrain] {epochs} epochs | "
                    f"train={train_size} val={val_size} | bs={bs}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_f1 = self._run_epoch(train_loader, train=True)
            vl_loss, vl_f1 = self._run_epoch(val_loader,   train=False)

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss={tr_loss:.4f} F1={tr_f1:.4f} | "
                f"val loss={vl_loss:.4f} F1={vl_f1:.4f}"
            )

            if vl_f1 > best_f1:
                best_f1 = vl_f1
                ckpt = self.ckpt_dir / "phenology_transformer.pth"
                torch.save(self.model.state_dict(), ckpt)
                logger.info(f"[PhenologyTrain] ✓ Best model saved (F1={best_f1:.4f}) → {ckpt}")

        logger.info(f"[PhenologyTrain] Done. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/pipeline_config.yaml")
    parser.add_argument("--sequences",  default="data/processed/phenology_sequences.npy")
    parser.add_argument("--labels",     default="data/processed/phenology_labels.npy")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["models"]["phenology_mapper"]

    PhenologyTrainer(cfg, device=args.device).train(args.sequences, args.labels)
