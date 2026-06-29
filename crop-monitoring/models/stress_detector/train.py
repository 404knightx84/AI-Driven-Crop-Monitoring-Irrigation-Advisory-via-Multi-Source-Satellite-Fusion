"""
models/stress_detector/train.py
Training script for the dual-stream fusion CNN stress detector.

Dataset pairs:
  optical_dir/  *.npy  — (H, W, C_opt) optical + index stacks
  sar_dir/      *.npy  — (H, W, C_sar) SAR stacks
  masks_dir/    *.npy  — (H, W) stress labels  {0, 1, 2}
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
from sklearn.metrics import f1_score, confusion_matrix
from loguru import logger

from models.stress_detector.fusion_cnn import StressDetector, build_model


class StressDataset(Dataset):
    def __init__(self, optical_dir: str, sar_dir: str, masks_dir: str):
        self.opt_paths   = sorted(Path(optical_dir).glob("*.npy"))
        self.sar_paths   = sorted(Path(sar_dir).glob("*.npy"))
        self.mask_paths  = sorted(Path(masks_dir).glob("*.npy"))
        assert len(self.opt_paths) == len(self.sar_paths) == len(self.mask_paths), \
            "Optical / SAR / mask file counts must match."
        logger.info(f"[StressDataset] {len(self.opt_paths)} samples")

    def __len__(self):
        return len(self.opt_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        opt  = np.load(self.opt_paths[idx]).astype(np.float32)    # (H, W, C_opt)
        sar  = np.load(self.sar_paths[idx]).astype(np.float32)    # (H, W, C_sar)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)     # (H, W)

        opt_t  = torch.from_numpy(opt).permute(2, 0, 1)           # (C, H, W)
        sar_t  = torch.from_numpy(sar).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)
        return opt_t, sar_t, mask_t


class StressTrainer:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = build_model(cfg).to(self.device)

        # Weighted CE: severe stress (class 2) is underrepresented
        weights = torch.tensor([1.0, 2.0, 4.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.get("lr", 1e-4))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )

        ckpt_dir = Path(cfg.get("checkpoint_dir", "models/weights"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for opt, sar, mask in loader:
                opt, sar, mask = opt.to(self.device), sar.to(self.device), mask.to(self.device)
                logits = self.model(opt, sar)          # (B, 3, H, W)
                loss   = self.criterion(logits, mask)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                preds = logits.argmax(dim=1)
                total_loss  += loss.item()
                all_preds.extend(preds.cpu().numpy().ravel().tolist())
                all_labels.extend(mask.cpu().numpy().ravel().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / max(len(loader), 1), macro_f1

    def train(self, optical_dir: str, sar_dir: str, masks_dir: str):
        dataset    = StressDataset(optical_dir, sar_dir, masks_dir)
        val_size   = max(1, int(0.15 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        bs = self.cfg.get("batch_size", 8)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)

        epochs  = self.cfg.get("epochs", 40)
        best_f1 = 0.0
        logger.info(f"[StressTrain] {epochs} epochs | train={train_size} val={val_size}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_f1 = self._run_epoch(train_loader, train=True)
            vl_loss, vl_f1 = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(vl_f1)

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss={tr_loss:.4f} F1={tr_f1:.4f} | "
                f"val loss={vl_loss:.4f} F1={vl_f1:.4f}"
            )

            if vl_f1 > best_f1:
                best_f1 = vl_f1
                ckpt = self.ckpt_dir / "stress_cnn.pth"
                torch.save(self.model.state_dict(), ckpt)
                logger.info(f"[StressTrain] ✓ Best model saved (F1={best_f1:.4f}) → {ckpt}")

        logger.info(f"[StressTrain] Done. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/pipeline_config.yaml")
    parser.add_argument("--optical", default="data/processed/stress_optical")
    parser.add_argument("--sar",     default="data/processed/stress_sar")
    parser.add_argument("--masks",   default="data/processed/stress_masks")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["models"]["stress_detector"]

    StressTrainer(cfg, device=args.device).train(args.optical, args.sar, args.masks)
