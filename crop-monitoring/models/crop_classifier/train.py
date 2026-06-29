"""
models/crop_classifier/train.py
Training script for the U-Net + LSTM crop type classifier.
Supports multi-GPU via DataParallel, mixed-precision, and checkpoint saving.

Expected dataset structure:
  data/processed/
    images/          *.npy  — (T, H, W, C) fused tensors
    masks/           *.npy  — (H, W) integer label maps
"""

import os
import time
import yaml
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

from models.crop_classifier.unet_lstm import CropClassifier, build_model


# ── Dataset ──────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    """
    Loads paired (fused_sequence, crop_mask) numpy files.
    fused: (T, H, W, C)  →  model input
    mask:  (H, W)         →  segmentation target
    """

    def __init__(self, images_dir: str, masks_dir: str, sequence_length: int = 6):
        self.image_paths = sorted(Path(images_dir).glob("*.npy"))
        self.mask_paths  = sorted(Path(masks_dir).glob("*.npy"))
        assert len(self.image_paths) == len(self.mask_paths), \
            "Mismatch between image and mask counts."
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img  = np.load(self.image_paths[idx])   # (T, H, W, C)
        mask = np.load(self.mask_paths[idx])    # (H, W)

        # Pad or trim to fixed sequence length
        T = img.shape[0]
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, *img.shape[1:]), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=0)
        else:
            img = img[: self.seq_len]

        # (T, C, H, W) for PyTorch
        img_t  = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss — down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Metrics ──────────────────────────────────────────────────────────────────

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> float:
    ious = []
    pred_np   = preds.cpu().numpy().ravel()
    target_np = targets.cpu().numpy().ravel()
    for cls in range(1, n_classes):   # skip background
        tp = ((pred_np == cls) & (target_np == cls)).sum()
        fp = ((pred_np == cls) & (target_np != cls)).sum()
        fn = ((pred_np != cls) & (target_np == cls)).sum()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = build_model(cfg).to(self.device)

        if torch.cuda.device_count() > 1:
            logger.info(f"[Train] Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.criterion = FocalLoss(gamma=2.0, ignore_index=0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.get("lr", 1e-4),
                                     weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.get("epochs", 50)
        )
        self.scaler    = GradScaler()
        self.n_classes = cfg.get("n_classes", 8)

        ckpt_dir = Path(cfg.get("checkpoint_dir", "models/weights"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss, total_iou, n = 0.0, 0.0, 0

        with torch.set_grad_enabled(train):
            for imgs, masks in loader:
                imgs  = imgs.to(self.device)    # (B, T, C, H, W)
                masks = masks.to(self.device)   # (B, H, W)

                with autocast():
                    logits = self.model(imgs)   # (B, n_classes, H, W)
                    loss   = self.criterion(logits, masks)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                preds = logits.argmax(dim=1)
                total_loss += loss.item()
                total_iou  += mean_iou(preds, masks, self.n_classes)
                n += 1

        return total_loss / max(n, 1), total_iou / max(n, 1)

    def train(self, images_dir: str, masks_dir: str):
        dataset = CropDataset(images_dir, masks_dir, self.cfg.get("sequence_length", 6))
        val_size   = max(1, int(0.15 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.cfg.get("batch_size", 4),
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.get("batch_size", 4),
                                  shuffle=False, num_workers=2, pin_memory=True)

        best_iou = 0.0
        epochs   = self.cfg.get("epochs", 50)
        logger.info(f"[Train] Starting: {epochs} epochs | "
                    f"train={train_size} val={val_size}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_iou = self._run_epoch(train_loader, train=True)
            vl_loss, vl_iou = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss={tr_loss:.4f} mIoU={tr_iou:.4f} | "
                f"val loss={vl_loss:.4f} mIoU={vl_iou:.4f} | "
                f"{elapsed:.1f}s"
            )

            if vl_iou > best_iou:
                best_iou = vl_iou
                ckpt = self.ckpt_dir / "crop_unet_lstm.pth"
                state = self.model.module.state_dict() \
                    if isinstance(self.model, nn.DataParallel) \
                    else self.model.state_dict()
                torch.save(state, ckpt)
                logger.info(f"[Train] ✓ Best model saved (mIoU={best_iou:.4f}) → {ckpt}")

        logger.info(f"[Train] Done. Best val mIoU: {best_iou:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--images", default="data/processed/images")
    parser.add_argument("--masks",  default="data/processed/masks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr",     type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["models"]["crop_classifier"]

    if args.epochs: cfg["epochs"] = args.epochs
    if args.lr:     cfg["lr"]     = args.lr

    trainer = Trainer(cfg, device=args.device)
    trainer.train(args.images, args.masks)
