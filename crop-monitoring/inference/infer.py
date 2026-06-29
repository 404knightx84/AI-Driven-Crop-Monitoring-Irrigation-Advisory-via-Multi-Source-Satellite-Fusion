"""
inference/infer.py
Tile-based batch inference over large AOI rasters.
Loads all three models, slides a window across the fused tensor,
and assembles full-scene prediction maps.
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from models.crop_classifier.unet_lstm  import CropClassifier, build_model as build_crop
from models.phenology_mapper.transformer import PhenologyTransformer, build_model as build_pheno
from models.stress_detector.fusion_cnn   import StressDetector, build_model as build_stress


STRESS_LABELS = ["no_stress", "mild_stress", "severe_stress"]
CROP_LABELS   = ["background", "wheat", "rice", "maize",
                 "sugarcane", "cotton", "soybean", "other"]
STAGE_LABELS  = ["pre_sowing", "germination", "vegetative",
                 "reproductive", "maturation", "harvest_fallow"]


class InferencePipeline:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)

        logger.info("[Inference] Loading models...")
        self.crop_model  = self._load(build_crop(cfg["crop_classifier"]),
                                      cfg["crop_classifier"]["weights"])
        self.pheno_model = self._load(build_pheno(cfg["phenology_mapper"]),
                                      cfg["phenology_mapper"]["weights"])
        self.stress_model = self._load(build_stress(cfg["stress_detector"]),
                                       cfg["stress_detector"]["weights"])
        logger.info("[Inference] All models loaded.")

    def _load(self, model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
        model = model.to(self.device)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state)
            logger.info(f"[Inference] Loaded weights: {weights_path}")
        else:
            logger.warning(f"[Inference] Weights not found ({weights_path}), using random init.")
        model.eval()
        return model

    @torch.no_grad()
    def predict_crop(
        self,
        fused_sequence: np.ndarray,   # (T, H, W, C)
        tile_size: int = 256,
        overlap: int = 32,
    ) -> np.ndarray:
        """
        Sliding-window crop classification.
        Returns (H, W) integer label map.
        """
        T, H, W, C = fused_sequence.shape
        output = np.zeros((H, W), dtype=np.int64)
        count  = np.zeros((H, W), dtype=np.int32)
        logit_accum = np.zeros((H, W, len(CROP_LABELS)), dtype=np.float32)

        step = tile_size - overlap
        ys = list(range(0, H - tile_size + 1, step)) + [max(0, H - tile_size)]
        xs = list(range(0, W - tile_size + 1, step)) + [max(0, W - tile_size)]

        for y in set(ys):
            for x in set(xs):
                tile = fused_sequence[:, y:y+tile_size, x:x+tile_size, :]  # (T, th, tw, C)
                t = torch.from_numpy(tile).permute(0, 3, 1, 2).unsqueeze(0).float()
                # shape: (1, T, C, th, tw)
                t = t.to(self.device)
                logits = self.crop_model(t)          # (1, n_classes, th, tw)
                logits_np = logits[0].permute(1, 2, 0).cpu().numpy()
                logit_accum[y:y+tile_size, x:x+tile_size] += logits_np
                count[y:y+tile_size, x:x+tile_size] += 1

        count = np.maximum(count, 1)
        averaged = logit_accum / count[..., None]
        return averaged.argmax(axis=-1)

    @torch.no_grad()
    def predict_phenology(
        self, modis_ts: np.ndarray   # (T, H, W, 2)
    ) -> np.ndarray:
        """
        Per-pixel phenology stage prediction.
        Returns (H, W) integer stage map.
        """
        T, H, W, _ = modis_ts.shape
        flat = modis_ts.reshape(T, H * W, 2).transpose(1, 0, 2)  # (H*W, T, 2)
        t = torch.from_numpy(flat).float().to(self.device)
        logits = self.pheno_model(t)                              # (H*W, n_stages)
        labels = logits.argmax(dim=-1).cpu().numpy()
        return labels.reshape(H, W)

    @torch.no_grad()
    def predict_stress(
        self,
        optical: np.ndarray,   # (H, W, C_opt)
        sar: np.ndarray,        # (H, W, C_sar)
        tile_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Moisture stress classification.
        Returns (H, W) label map and (H, W, 3) probability map.
        """
        H, W, _ = optical.shape
        opt_t = torch.from_numpy(optical).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        sar_t = torch.from_numpy(sar).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        logits = self.stress_model(opt_t, sar_t)   # (1, 3, H, W)
        probs  = F.softmax(logits, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        labels = probs.argmax(axis=-1)
        return labels, probs

    def run_full(
        self,
        fused_sequence: np.ndarray,
        modis_ts: np.ndarray,
        optical_latest: np.ndarray,
        sar_latest: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run all three models and return prediction maps.
        """
        logger.info("[Inference] Running crop classifier...")
        crop_map = self.predict_crop(fused_sequence)

        logger.info("[Inference] Running phenology mapper...")
        stage_map = self.predict_phenology(modis_ts)

        logger.info("[Inference] Running stress detector...")
        stress_map, stress_probs = self.predict_stress(optical_latest, sar_latest)

        return {
            "crop_map":    crop_map,      # (H, W) int
            "stage_map":   stage_map,     # (H, W) int
            "stress_map":  stress_map,    # (H, W) int
            "stress_probs": stress_probs, # (H, W, 3) float
        }
