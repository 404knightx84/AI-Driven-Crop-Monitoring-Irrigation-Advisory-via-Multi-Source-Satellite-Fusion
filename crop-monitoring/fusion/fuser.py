"""
fusion/fuser.py
Aligns Sentinel-2, Sentinel-1, and MODIS-derived features to a common
spatial grid and stacks them into a single multi-channel data cube.

Fusion strategy:
  - Reproject all sources to target CRS (default UTM)
  - Resample SAR and MODIS to Sentinel-2 native resolution (10m)
  - Stack channels: [S2 bands × T, SAR × T, MODIS indices × T]
  - Normalize each channel to [0, 1] using percentile clipping
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from loguru import logger


def _reproject_array(
    src_array: np.ndarray,
    src_crs: str,
    src_transform,
    dst_crs: str,
    dst_shape: Tuple[int, int],
    dst_transform,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject a 2D array from src_crs/transform to dst_crs/transform."""
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=src_array.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


def percentile_normalize(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Robust min-max normalization using percentile clipping."""
    p_lo = np.nanpercentile(arr, lo)
    p_hi = np.nanpercentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-9), 0.0, 1.0).astype(np.float32)


class MultiSourceFuser:
    """
    Fuses Sentinel-2, Sentinel-1, and MODIS arrays to a unified spatial grid.
    """

    def __init__(self, target_crs: str = "EPSG:32643", target_resolution: int = 10):
        self.target_crs = target_crs
        self.target_res  = target_resolution

    def align_to_reference(
        self,
        source_array: np.ndarray,
        source_profile: dict,
        reference_profile: dict,
    ) -> np.ndarray:
        """
        Align source_array to the spatial grid defined by reference_profile.
        Handles both single-band (H, W) and multi-band (H, W, C) inputs.
        """
        ref_h = reference_profile["height"]
        ref_w = reference_profile["width"]
        ref_transform = reference_profile["transform"]
        ref_crs = reference_profile["crs"]
        src_crs = source_profile["crs"]
        src_transform = source_profile["transform"]

        if source_array.ndim == 2:
            return _reproject_array(
                source_array, str(src_crs), src_transform,
                str(ref_crs), (ref_h, ref_w), ref_transform,
            )

        # Multi-band: reproject each channel independently
        channels = []
        for c in range(source_array.shape[-1]):
            channels.append(
                _reproject_array(
                    source_array[..., c], str(src_crs), src_transform,
                    str(ref_crs), (ref_h, ref_w), ref_transform,
                )
            )
        return np.stack(channels, axis=-1)

    def fuse(
        self,
        s2_sequence: List[np.ndarray],        # list of (H, W, C_s2) arrays, length T
        sar_sequence: List[np.ndarray],       # list of (H, W, C_sar) arrays, length T
        modis_sequence: np.ndarray,           # (T, H_m, W_m, 2)
        reference_profile: dict,
        s2_profile: dict,
        sar_profile: dict,
        modis_profile: dict,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Produce a fused tensor of shape (T, H, W, C_total).

        C_total = C_s2 + C_sar + 2 (MODIS NDVI + EVI)
        """
        T = len(s2_sequence)
        assert len(sar_sequence) == T, "S2 and SAR sequences must have equal length."

        fused_frames = []
        for t in range(T):
            s2  = self.align_to_reference(s2_sequence[t],  s2_profile,    reference_profile)
            sar = self.align_to_reference(sar_sequence[t], sar_profile,   reference_profile)
            mod = self.align_to_reference(modis_sequence[t], modis_profile, reference_profile)

            frame = np.concatenate([s2, sar, mod], axis=-1)  # (H, W, C_total)
            if normalize:
                channels = []
                for c in range(frame.shape[-1]):
                    channels.append(percentile_normalize(frame[..., c]))
                frame = np.stack(channels, axis=-1)
            fused_frames.append(frame)

        result = np.stack(fused_frames, axis=0)  # (T, H, W, C)
        logger.info(f"[Fuser] Fused tensor: {result.shape}  (T={T})")
        return result
