"""
utils/sar.py
SAR-specific processing utilities:
  - Radiometric terrain correction helpers
  - Multi-temporal SAR coherence computation
  - Change detection between two SAR acquisitions
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from loguru import logger


def terrain_flattening(
    backscatter: np.ndarray,
    incidence_angle: np.ndarray,
) -> np.ndarray:
    """
    Simple radiometric terrain flattening via cosine correction.
    backscatter:     linear scale (H, W)
    incidence_angle: degrees (H, W)
    Returns corrected backscatter in linear scale.
    """
    cos_ref = np.cos(np.radians(35.0))   # reference incidence angle
    cos_local = np.cos(np.radians(np.clip(incidence_angle, 1, 89)))
    corrected = backscatter * (cos_ref / (cos_local + 1e-9))
    return corrected.astype(np.float32)


def multi_temporal_filter(
    stack: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Multi-temporal speckle filter: applies spatial mean across T acquisitions.
    stack: (T, H, W) — temporal SAR stack in linear scale
    Returns filtered stack (T, H, W).
    """
    out = np.zeros_like(stack)
    for t in range(stack.shape[0]):
        out[t] = uniform_filter(stack[t].astype(np.float64), size=window).astype(np.float32)
    return out


def coherence(
    slc1: np.ndarray,
    slc2: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Estimate InSAR coherence between two complex SLC images.
    slc1, slc2: complex64 (H, W)
    Returns coherence magnitude [0, 1] (H, W).
    """
    cross = slc1 * np.conj(slc2)
    cross_smooth = uniform_filter(cross.real, window) + \
                   1j * uniform_filter(cross.imag, window)
    pow1 = uniform_filter(np.abs(slc1) ** 2, window)
    pow2 = uniform_filter(np.abs(slc2) ** 2, window)
    coh  = np.abs(cross_smooth) / (np.sqrt(pow1 * pow2) + 1e-9)
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def change_detection(
    before: np.ndarray,
    after: np.ndarray,
    threshold_db: float = 3.0,
) -> np.ndarray:
    """
    Detect significant backscatter changes between two SAR acquisitions.
    Both arrays in dB (H, W).
    Returns boolean change mask — True where |after - before| > threshold_db.
    """
    diff = np.abs(after - before)
    logger.debug(f"[SAR] Change map: {(diff > threshold_db).mean():.2%} pixels changed.")
    return diff > threshold_db


def vv_vh_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Compute VV/VH ratio in dB. Sensitive to surface roughness and moisture content.
    Returns (H, W) float32.
    """
    return (vv_db - vh_db).astype(np.float32)


def rfdi(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Radar Forest Degradation Index — sensitive to vegetation structure.
    RFDI = (VV - VH) / (VV + VH)   [linear scale]
    """
    return ((vv - vh) / (vv + vh + 1e-9)).astype(np.float32)
