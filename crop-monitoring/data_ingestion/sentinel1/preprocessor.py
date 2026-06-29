"""
data_ingestion/sentinel1/preprocessor.py
Downloads and preprocesses Sentinel-1 GRD scenes:
  1. Download via sentinelsat
  2. Apply Lee speckle filter
  3. Terrain correction (range-Doppler)
  4. dB conversion and normalization
  5. Backscatter ratio (VV/VH) as moisture proxy
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter, variance as ndimage_variance
from sentinelsat import SentinelAPI
from loguru import logger


def lee_filter(image: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Lee adaptive speckle filter.
    Reduces SAR multiplicative noise while preserving edges.
    """
    img_mean = uniform_filter(image.astype(np.float64), size)
    img_sqr_mean = uniform_filter(image.astype(np.float64) ** 2, size)
    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = ndimage_variance(image)
    img_weights = img_variance / (img_variance + overall_variance + 1e-9)
    return (img_mean + img_weights * (image - img_mean)).astype(np.float32)


def to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear backscatter to dB. Clips negative to near-zero first."""
    return 10 * np.log10(np.clip(linear, 1e-9, None))


def normalize_sar(arr: np.ndarray, vmin: float = -25.0, vmax: float = 0.0) -> np.ndarray:
    """Clip and min-max normalize dB values to [0, 1]."""
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def backscatter_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    VV/VH ratio in dB. Sensitive to surface roughness and moisture.
    Higher values → rougher/wetter surface.
    """
    return vv - vh   # subtraction in dB = division in linear


class Sentinel1Preprocessor:
    def __init__(self, user: str, password: str, data_dir: str, filter_size: int = 7):
        self.api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")
        self.data_dir = Path(data_dir) / "raw" / "sentinel1"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.filter_size = filter_size

    def search(self, aoi_wkt: str, start_date, end_date, orbit: str = "ASCENDING"):
        products = self.api.query(
            area=aoi_wkt,
            date=(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")),
            platformname="Sentinel-1",
            producttype="GRD",
            orbitdirection=orbit,
        )
        logger.info(f"[S1] Found {len(products)} GRD scenes")
        return self.api.to_geodataframe(products)

    def load_polarization(self, scene_path: Path, pol: str) -> Tuple[np.ndarray, dict]:
        """Load a single polarization band (VV or VH) from a GRD TIFF."""
        pattern = f"**/*{pol}*.tiff"
        matches = list(scene_path.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Polarization {pol} not found in {scene_path}")
        with rasterio.open(matches[0]) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile
        return data, profile

    def preprocess(self, scene_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Full preprocessing chain:
        load → lee filter → dB convert → normalize → stack [VV, VH, ratio]
        Returns (H, W, 3) float32 array.
        """
        vv_raw, profile = self.load_polarization(scene_path, "VV")
        vh_raw, _       = self.load_polarization(scene_path, "VH")

        vv_filt = lee_filter(vv_raw, self.filter_size)
        vh_filt = lee_filter(vh_raw, self.filter_size)

        vv_db = to_db(vv_filt)
        vh_db = to_db(vh_filt)

        vv_norm = normalize_sar(vv_db)
        vh_norm = normalize_sar(vh_db)
        ratio   = normalize_sar(backscatter_ratio(vv_db, vh_db), vmin=-5, vmax=5)

        stack = np.stack([vv_norm, vh_norm, ratio], axis=-1)  # (H, W, 3)
        logger.info(f"[S1] Preprocessed: {stack.shape}")
        return stack, profile
