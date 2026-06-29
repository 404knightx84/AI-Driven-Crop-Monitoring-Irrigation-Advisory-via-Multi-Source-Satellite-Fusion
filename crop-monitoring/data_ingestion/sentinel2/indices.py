"""
data_ingestion/sentinel2/indices.py
Compute spectral indices from Sentinel-2 band stacks.
Assumes band order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
Index:                 0    1    2    3    4    5    6    7    8    9
"""

import numpy as np

EPS = 1e-9


def _norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def ndvi(bands: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index — (NIR - Red) / (NIR + Red)."""
    return _norm(bands[..., 6], bands[..., 2])   # B08, B04


def evi(bands: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index."""
    nir, red, blue = bands[..., 6], bands[..., 2], bands[..., 0]
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)


def ndwi(bands: np.ndarray) -> np.ndarray:
    """Normalised Difference Water Index (Gao) — sensitive to canopy water."""
    return _norm(bands[..., 6], bands[..., 8])   # B08, B11


def ndre(bands: np.ndarray) -> np.ndarray:
    """Normalised Difference Red Edge — early stress indicator."""
    return _norm(bands[..., 6], bands[..., 3])   # B08, B05


def savi(bands: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index."""
    nir, red = bands[..., 6], bands[..., 2]
    return (1 + L) * (nir - red) / (nir + red + L + EPS)


def lswi(bands: np.ndarray) -> np.ndarray:
    """Land Surface Water Index — NIR vs SWIR, tracks leaf water content."""
    return _norm(bands[..., 6], bands[..., 9])   # B08, B12


def compute_all(bands: np.ndarray) -> np.ndarray:
    """Stack all indices along last axis. Returns (H, W, 6) float32 array."""
    indices = [ndvi, evi, ndwi, ndre, savi, lswi]
    return np.stack([fn(bands).astype(np.float32) for fn in indices], axis=-1)
