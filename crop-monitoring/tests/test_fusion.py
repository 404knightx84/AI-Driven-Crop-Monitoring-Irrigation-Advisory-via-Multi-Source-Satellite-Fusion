"""
tests/test_fusion.py — unit tests for multi-source fusion layer
"""

import numpy as np
import pytest
from fusion.fuser import MultiSourceFuser, percentile_normalize


def make_dummy_profile(h: int, w: int, crs: str = "EPSG:4326"):
    from rasterio.transform import from_bounds
    return {
        "height": h, "width": w,
        "crs": crs,
        "transform": from_bounds(73.5, 18.5, 74.5, 19.5, w, h),
    }


def test_percentile_normalize():
    arr = np.arange(100, dtype=np.float32)
    norm = percentile_normalize(arr)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.dtype == np.float32


def test_fuser_output_shape():
    """Fused output should be (T, H, W, C_s2+C_sar+C_modis)."""
    T, H, W = 3, 64, 64
    C_s2, C_sar, C_mod = 10, 3, 2

    s2_seq  = [np.random.rand(H, W, C_s2).astype(np.float32)  for _ in range(T)]
    sar_seq = [np.random.rand(H, W, C_sar).astype(np.float32) for _ in range(T)]
    modis   = np.random.rand(T, H, W, C_mod).astype(np.float32)

    ref_profile  = make_dummy_profile(H, W, "EPSG:32643")
    s2_profile   = make_dummy_profile(H, W, "EPSG:32643")
    sar_profile  = make_dummy_profile(H, W, "EPSG:32643")
    mod_profile  = make_dummy_profile(H, W, "EPSG:32643")

    fuser = MultiSourceFuser(target_crs="EPSG:32643")
    fused = fuser.fuse(s2_seq, sar_seq, modis, ref_profile, s2_profile, sar_profile, mod_profile)

    assert fused.shape == (T, H, W, C_s2 + C_sar + C_mod)
    assert fused.dtype == np.float32
    assert fused.min() >= 0.0
    assert fused.max() <= 1.0
