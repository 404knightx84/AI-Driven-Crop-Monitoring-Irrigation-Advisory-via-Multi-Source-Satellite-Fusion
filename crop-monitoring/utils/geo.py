"""
utils/geo.py
GeoTIFF read/write, reprojection, tiling, and visualization helpers.
"""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger


def read_geotiff(path: str) -> Tuple[np.ndarray, dict]:
    """Read all bands from a GeoTIFF. Returns (H, W, C) float32 and profile."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # (C, H, W)
        profile = src.profile
    return data.transpose(1, 2, 0), profile    # (H, W, C)


def write_geotiff(
    path: str,
    data: np.ndarray,         # (H, W, C) or (H, W)
    profile: dict,
    dtype: str = "float32",
):
    """Write array to GeoTIFF."""
    if data.ndim == 2:
        data = data[..., np.newaxis]
    H, W, C = data.shape
    profile.update(count=C, dtype=dtype, driver="GTiff")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for c in range(C):
            dst.write(data[..., c], c + 1)
    logger.info(f"[GeoTIFF] Written: {path}  shape={data.shape}")


def reproject_to_crs(
    data: np.ndarray,
    src_profile: dict,
    dst_crs: str,
    resolution: float = 10.0,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, dict]:
    """Reproject array to a target CRS at a given resolution."""
    if data.ndim == 2:
        data = data[..., np.newaxis]
    H, W, C = data.shape

    transform, width, height = calculate_default_transform(
        src_profile["crs"], dst_crs, W, H,
        left=src_profile["transform"].c,
        top=src_profile["transform"].f,
        right=src_profile["transform"].c + W * src_profile["transform"].a,
        bottom=src_profile["transform"].f + H * src_profile["transform"].e,
        resolution=resolution,
    )
    out_profile = src_profile.copy()
    out_profile.update(crs=dst_crs, transform=transform, width=width, height=height)

    out = np.zeros((height, width, C), dtype=np.float32)
    for c in range(C):
        reproject(
            source=data[..., c],
            destination=out[..., c],
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out, out_profile


def tile_array(
    arr: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Tile a (H, W, C) array into overlapping patches.
    Returns list of (tile, (y_offset, x_offset)).
    """
    H, W = arr.shape[:2]
    step = tile_size - overlap
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            tile = arr[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                # Pad to tile_size
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                if tile.ndim == 3:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))
                else:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)))
            tiles.append((tile, (y, x)))
    return tiles


# ── Visualization ─────────────────────────────────────────────────────────────

CROP_COLORS = {
    0: "#e0e0e0", 1: "#f5d76e", 2: "#5dade2",
    3: "#58d68d", 4: "#a569bd", 5: "#f0b27a",
    6: "#48c9b0", 7: "#aab7b8",
}
STRESS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
STAGE_COLORS  = {0: "#f7f9f9", 1: "#d5f5e3", 2: "#27ae60",
                 3: "#f1c40f", 4: "#e67e22", 5: "#784212"}


def plot_prediction_maps(
    crop_map: np.ndarray,
    stage_map: np.ndarray,
    stress_map: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot crop, phenology, and stress maps side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _make_cmap(color_dict):
        n = max(color_dict.keys()) + 1
        colors = [color_dict.get(i, "#ffffff") for i in range(n)]
        return mcolors.ListedColormap(colors)

    axes[0].imshow(crop_map,   cmap=_make_cmap(CROP_COLORS),   interpolation="nearest")
    axes[0].set_title("Crop Type Map")
    axes[0].axis("off")

    axes[1].imshow(stage_map,  cmap=_make_cmap(STAGE_COLORS),  interpolation="nearest")
    axes[1].set_title("Phenology Stage Map")
    axes[1].axis("off")

    axes[2].imshow(stress_map, cmap=_make_cmap(STRESS_COLORS), interpolation="nearest")
    axes[2].set_title("Moisture Stress Map")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[Viz] Saved prediction maps to {save_path}")
    else:
        plt.show()
    plt.close()
