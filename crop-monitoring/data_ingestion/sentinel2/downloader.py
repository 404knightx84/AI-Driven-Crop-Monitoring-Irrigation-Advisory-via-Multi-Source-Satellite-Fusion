"""
data_ingestion/sentinel2/downloader.py
Downloads Sentinel-2 L2A scenes from Copernicus Open Access Hub,
applies SCL-based cloud masking, and produces cloud-free composites.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.merge import merge
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from loguru import logger

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SCL_VALID = [4, 5]          # 4=vegetation, 5=bare soil — exclude cloud/shadow


class Sentinel2Downloader:
    def __init__(self, user: str, password: str, data_dir: str):
        self.api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")
        self.data_dir = Path(data_dir) / "raw" / "sentinel2"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        aoi_wkt: str,
        start_date: datetime,
        end_date: datetime,
        cloud_cover_max: int = 20,
    ):
        """Search for available Sentinel-2 L2A scenes."""
        products = self.api.query(
            area=aoi_wkt,
            date=(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")),
            platformname="Sentinel-2",
            processinglevel="Level-2A",
            cloudcoverpercentage=(0, cloud_cover_max),
        )
        logger.info(f"[S2] Found {len(products)} scenes")
        return self.api.to_geodataframe(products)

    def download(self, product_id: str) -> Path:
        """Download a single product by UUID."""
        logger.info(f"[S2] Downloading {product_id}")
        self.api.download(product_id, directory_path=str(self.data_dir))
        return self.data_dir / product_id

    def cloud_mask(self, scene_path: Path) -> np.ndarray:
        """
        Load SCL band and return boolean valid-pixel mask.
        SCL classes 4 (vegetation) and 5 (bare soil) are treated as valid.
        """
        scl_path = next(scene_path.glob("**/SCL_20m.jp2"), None)
        if scl_path is None:
            logger.warning(f"[S2] SCL band not found in {scene_path}")
            return None
        with rasterio.open(scl_path) as src:
            scl = src.read(1)
        mask = np.isin(scl, SCL_VALID)
        logger.debug(f"[S2] Valid pixel ratio: {mask.mean():.2%}")
        return mask

    def load_bands(
        self, scene_path: Path, bands: List[str] = S2_BANDS, resolution: int = 10
    ) -> Tuple[np.ndarray, dict]:
        """
        Load and stack requested bands at the given resolution.
        Returns (H, W, C) array and rasterio profile.
        """
        res_str = f"{resolution}m"
        stack = []
        profile = None
        for band in bands:
            pattern = f"**/*{band}_{res_str}.jp2"
            matches = list(scene_path.glob(pattern))
            if not matches:
                logger.warning(f"[S2] Band {band} not found at {res_str}")
                continue
            with rasterio.open(matches[0]) as src:
                stack.append(src.read(1).astype(np.float32))
                if profile is None:
                    profile = src.profile
        if not stack:
            raise FileNotFoundError(f"No bands found in {scene_path}")
        return np.stack(stack, axis=-1), profile  # (H, W, C)

    def composite(
        self,
        scene_paths: List[Path],
        bands: List[str] = S2_BANDS,
        resolution: int = 10,
    ) -> Tuple[np.ndarray, dict]:
        """
        Build a median cloud-free composite from multiple scenes.
        Pixels masked by SCL are excluded per scene before median aggregation.
        """
        all_arrays = []
        base_profile = None
        for sp in scene_paths:
            try:
                arr, profile = self.load_bands(sp, bands, resolution)
                mask = self.cloud_mask(sp)
                if mask is not None:
                    arr[~mask] = np.nan
                all_arrays.append(arr)
                if base_profile is None:
                    base_profile = profile
            except Exception as e:
                logger.warning(f"[S2] Skipping {sp}: {e}")

        if not all_arrays:
            raise ValueError("No valid scenes to composite")

        stack = np.stack(all_arrays, axis=0)           # (N, H, W, C)
        composite = np.nanmedian(stack, axis=0)        # (H, W, C)
        logger.info(f"[S2] Composite from {len(all_arrays)} scenes: {composite.shape}")
        return composite, base_profile
