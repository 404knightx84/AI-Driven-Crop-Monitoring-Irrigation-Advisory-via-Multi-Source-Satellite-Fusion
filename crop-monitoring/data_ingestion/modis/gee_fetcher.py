"""
data_ingestion/modis/gee_fetcher.py
Fetches MODIS MOD13Q1 (250m, 16-day) NDVI / EVI time-series
via the Google Earth Engine Python API.
Used to build a long temporal profile for phenology tracking.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import ee
from loguru import logger


def init_gee(service_account: str, key_file: str, project: str):
    """Authenticate GEE with a service account key file."""
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials, project=project)
    logger.info("[GEE] Earth Engine initialized.")


class MODISFetcher:
    """
    Pulls MODIS MOD13Q1 16-day composites as numpy arrays for a given AOI.
    Returns a time-ordered stack of (NDVI, EVI) for phenology modelling.
    """

    COLLECTION = "MODIS/061/MOD13Q1"
    BANDS      = ["NDVI", "EVI"]
    SCALE      = 250          # native resolution, metres
    SCALE_FACTOR = 0.0001     # MOD13Q1 integer → physical value

    def __init__(self, data_dir: str):
        self.cache_dir = Path(data_dir) / "raw" / "modis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _aoi_to_ee(self, aoi_wkt: str) -> ee.Geometry:
        """Convert WKT polygon to ee.Geometry."""
        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(aoi_wkt)
        coords = list(geom.exterior.coords)
        return ee.Geometry.Polygon(coords)

    def fetch_time_series(
        self,
        aoi_wkt: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fetch NDVI + EVI stack over the AOI for the given date range.

        Returns:
            ndarray of shape (T, H, W, 2)  — T time steps, 2 bands
            list of date strings, length T
        """
        aoi = self._aoi_to_ee(aoi_wkt)
        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(start_date, end_date)
            .filterBounds(aoi)
            .select(self.BANDS)
        )
        image_list = collection.toList(collection.size())
        size = int(collection.size().getInfo())
        logger.info(f"[MODIS] {size} composites found between {start_date} and {end_date}")

        arrays, dates = [], []
        for i in range(size):
            img = ee.Image(image_list.get(i))
            date_str = img.date().format("YYYY-MM-dd").getInfo()
            try:
                data = img.reduceRegion(
                    reducer=ee.Reducer.toList(),
                    geometry=aoi,
                    scale=self.SCALE,
                    maxPixels=1e8,
                ).getInfo()
                ndvi_vals = np.array(data["NDVI"], dtype=np.float32) * self.SCALE_FACTOR
                evi_vals  = np.array(data["EVI"],  dtype=np.float32) * self.SCALE_FACTOR
                # infer spatial shape from pixel count (approximate square)
                n = len(ndvi_vals)
                side = int(np.ceil(np.sqrt(n)))
                pad = side * side - n
                ndvi_arr = np.pad(ndvi_vals, (0, pad)).reshape(side, side)
                evi_arr  = np.pad(evi_vals,  (0, pad)).reshape(side, side)
                arrays.append(np.stack([ndvi_arr, evi_arr], axis=-1))
                dates.append(date_str)
            except Exception as e:
                logger.warning(f"[MODIS] Skipping composite {date_str}: {e}")

        if not arrays:
            raise ValueError("No MODIS composites retrieved.")

        stack = np.stack(arrays, axis=0)  # (T, H, W, 2)
        logger.info(f"[MODIS] Time series stack: {stack.shape}")
        return stack, dates

    def compute_phenology_signal(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute spatial-mean NDVI per time step — 1D signal for curve fitting.
        Input:  (T, H, W, 2)
        Output: (T,) float32
        """
        return time_series[:, :, :, 0].mean(axis=(1, 2)).astype(np.float32)
