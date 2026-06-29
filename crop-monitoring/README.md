# AI-Driven Crop Monitoring & Irrigation Advisory
### Multi-Source Satellite Fusion · Deep Learning · Real-Time Advisory

Automated crop type classification, phenological stage mapping, and moisture stress detection across growth stages — combining Sentinel-2 optical, Sentinel-1 SAR microwave, and MODIS time-series data with deep learning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                    │
│  Sentinel-2 (optical)  │  Sentinel-1 SAR  │  MODIS (NDVI)  │
└────────────┬───────────┴────────┬──────────┴──────┬─────────┘
             │                   │                  │
             ▼                   ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-SOURCE FUSION LAYER                 │
│         Spatial alignment · Band stacking · Normalization   │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
┌───────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Crop Classifier  │ │ Phenology Mapper  │ │  Stress Detector │
│  (U-Net + LSTM)   │ │  (Transformer)    │ │  (CNN + SAR)     │
└────────┬──────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                     │                     │
         └─────────────────────▼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ADVISORY ENGINE   │
                    │  Irrigation rules + │
                    │  crop-stage logic   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Grafana Dashboard  │
                    │  + REST API output  │
                    └─────────────────────┘
```

## Quick Start

```bash
cp configs/.env.example configs/.env
# Fill in Earth Engine credentials, Copernicus Hub credentials
docker-compose up --build
```

API → http://localhost:8000/docs  
Dashboard → http://localhost:3000

## Project Structure

```
crop-monitoring/
├── data_ingestion/
│   ├── sentinel2/       # Optical band download + cloud masking
│   ├── sentinel1/       # SAR GRD preprocessing (Lee filter, terrain correction)
│   └── modis/           # NDVI/EVI time-series from MODIS MOD13Q1
├── fusion/              # Multi-source spatial alignment & feature stacking
├── models/
│   ├── crop_classifier/ # U-Net segmentation + LSTM temporal encoder
│   ├── phenology_mapper/# Transformer on NDVI time-series
│   └── stress_detector/ # CNN fusing optical + SAR for moisture stress
├── inference/           # Tile-based batch inference pipeline
├── advisory/            # Irrigation advisory rule engine
├── pipeline/            # Orchestrator + scheduler (Airflow DAGs)
├── utils/               # GeoTIFF I/O, reprojection, visualization
├── configs/             # YAML configs + .env.example
├── dashboards/          # Grafana JSON provisioning
├── notebooks/           # EDA and model training notebooks
└── tests/               # Unit + integration tests
```

## Key Dependencies

| Library | Purpose |
|---|---|
| `sentinelsat` | Copernicus Open Access Hub download |
| `earthengine-api` | Google Earth Engine (MODIS, cloud-free composites) |
| `rasterio` / `GDAL` | GeoTIFF I/O, reprojection, mosaicking |
| `torch` + `torchvision` | U-Net, CNN, Transformer models |
| `segmentation_models_pytorch` | Pre-built U-Net backbone |
| `pyproj` / `shapely` | CRS handling and AOI geometry |
| `numpy` / `xarray` | Array ops and multi-dim data cubes |
| `FastAPI` | REST API for advisory output |
| `Grafana` | Live monitoring dashboard |
