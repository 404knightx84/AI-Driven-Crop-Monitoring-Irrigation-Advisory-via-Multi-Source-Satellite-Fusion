# AI-Driven-Crop-Monitoring-Irrigation-Advisory-via-Multi-Source-Satellite-Fusion
Automated crop type classification, phenological stage mapping, and moisture stress detection across growth stages — combining optical and microwave SAR satellite data with deep learning.


# AI-Driven Crop Monitoring & Irrigation Advisory via Multi-Source Satellite Fusion

An enterprise-grade, deep learning pipeline for automated crop type classification, phenological stage mapping, and moisture stress detection. This system eliminates optical cloud-cover limitations by fusing Sentinel-1 (SAR) and Sentinel-2 (Optical) data to deliver high-frequency, actionable irrigation insights throughout the agricultural lifecycle.

---

## 🚀 Key Features

*   **Multi-Modal Satellite Fusion**: Coregisters and fuses Optical (Sentinel-2 MSI) and Microwave (Sentinel-1 SAR VV/VH) data to maintain operational visibility despite 100% cloud cover.
*   **Automated Crop Classification**: Implements pixel-based and object-based deep learning models to identify crop types early in the season.
*   **Dynamic Phenological Mapping**: Tracks crop growth stages (e.g., emergence, vegetative, flowering, senescence) using time-series analysis.
*   **Moisture Stress & Irrigation Risk Analysis**: Detects early-stage canopy water stress by combining SAR backscatter anomalies with optical indices (NDWI, MSI) to generate automated irrigation advisories.

---

## 🛠️ Architecture & Data Pipeline

The pipeline processes raw spatial data into analysis-ready data tensors, passes them through a deep learning core, and outputs localized irrigation recommendations.



### 1. Data Ingestion & Preprocessing
*   **SAR Pipeline**: Radiometric calibration, terrain correction (using DEM), speckle filtering (Lee filter), and conversion to decibel (dB) scale.
*   **Optical Pipeline**: Atmospheric correction, cloud masking (SCL layer), and generation of vegetation indices (NDVI, NDWI, EVI).
*   **Fusion Layer**: Spatial resampling to a uniform 10m grid and temporal interpolation to handle irregular satellite revisit cycles.

### 2. Deep Learning Core
*   **Classification & Phenology**: Time-Series Bidirectional LSTMs (Bi-LSTM) and Temporal Convolutional Networks (TCN) extract temporal phenological signatures.
*   **Spatial Segmentation**: A modified 3D U-Net handles pixel-level crop boundary and type isolation across temporal stacks.

---

## 💻 Tech Stack

*   **Earth Observation**: GeoPandas, Rasterio, GDAL, SentinelHub API, Google Earth Engine (GEE)
*   **Deep Learning**: PyTorch, PyTorch Lightning, Hugging Face Accelerate
*   **Data Science**: NumPy, SciPy, Pandas, Scikit-Learn
*   **Visualizations**: Matplotlib, Folium, Planetary Computer tools

---

## 📦 Directory Structure

```text
├── config/                  # Configuration files for data pipelines and models
├── data/
│   ├── raw/                 # Raw downloaded Sentinel-1/2 products
│   └── processed/           # Co-registered, cloud-free fused tensors
├── src/
│   ├── ingestion/           # Scripts to query and download satellite imagery
│   ├── preprocessing/       # SAR calibration, optical indices computation
│   ├── models/              # PyTorch architectures (U-Net, Bi-LSTM, TCN)
│   ├── evaluation/          # Confusion matrices, F1-scores, validation scripts
│   └── advisory/            # Water stress logic and automated report generation
├── notebooks/               # Jupyter notebooks for exploratory data analysis (EDA)
├── main.py                  # Main entry point to run the execution pipeline
└── README.md                # Project documentation
```

---

## ⚙️ Getting Started

### Prerequisites
*   Python 3.10 or higher
*   CUDA-compatible GPU (Highly recommended for deep learning training)
*   Copernicus Data Space Ecosystem or SentinelHub API credentials

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd AI-Driven-Crop-Monitoring-Irrigation-Advisory
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Open .env and add your SentinelHub/Copernicus API credentials
   ```

### Running the Pipeline

To execute the entire pipeline from data downloading to generating an irrigation advisory for a specific Area of Interest (AOI):

```bash
python main.py --aoi path/to/geojson.json --start_date 2026-03-01 --end_date 2026-06-28
```

---

## 📈 Performance & Evaluation metrics

*   **Crop Classification**: Target Macro F1-Score of `> 0.88` across dominant regional crop classes.
*   **Phenological Mapping**: Mean Absolute Error (MAE) of `< 6 days` for key transition states compared to ground-truth observations.
*   **Stress Detection**: Validated against volumetric soil moisture probes using Root Mean Squared Error (RMSE).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
