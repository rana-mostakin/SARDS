# SARDS — Satellite-Based Activity Risk Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A production-grade satellite image analysis pipeline for detecting urban change,
environmental risk, and temporal anomalies — focused on Dhaka, Bangladesh.**

[Features](#features) • [Live Demo](#live-demo) • [Architecture](#system-architecture) •
[Installation](#installation) • [Usage](#usage) • [Data Sources](#data-sources)

</div>

---

## Live Demo

The dashboard is publicly deployed and requires no installation.

**URL:** [https://rana-mostakin-sards-qykwz6k9stxtcu8aapphw8x.streamlit.app](https://rana-mostakin-sards-qykwz6k9stxtcu8aapphw8x.streamlit.app)

Steps to use the live dashboard:
1. Open the URL above in any browser
2. Click **Run Full Analysis** in the sidebar
3. Explore the five analysis tabs: Dataset Overview, Change Detection, Heatmaps, Time-Series, Risk Score
4. Download heatmaps, CSVs, and the auto-generated risk report

---

## Problem Statement

Rapid, unplanned urbanisation in developing-world megacities is one of the most pressing
environmental challenges of our time. **Dhaka, Bangladesh** — population 21 million and
growing — exemplifies this: over the past two decades the city has expanded at the expense
of wetlands, river floodplains, and agricultural land, dramatically increasing flood risk,
urban heat island intensity, and biodiversity loss.

Traditional monitoring relies on periodic ground surveys that are expensive, slow, and
subject to access constraints. **SARDS** demonstrates how freely available satellite
imagery — combined with computer vision and geospatial analytics — can deliver objective,
scalable, repeatable change monitoring at a fraction of the cost.

---

## Why Dhaka?

| Factor | Detail |
|---|---|
| **Urban growth** | Population density > 44,000 / km² — one of the highest on Earth |
| **River encroachment** | Buriganga & Turag rivers shrinking every decade |
| **Flood risk** | ~70% of Dhaka sits within 4 m of flood level |
| **Green cover loss** | Tree canopy down ~30% in 15 years (2008–2023) |
| **Night-light growth** | VIIRS night-light proxy shows 2× increase 2012–2022 |
| **Data availability** | Well-covered by Landsat 8/9, Sentinel-2 since 2013 |

---

## Features

### 1. Spatial Change Detection
- Three detection engines: **Absolute Difference**, **SSIM**, **Optical Flow**
- ECC-based image co-registration (corrects orbital drift)
- Morphological noise removal
- Region-labelled change masks with area statistics

### 2. Heatmap Visualisation
- Per-pair and cumulative change heatmaps
- Six selectable colormaps (jet, inferno, hot, viridis …)
- Transparent overlay on original imagery
- Comparison panel: before / after / heatmap side-by-side

### 3. Time-Series Analysis
- Five metrics extracted per frame: mean intensity, urban index,
  NDVI proxy, entropy, change percentage
- Linear trend analysis with R² and p-value
- Savitzky-Golay smoothing for trend lines
- CSV export of all metrics

### 4. Anomaly Detection
- Three methods: Z-score, IQR fence, Isolation Forest (ML)
- Per-year composite anomaly score (0–1)
- Annotated time-series charts with flagged years

### 5. Risk Scoring
- Weighted composite score from 4 evidence streams
- Four risk levels: LOW / MEDIUM / HIGH / CRITICAL
- Year-by-year risk breakdown
- Automatically generated plain-text assessment report

### 6. Streamlit Dashboard
- Upload custom imagery OR use the built-in Dhaka demo
- Interactive controls: detection method, colormap, threshold
- Five analysis tabs with live rendering
- Download heatmaps, CSVs, and reports

---

## System Architecture

```
+------------------------------------------------------------------+
|                         SARDS Pipeline                           |
|                                                                  |
|  +-------------+     +--------------+     +-------------------+  |
|  | Image Loader|---->| Preprocessor |---->| Change Detector   |  |
|  |             |     |              |     |                   |  |
|  | - Dhaka demo|     | - Resize     |     | - AbsDiff         |  |
|  | - User files|     | - CLAHE EQ   |     | - SSIM            |  |
|  | - Year index|     | - Gaussian   |     | - Optical Flow    |  |
|  +-------------+     |   blur       |     | - ECC alignment   |  |
|                      | - ECC align  |     +--------+----------+  |
|                      +--------------+              |             |
|                                                    v             |
|  +-------------+     +--------------+     +-------------------+  |
|  | Risk Scorer |<----| Anomaly Detect|<---| Heatmap Generator |  |
|  |             |     |              |     |                   |  |
|  | - Weighted  |     | - Z-score    |     | - Per-pair        |  |
|  |   composite |     | - IQR fence  |     | - Cumulative      |  |
|  | - 4 levels  |     | - Iso-Forest |     | - Comparison panel|  |
|  | - Report gen|     +--------------+     +-------------------+  |
|  +------+------+                                                 |
|         |          +----------------------------------+          |
|         +--------->| Time-Series Analyzer             |          |
|                    |                                  |          |
|                    | - 5 metrics per frame            |          |
|                    | - OLS trend regression           |          |
|                    | - Change progression             |          |
|                    +----------------------------------+          |
+------------------------------------------------------------------+
                              |
                              v
                  +-----------------------+
                  | Streamlit Dashboard   |
                  | - 5 interactive tabs  |
                  | - Upload / Demo mode  |
                  | - Live visualisations |
                  | - Report downloads    |
                  +-----------------------+
```

---

## Project Structure

```
SARDS/
|
+-- data/
|   +-- dhaka/
|       +-- 2018.jpg         <- Satellite images (demo or real)
|       +-- 2019.jpg
|       +-- 2020.jpg
|       +-- 2021.jpg
|       +-- 2022.jpg
|       +-- 2023.jpg
|       +-- 2024.jpg
|
+-- src/                     <- Core library modules
|   +-- __init__.py
|   +-- utils.py             <- Config, logging, I/O helpers
|   +-- image_loader.py      <- Dataset loading & SatelliteImage schema
|   +-- preprocessing.py     <- Resize, blur, CLAHE, ECC alignment
|   +-- change_detection.py  <- AbsDiff / SSIM / Optical Flow detection
|   +-- heatmap_generator.py <- Heatmap generation & composite panels
|   +-- time_series_analysis.py <- Metric extraction & trend analysis
|   +-- anomaly_detection.py <- Z-score / IQR / Isolation Forest
|   +-- risk_scoring.py      <- Composite risk score + report
|
+-- app/
|   +-- streamlit_app.py     <- Interactive web dashboard
|
+-- scripts/
|   +-- generate_demo_data.py <- Synthetic Dhaka demo dataset generator
|
+-- results/                  <- All generated outputs (auto-created)
|   +-- heatmaps/
|   +-- time_series/
|   +-- reports/
|
+-- config.yaml               <- Central configuration
+-- main.py                   <- Full pipeline CLI runner
+-- requirements.txt
+-- README.md
```

---

## Data Sources

### Option A — Demo Dataset (Recommended, No Account Required)

The included generator creates **realistic synthetic Dhaka satellite imagery**
that simulates true patterns: urban expansion, river encroachment, vegetation loss.

```bash
python scripts/generate_demo_data.py
```

### Option B — Real Landsat 8/9 Data (Free)

1. Create a free account at [USGS EarthExplorer](https://earthexplorer.usgs.gov)
2. Search for: `Path 136 / Row 044` (covers Dhaka)
3. Select: **Landsat Collection 2 Level-2** product
4. Download the Band 4 (Red), Band 3 (Green), Band 2 (Blue) GeoTIFF files
5. Combine bands and crop to the Dhaka bounding box:
   ```bash
   # Example using GDAL (install with: pip install GDAL)
   gdal_merge.py -separate -o dhaka_RGB.tif B4.TIF B3.TIF B2.TIF
   gdal_translate -projwin 90.25 24.00 90.60 23.60 dhaka_RGB.tif data/dhaka/2024.tif
   ```

### Option C — Sentinel-2 (10m Resolution, Free)

1. Create a free account at [Copernicus Open Access Hub](https://scihub.copernicus.eu)
2. Search for tile `T45QVF` (Dhaka area)
3. Select Level-2A (surface reflectance) products
4. Download Bands 4, 3, 2 (Red, Green, Blue) at 10m
5. Rename per year and place in `data/dhaka/`

### Option D — Google Earth Engine (GEE)

Requires a GEE account. The following GEE JavaScript snippet downloads
a true-colour composite for Dhaka for a given year:

```javascript
var dhaka = ee.Geometry.Rectangle([90.25, 23.60, 90.60, 24.00]);
var composite = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(dhaka)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .median()
  .select(['B4', 'B3', 'B2'])
  .visualize({min: 0, max: 3000});
Export.image.toDrive({image: composite, region: dhaka,
  scale: 10, description: 'dhaka_2023'});
```

---

## Installation

### Prerequisites
- Python 3.9 or newer
- Git
- 4 GB RAM minimum (8 GB recommended)
- Windows 10 / macOS 12+ / Ubuntu 20.04+

### Step-by-Step Setup

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SARDS.git
cd SARDS
```

#### 2. Create a Python virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `rasterio` may require additional system libraries.
> - **Ubuntu:** `sudo apt-get install libgdal-dev`
> - **macOS:** `brew install gdal`
> - **Windows:** Use the pre-built wheel from
>   [Christoph Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

#### 4. Generate the demo dataset
```bash
python scripts/generate_demo_data.py
```

This creates `data/dhaka/2018.jpg ... 2024.jpg` instantly.

---

## Usage

### Run the full pipeline (CLI)

```bash
# Run all 9 steps and save results to results/
python main.py

# Quiet mode (no console output)
python main.py --quiet

# Custom config
python main.py --config my_config.yaml
```

### Launch the interactive Streamlit dashboard (local)

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Dashboard workflow:**
1. Click **Run Full Analysis** in the sidebar
2. Explore the five analysis tabs:
   - **Dataset Overview** — image timeline grid
   - **Change Detection** — before/after pairs with overlay
   - **Heatmaps** — per-pair and cumulative heat maps
   - **Time-Series** — metric trends and change progression
   - **Risk Score** — gauge, year breakdown, text report

### Use individual modules in your own code

```python
from src.image_loader     import ImageLoader
from src.preprocessing    import Preprocessor
from src.change_detection import ChangeDetector
from src.heatmap_generator import HeatmapGenerator

# Load
loader  = ImageLoader()
dataset = loader.load_dhaka_timeline()

# Pre-process
prep    = Preprocessor()
proc_ds = prep.process_dataset(dataset)

# Detect change between 2020 and 2024
img_2020 = proc_ds.get(2020)
img_2024 = proc_ds.get(2024)
detector = ChangeDetector()
result   = detector.detect(img_2020, img_2024)

print(f"Change: {result.change_percentage:.2f}%")
print(f"Regions: {result.stats['contour_count']}")
```

---

## Sample Outputs

### Change Detection
- Binary change mask with highlighted regions
- Per-pair statistics: % changed pixels, number of regions, mean difference

### Heatmaps
- Gradient colour map from low (blue/purple) to high (red/yellow) change
- Cumulative map stacks all years into a single risk surface

### Time-Series Metrics
| Metric | 2018 | 2020 | 2022 | 2024 | Trend |
|---|---|---|---|---|---|
| Urban Index | 0.14 | 0.21 | 0.29 | 0.38 | Increasing |
| NDVI Proxy | 0.08 | 0.05 | 0.02 | -0.01 | Decreasing |
| Mean Intensity | 108 | 118 | 129 | 142 | Increasing |
| Entropy | 6.8 | 7.1 | 7.3 | 7.5 | Increasing |

### Risk Score
```
OVERALL RISK SCORE : 0.6812  ->  HIGH RISK

COMPONENT BREAKDOWN
  change_percentage    [||||||||||||........]  0.612
  heatmap_intensity    [||||||||||..........]  0.501
  temporal_trend       [||||||||||||||||....]  0.781
  anomaly_flag         [||||||||............]  0.425
```

---

## Future Improvements

| Feature | Description |
|---|---|
| **Real-time integration** | Connect to Copernicus Sentinel Hub API for daily imagery ingestion |
| **ML classification** | Train a CNN (U-Net) on labelled change maps for semantic segmentation |
| **NDVI/NDWI computation** | Use near-infrared bands from Landsat/Sentinel for proper vegetation and water indices |
| **3D risk surface** | Interactive 3D heatmap using Plotly or Deck.gl |
| **Alert system** | Email/Slack notifications when risk score crosses threshold |
| **Cloud deployment** | Docker + FastAPI backend, deploy to AWS/GCP with persistent storage |
| **Multi-city support** | Generalise pipeline to any city using GEE bounding box |
| **SAR integration** | Add Sentinel-1 SAR data for flood and subsidence detection |
| **Time-series prediction** | LSTM or Prophet model to forecast future risk scores |
| **PDF report generation** | Automated PDF report with embedded maps and charts |

---

## Configuration Reference

All pipeline parameters are controlled through `config.yaml`:

```yaml
preprocessing:
  target_size: [512, 512]        # Resize resolution
  gaussian_blur_kernel: 3        # Noise reduction kernel

change_detection:
  method: "absolute_diff"        # absolute_diff | ssim | optical_flow
  threshold: 25                  # Pixel diff threshold (0-255)

heatmap:
  colormap: "jet"                # Any matplotlib colormap
  alpha_overlay: 0.6             # Transparency (0=original, 1=heatmap)

risk_scoring:
  weights:
    change_percentage: 0.35      # Adjust evidence weights
    heatmap_intensity: 0.30
    temporal_trend: 0.20
    anomaly_flag: 0.15
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **NASA / USGS** — Landsat open data program
- **ESA / Copernicus** — Sentinel-2 open data program
- **Google Earth Engine** — cloud geospatial processing
- **OpenCV community** — image processing algorithms
- **Streamlit** — rapid ML app development framework

---

<div align="center">
Built for open geospatial science · SARDS v1.0.0
</div>