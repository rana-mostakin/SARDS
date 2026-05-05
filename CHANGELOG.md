# Changelog

All notable changes to SARDS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024-12-01

### Initial Release

#### Added
- **`src/image_loader.py`** — `SatelliteImage` & `ImageDataset` schema with
  year-indexed multi-date loading and directory scanning
- **`src/preprocessing.py`** — Resize, CLAHE histogram equalisation, Gaussian
  blur, ECC-based image co-registration
- **`src/change_detection.py`** — Three detection engines: Absolute Difference,
  SSIM, Optical Flow; morphological noise removal; annotated change overlays
- **`src/heatmap_generator.py`** — Per-pair and cumulative heatmaps with six
  colormap options; comparison panel figure generation
- **`src/time_series_analysis.py`** — Five metrics per frame (mean intensity,
  urban index, NDVI proxy, entropy, change %); OLS trend regression;
  Savitzky-Golay smoothing
- **`src/anomaly_detection.py`** — Z-score, IQR fence, and Isolation Forest
  anomaly detectors; composite per-year anomaly score
- **`src/risk_scoring.py`** — Weighted 4-component composite risk score;
  four risk levels (LOW/MEDIUM/HIGH/CRITICAL); auto-generated text report
- **`app/streamlit_app.py`** — Full interactive dashboard with 5 tabs,
  demo/upload modes, live visualisations, and download buttons
- **`scripts/generate_demo_data.py`** — Synthetic Dhaka satellite image
  generator (2018-2024) with urban sprawl, river shift, vegetation decline
- **`main.py`** — 9-step CLI pipeline runner with verbose/quiet modes
- **`config.yaml`** — Centralised configuration for all pipeline parameters
- **`tests/test_pipeline.py`** — 40+ unit tests covering all modules
- **`.github/workflows/ci.yml`** — 4-job CI pipeline (lint, test, smoke, import)
- **`README.md`** — Comprehensive portfolio documentation

#### Infrastructure
- GitHub Actions CI/CD (lint + pytest + pipeline smoke test)
- `.gitignore` for Python, OS, IDE, and generated data files
- MIT License
- Issue templates (bug report + feature request)
- `CONTRIBUTING.md` guide

---

## [Unreleased]

### Planned
- Real Sentinel-2 API integration via `sentinelsat`
- Google Earth Engine data connector
- NDWI (water index) and NDBI (built-up index) computation
- PDF report generation via `fpdf2`
- Docker containerisation
- Streamlit Cloud deployment workflow
- ML-based change classification (U-Net)
- Time-series forecasting (LSTM / Prophet)