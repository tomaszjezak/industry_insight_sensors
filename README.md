# C&D Debris Insights - Phase 1 MVP

AI-powered analysis of construction & demolition debris piles for volume estimation, material classification, and change detection.

## Overview

This prototype uses computer vision and deep learning to provide actionable insights from overhead images of C&D recycling yards:

- **Volume Estimation**: Monocular depth estimation (MiDaS) + segmentation to calculate pile volumes
- **Material Classification**: Color-based analysis to identify concrete, wood, metal, and mixed debris
- **Change Detection**: Temporal comparison to track pile growth/shrinkage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Sample Images

```bash
python src/scraper.py
```

This fetches ~20-30 images of construction debris from Bing Images into `data/rgb_images/`.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 to use the interactive dashboard.

## Project Structure

```
├── data/
│   └── rgb_images/          # Input images (scraped or uploaded)
├── src/
│   ├── scraper.py           # Image download script
│   ├── segmentation.py      # SAM-based pile segmentation
│   ├── depth.py             # MiDaS monocular depth estimation
│   ├── volume.py            # Volume calculation
│   ├── materials.py         # Material classification
│   ├── change_detection.py  # Temporal diff analysis
│   └── pipeline.py          # Main orchestrator
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Pipeline Output

```json
{
  "volume_m3": 12.5,
  "materials": {"concrete": 0.45, "wood": 0.30, "metal": 0.10, "mixed": 0.15},
  "pile_area_m2": 8.2,
  "avg_height_m": 1.5,
  "timestamp": "2026-01-21T14:30:00"
}
```

## Hardware Requirements

- GPU recommended for faster inference (CUDA-compatible)
- CPU-only mode works but slower (~10-30s per image)
- ~4GB RAM minimum

## Next Steps (Phase 2)

- Deploy on Qualcomm RB3 Gen 2 with Luxonis OAK-D Pro W camera
- Real-time streaming from pole-mounted sensors
- Pilot at EDCO Recovery & Transfer, San Diego
