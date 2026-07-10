# IrimidLab

**Geometric and Anthropometric Facial Analysis** — a professional computer vision tool for analyzing measurable facial characteristics.

## Features

- Detection of 478 facial landmarks (MediaPipe Face Landmarker)
- Image quality validation (resolution, lighting, tilt, obstructions)
- **Symmetry** analysis (overall, eyes, nose, mouth, jaw)
- **Facial proportion** calculation (thirds, widths, distances)
- Detailed analysis of **eyes**, **nose**, **lips**, **jaw**, and **chin**
- Composite **geometric harmony** index
- **Profile** analysis (optional lateral image)
- Configurable visualization layers
- Interactive dashboard with 5 tabs
- **PDF** report generation
- Export to **JSON**, **CSV**, and **Excel**
- Analysis history for temporal comparison

## Requirements

- Python 3.11+ (3.12+ recommended)
- Webcam not required (analysis via image upload)

## Installation

```bash
cd IrimidLab
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
python face_analyzer/scripts/download_model.py
```

The download script fetches the `face_landmarker.task` model (~3.6 MB) required for landmark detection. Alternatively, download it manually and place it in `face_analyzer/assets/face_landmarker.task`.

## Running

```bash
streamlit run face_analyzer/main.py
```

Or on Windows: `run.bat`

The app opens in your browser at `http://localhost:8501`.

## Project Structure

```
IrimidLab/
├── face_analyzer/
│   ├── main.py              # Streamlit dashboard
│   ├── constants.py         # Landmarks and references
│   ├── analyzer/
│   │   ├── landmarks.py     # MediaPipe detection
│   │   ├── validation.py    # Image validation
│   │   ├── symmetry.py      # Symmetry analysis
│   │   ├── proportions.py   # Facial proportions
│   │   ├── eyes.py          # Ocular analysis
│   │   ├── nose.py          # Nasal analysis
│   │   ├── lips.py          # Lip analysis
│   │   ├── jaw.py           # Jaw analysis
│   │   ├── chin.py          # Chin analysis
│   │   ├── profile.py       # Profile analysis
│   │   ├── harmony.py       # Harmony index
│   │   ├── engine.py        # Main orchestrator
│   │   ├── charts.py        # Chart generation
│   │   └── report.py        # PDF report
│   ├── utils/
│   │   ├── geometry.py      # Geometry functions
│   │   ├── drawing.py       # Visual annotations
│   │   ├── image.py         # Image processing
│   │   └── export.py        # Data export
│   ├── assets/
│   │   └── face_landmarker.task   # MediaPipe model (via download_model.py)
│   ├── scripts/
│   │   └── download_model.py      # Model download
├── reports/
└── requirements.txt
```

## Usage

1. Upload a **frontal photo** (minimum 1000px on the longest side)
2. Optionally upload a **profile photo**
3. Configure **visualization layers** in the sidebar
4. Click **Analyze**
5. Browse the tabs for results, charts, and reports
6. Export the PDF report or data as JSON/CSV/Excel

## Image Validation

The tool automatically checks:

| Criterion | Requirement |
|-----------|-------------|
| Resolution | Minimum 1000px on the longest side |
| Lighting | Brightness between 40–230 |
| Head tilt | Maximum ±15° |
| Visibility | Full face, no cropping |
| Obstructions | Eyes, nose, and mouth visible |

## Disclaimer

The **Harmony Index** represents a geometric assessment based on calculated metrics only. **It is not an absolute measure of attractiveness.**

## Extensibility

The project is structured for future features:

- Comparison between two photos
- Longitudinal tracking over time
- Video facial analysis
- Body analysis
- Multi-language support
- AI model integration

## License

Developed for anthropometric analysis and research purposes.
### CC0-1.0 license
