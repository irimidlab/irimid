# IrimidLab

<p align="center">
  <b>Advanced Facial Geometry & Anthropometric Analysis</b><br>
  A modern computer vision toolkit for objective facial measurement using Python, MediaPipe, and OpenCV.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-CC0--1.0-green)

</p>

---

## Overview

IrimidLab is an open-source facial anthropometry platform designed to extract **objective geometric measurements** from facial images.

Using **MediaPipe Face Landmarker (478 landmarks)** and classical computational geometry, the software analyzes measurable facial properties such as symmetry, proportions, distances, anatomical relationships, and facial harmony.

Unlike beauty-rating applications, IrimidLab focuses on **quantifiable facial geometry**, making it useful for research, education, experimentation, and computer vision projects.

---

# Features

### Facial Landmark Detection

- 478 facial landmarks
- Automatic face detection
- Landmark visualization
- High precision tracking

### Image Quality Validation

- Resolution validation
- Lighting validation
- Head pose estimation
- Face visibility
- Obstruction detection

### Symmetry Analysis

- Global symmetry
- Eye symmetry
- Nose symmetry
- Mouth symmetry
- Jaw symmetry
- Regional asymmetry indices

### Facial Geometry

- Facial thirds
- Width-height ratios
- Facial centroid
- Facial axis
- Geometric center
- Ovality index
- Facial area
- Regional facial areas

### Eyes

- Eye Aspect Ratio (EAR)
- Eye area
- Width/height ratio
- Canthal tilt
- Brow tilt
- Eye symmetry
- Interpupillary distance

### Nose

- Nasal index
- Nostril symmetry
- Nose width
- Nose height
- Nasal curvature

### Lips

- Lip area
- Cupid's bow symmetry
- Mouth width
- Lip proportions

### Jaw & Chin

- Jaw width
- Mandibular ratios
- Chin proportions
- Chin symmetry
- Jaw definition index

### Profile Analysis (Optional)

- Nasolabial angle
- Facial convexity
- Chin projection
- Lip projection
- Ricketts E-line
- Steiner S-line
- Holdaway H-line

### Visualization

- Landmark overlay
- Facial axis
- Symmetry lines
- Measurement labels
- Regional visualization
- Optional layers

### Reports

- Interactive dashboard
- PDF report
- JSON export
- CSV export
- Excel export
- Analysis history

---

# Installation

```bash
git clone https://github.com/<your-user>/IrimidLab.git

cd IrimidLab

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

python face_analyzer/scripts/download_model.py
```

---

# Running

```bash
streamlit run face_analyzer/main.py
```

The dashboard will be available at:

```
http://localhost:8501
```

---

# Workflow

```
Image Upload
      │
      ▼
Image Validation
      │
      ▼
Face Detection
      │
      ▼
478 Landmark Detection
      │
      ▼
Geometry Analysis
      │
      ▼
Metric Calculation
      │
      ▼
Visualization
      │
      ▼
Report Generation
```

---

# Project Structure

```
IrimidLab/
│
├── face_analyzer/
│
├── analyzer/
│
├── utils/
│
├── assets/
│
├── reports/
│
└── requirements.txt
```

---

# Mathematical Metrics

IrimidLab computes dozens of geometric metrics, including:

- Euclidean distances
- Facial ratios
- Anatomical angles
- Facial thirds
- Width-height proportions
- Centroid calculations
- Convex hull area
- Polygonal regions
- Symmetry scores
- Geometric harmony index

All measurements are derived from landmark geometry.

---

# Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Pandas
- Streamlit
- Matplotlib

---

# Disclaimer

IrimidLab does **not** measure beauty objectively.

The Harmony Index is a composite geometric indicator derived from measurable facial proportions and symmetry. It should not be interpreted as an absolute attractiveness score.

---

# Roadmap

Planned features:

- Facial comparison
- Timeline analysis
- Video support
- 3D face reconstruction
- Skin analysis
- Machine Learning models
- Facial aging simulation
- REST API
- Desktop version
- Cloud deployment

---

# License

Released under the **CC0-1.0** License.
