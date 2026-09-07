# Rebar Geometry Analysis — MoMS for Monocular Rebar Spacing Inspection

Official implementation of:

**A Mixture of Measurement Strategies Framework for Monocular Mobile Rebar Spacing Inspection**

Accepted at the **IEEE International Conference on Image Processing (ICIP), 2026**.

This repository provides an end-to-end framework for **rebar spacing inspection from a single monocular mobile image**.

The proposed **Mixture of Measurement Strategies (MoMS)** framework reconstructs rebar topology, estimates metric scale, incorporates monocular depth cues, and adaptively selects between **planar** and **depth-guided** measurement strategies according to scene reliability.

---

## Overview

Reliable rebar spacing inspection is challenging under real-world construction conditions, including perspective distortion, depth ambiguity, occlusion, multi-layer rebar structures, and viewpoint variations.

Instead of relying on a single fixed measurement strategy, **MoMS** formulates rebar spacing inspection as a reliability-aware measurement problem.

The overall framework consists of:

- Rebar intersection detection
- Rebar topology reconstruction
- Checkerboard-based metric scale estimation
- Monocular depth estimation
- Reliability-Aware Measurement Strategy Selection (RAMS)
- Planar or depth-guided spacing measurement

The framework operates from a single RGB image and does not require a dedicated depth sensor or dense 3D point-cloud reconstruction.

---

## Method

### Rebar Structure Extraction

Rebar intersections are detected using a construction-specific **YOLO12** detector.

The detected intersection points are then organized into grid-consistent line structures using:

- PCA-based dominant orientation estimation
- Topology-constrained line construction
- Circular-statistics filtering
- Hough Transform refinement

The resulting rebar line segments provide the geometric basis for spacing measurement.

### Metric Scale Estimation

A checkerboard with known physical dimensions is used as a metric reference.

The framework adopts a **Pyramid-ROI coarse-to-fine checkerboard detection strategy** to improve both robustness and computational efficiency.

### Monocular Depth Estimation

The framework uses **Depth Pro** to estimate monocular depth from a single RGB image.

Depth is treated as an auxiliary geometric cue rather than being used unconditionally for direct measurement, reducing the impact of unstable depth predictions.

### Reliability-Aware Measurement Strategy Selection

The proposed **RAMS** module determines whether the current scene should use:

- **Planar Measurement**
- **Depth-Guided Measurement**

RAMS jointly considers RGB information, monocular depth, and geometric features extracted from the rebar structure.

A **ResNet-18** based classifier is used to select the measurement strategy expected to provide more reliable spacing estimates.

---

## Project Structure

```text
Rebar-Geometry-Analysis/
├── weights/
│   ├── best.pt
│   └── 2d3d_with_depth.pt
├── 2d3d_with_depth_predict.py
├── batch_predict.py
├── line_extractor.py
├── gen_line.py
├── compute_features.py
├── advanced_lbr.py
└── README.md
```

| File | Description |
| --- | --- |
| `weights/best.pt` | Rebar intersection detector |
| `weights/2d3d_with_depth.pt` | RAMS classifier weights |
| `line_extractor.py` | Rebar topology reconstruction and line extraction |
| `gen_line.py` | Generates structural rebar line information |
| `compute_features.py` | Extracts geometric and depth-related features |
| `2d3d_with_depth_predict.py` | Reliability-aware strategy prediction |
| `batch_predict.py` | Batch inference utility |
| `advanced_lbr.py` | Utility functions for rebar geometry analysis |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Chengen-Li/Rebar-Geometry-Analysis.git
cd Rebar-Geometry-Analysis
```

### 2. Install Dependencies

```bash
pip install torch torchvision ultralytics opencv-python scikit-image numpy pyyaml
```

Additional dependencies may be required for Depth Pro.

---

## Usage

### 1. Generate Rebar Structural Lines

```bash
python gen_line.py
```

### 2. Extract Geometric Features

```bash
python compute_features.py
```

### 3. Run Reliability-Aware Strategy Prediction

```bash
python 2d3d_with_depth_predict.py \
    --folder ./your_images \
    --depth_dir ./your_depth_maps
```

The prediction determines which measurement strategy should be used:

- `0` — Planar Measurement
- `1` — Depth-Guided Measurement

---

## Dataset

The framework was evaluated using a custom dataset captured with **monocular mobile RGB cameras**, without dedicated depth sensors.

The dataset covers four representative construction scenarios:

- Indoor vertical
- Indoor slab
- Outdoor vertical
- Outdoor slab

The dataset contains:

| Annotation Type | Number |
| --- | ---: |
| Rebar Intersections | 6,142 |
| Rebar Line Segments | 2,073 |

Ground-truth spacing was obtained from manual measurements performed by civil engineering experts.

---

## Experimental Results

### Rebar Spacing Measurement

Performance is evaluated using **Mean Relative Error (MRE)**.

| Scenario | Planar | Depth-Guided | RAMS |
| --- | ---: | ---: | ---: |
| Indoor slab | 3.99% | 5.34% | **3.93%** |
| Outdoor slab | 17.24% | **7.28%** | 11.45% |
| Outdoor vertical | **10.36%** | 47.61% | 18.06% |

The results show that no single measurement strategy is consistently reliable across all scenarios.

Planar measurement performs well under stable geometric conditions, while depth-guided measurement can better handle severe perspective effects. RAMS is designed to adaptively select the more suitable strategy and avoid severe failure cases.

---

## Viewing Angle Analysis

The framework was further evaluated under frontal and oblique views in a dual-layer indoor vertical scenario.

| Axis | Front Planar | Front Depth-Guided | Oblique Planar | Oblique Depth-Guided |
| --- | ---: | ---: | ---: | ---: |
| X | 4.47% | 4.86% | 8.34% | **0.57%** |
| Y | 3.26% | **2.13%** | **0.74%** | 4.75% |
| Z | 35.10% | **1.64%** | 24.26% | **6.00%** |

These results further demonstrate that planar and depth-guided formulations have complementary strengths under different viewpoints and measurement directions.

---

## Intersection Detection Performance

The final rebar intersection detector was evaluated across three construction sites.

| Dataset | Precision | Recall | F1 Score | mAP |
| --- | ---: | ---: | ---: | ---: |
| Site A | 0.387 | 0.385 | 0.386 | 0.303 |
| Site B | 0.429 | 0.455 | 0.442 | 0.347 |
| Site C | 0.598 | 0.542 | 0.569 | 0.522 |

On the independent Site C test set, the final detector achieves:

- **F1 Score: 0.569**
- **mAP: 0.522**

---

## Runtime Optimization

The framework was optimized for practical cloud-based mobile deployment.

| Component | Baseline | Stage I | Stage II |
| --- | ---: | ---: | ---: |
| Load Model | 24.56 s | 11.69 s | 0.19 s |
| Load Image | 0.01 s | 0.03 s | 0.01 s |
| Inference | 518.00 s | 498.02 s | 6.45 s |
| Total Time | 542.57 s | 509.74 s | **6.65 s** |

The optimized pipeline reduces total execution time from **542.57 s to 6.65 s**.

Checkerboard detection performance also improves:

| Method | mAP |
| --- | ---: |
| Global Exhaustive | 0.413 |
| Pyramid-ROI | **0.535** |

---

## Main Contributions

- **Mixture of Measurement Strategies (MoMS):** An end-to-end monocular rebar spacing inspection framework using lightweight geometric cues.
- **Reliability-Aware Measurement Strategy Selection (RAMS):** An adaptive strategy selector between planar and depth-guided measurement.
- **Robust Real-World Evaluation:** Evaluated across multiple construction environments and viewing conditions.
- **Practical Deployment:** Uses sparse geometric processing and an optimized cloud-based pipeline for efficient mobile inspection.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026moms,
  title     = {A Mixture of Measurement Strategies Framework for Monocular Mobile Rebar Spacing Inspection},
  author    = {Li, Cheng-En and Lai, Jue-Yu and Hsiao, Hung-Kai and Hsiao, Chang-Yuan and Lu, Peggy Joy},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}
```
