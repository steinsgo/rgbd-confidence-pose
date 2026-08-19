# Confidence-Aware RGB-D Correspondence for Unseen Object Pose Estimation

A small-scale independent study inspired by the confidence-aware correspondence idea in **COG**. The project estimates the relative SE(3) pose of an unseen object from two RGB-D views. Appearance confidence, mutual consistency, valid depth, and geometric RANSAC are combined instead of treating every visual match equally.

## Research question

Can simple, interpretable confidence filtering make pretrained visual correspondences more reliable for RGB-D pose estimation under viewpoint change and partial overlap?

## Pipeline

1. Capture color-aligned RGB-D frames with an Intel RealSense D435i.
2. Crop or mask the target object.
3. Extract pretrained visual features (DINOv2 planned; SIFT/ORB baseline).
4. Retain mutual nearest-neighbor matches and rank them by similarity confidence.
5. Back-project matched pixels with depth and camera intrinsics.
6. Estimate target-to-query SE(3) with confidence-weighted RANSAC and Kabsch refinement.
7. Compare inlier ratio, registration RMSE, and pose error where ground truth is available.

This repository does **not** claim to reproduce COG's learned optimal-transport model.

## Current status

- [x] RGB-D frame format and D435i capture script
- [x] Pixel back-projection
- [x] Mutual-nearest confidence filtering
- [x] Confidence-weighted RANSAC and SE(3) refinement
- [x] Deterministic synthetic validation with outliers
- [x] OpenCV SIFT/ORB feature baseline
- [x] DINOv2 patch feature backend
- [ ] Self-collected D435i benchmark
- [ ] Viewpoint and occlusion experiments

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/synthetic_demo.py
```

Expected synthetic output is a sub-degree rotation error and sub-centimeter translation error despite low-confidence outlier correspondences.

### Capture a D435i frame

```bash
pip install -e '.[realsense]'
python scripts/capture_realsense.py data/cup/view_000
python scripts/capture_realsense.py data/cup/view_001
```

Each frame directory contains `rgb.png`, `depth_m.npy`, and `intrinsics.txt`.

### Estimate one RGB-D pair

Start with SIFT because it is quick to debug:

```bash
pip install -e '.[vision]'
python scripts/run_pair.py data/cup/view_000 data/cup/view_001 \
  --backend sift --output results/cup_000_001_sift.json
```

Then run pretrained DINOv2 features:

```bash
pip install -e '.[dino]'
python scripts/run_pair.py data/cup/view_000 data/cup/view_001 \
  --backend dino --min-similarity 0.6 \
  --output results/cup_000_001_dino.json
```

## Planned evaluation

Use 5 everyday objects with small, medium, and large viewpoint changes. Compare:

| Variant | Appearance | Confidence | Geometry |
|---|---|---|---|
| SIFT baseline | SIFT | ratio test | RANSAC |
| DINO baseline | DINOv2 | similarity threshold | RANSAC |
| Proposed | DINOv2 | mutual NN + similarity + valid depth | weighted RANSAC |

Primary metrics: correspondence inlier ratio, RANSAC inlier count, 3D registration RMSE, rotation error, and translation error. ArUco-based ground truth is optional; results without ground truth must be labeled as registration metrics rather than pose accuracy.

## Limitations

The current confidence score is heuristic and is not a calibrated probability. Object masks, symmetries, low-texture surfaces, missing depth, and large viewpoint changes can still cause failure. The study tests a lightweight interpretation of confidence-aware matching, not the learned marginals or optimal transport used by COG.
