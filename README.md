# Confidence-Aware RGB-D Correspondence for Unseen Object Pose Estimation

[![English](https://img.shields.io/badge/README-English-2ea44f?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/README-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-0969da?style=for-the-badge)](README.zh-CN.md)

Small research code for matching an unseen object between two RGB-D
observations, lifting 2-D matches into 3-D, and estimating a relative rigid
transform. The project is inspired by the confidence-aware correspondence idea
in COG, but it is **not an exact reproduction of COG**.

The current implementation is deliberately modest:

- OpenCV SIFT/ORB feature matching;
- optional DINOv2 feature matching;
- explicit RGB-D validity checks and 2-D-to-3-D back-projection;
- transparent heuristic correspondence confidence;
- confidence-aware Kabsch refinement and RANSAC;
- read-only tools for checking recorded D435i sessions.

For one pair, the main output is a relative `4 x 4` transform together with
match counts, RANSAC inliers, and residual diagnostics. A transform from a
static-camera recording is a correspondence sanity check, not automatically a
pose-accuracy measurement.

## Current status

| Area | Status | What has actually been checked |
|---|---|---|
| Custom frame-directory loader | Implemented | Unit tests |
| D435i sequence adapter | Implemented | Unit tests and read-only real-session decoding |
| Intrinsics, depth scaling, invalid-depth handling, back-projection | Implemented | Unit tests and real-frame diagnostics |
| SIFT baseline | Implemented | CPU runs on real D435i frames |
| Confidence-aware Kabsch/RANSAC | Implemented | Deterministic synthetic validation |
| Recording, motion, and tabletop-depth checks | Implemented | Unit tests and read-only scans |
| Polygon and tabletop foreground masks | Implemented (heuristic) | Unit tests and real-frame inspection; no segmentation ground truth |
| DINOv2 backend | Experimental | Real-pair smoke runs; no ground-truth validation |
| Weighted-vs-unweighted comparison | Diagnostic code only | No scientifically meaningful conclusion yet |
| Ground-truth pose benchmark | Not available | Manual rotation was not independently measured |

### Verified synthetic result

```text
inliers=105/160
weighted_rmse_m=0.003205
rotation_error_deg=0.0839
translation_error_m=0.001614
```

These numbers come from a generated scene with known motion and injected
outliers. They are not real-world measurements.

### Verified D435i smoke checks

The latest complete recording, `20260820_142523_animebox`, passed the recorder's
native integrity check:

- 600 color frames and 600 depth frames;
- 2,007 accelerometer samples and 3,997 gyroscope samples;
- RGB/depth rate about 30 FPS;
- maximum RGB-depth timestamp difference: 10.08 ms;
- zero video queue drops and no writer errors.

The read-only adapter, tabletop mask, SIFT, and DINOv2 paths were also run on
selected frames. The recording used a static camera and manually handled object
motion, so its transforms are reported only as diagnostics. It is not a
viewpoint-change benchmark, ground-truth evaluation, or claim of robustness.

## Environment

The intended setup is:

- WSL2 Ubuntu 22.04 for development and inference;
- Python 3.10 in the project-local `.venv`;
- Windows-side D435i capture with aligned color/depth;
- completed sessions read from a mounted path such as `/mnt/e/...`.

The CPU adapter and SIFT baseline do not need direct D435i USB access from WSL.
PyTorch and DINOv2 are optional. Open3D, ROS, Gazebo, and Isaac Sim are not
required.

The Windows recorder and original RGB-D data are not part of the public
research workflow here. The adapter reads a completed session without writing
to it.

## Installation

Use a project-local environment; do not install into the system or Conda base
environment.

```bash
source .venv/bin/activate
pip install -e '.[vision,dev]'
```

Basic checks:

```bash
python -m pytest -q
python scripts/synthetic_demo.py
python -m pip check
```

## Input data

The original pair interface remains compatible with:

```text
frame_directory/
├── rgb.png
├── depth_m.npy
└── intrinsics.txt
```

A recorded D435i session has the following relevant files:

```text
session/
├── calibration.json
├── color/*.png
├── depth/*.png
├── frames.csv
├── imu.csv
├── metadata.json
└── validation_report.json
```

`frames.csv` is the authoritative RGB/depth pairing table. The adapter parses
the aligned color-camera intrinsics from `calibration.json`, decodes raw
`uint16` depth PNGs using the recorded depth scale, and maps raw values `0` and
`65535` to invalid metric depth `0.0`. It does not impose a hidden 3 m cutoff.

## Quick start

### Inspect one D435i frame

```bash
python scripts/inspect_realsense_sequence.py \
  /mnt/e/path/to/session \
  --frame-index 30 \
  --output results/frame_030
```

This writes an RGB preview, a normalized depth preview, and a JSON summary
outside the original session.

### Check a recording

The normal scan checks file counts, timestamps, calibration, image decoding,
depth validity, and sampled SIFT availability:

```bash
python scripts/validate_experiment_candidate.py \
  /mnt/e/path/to/session \
  --full-scan \
  --output results/session_quality.json
```

For a tabletop scene, the optional depth-plane check can screen a foreground
mask. The plane points are scene-specific and must be reviewed visually:

```bash
python scripts/validate_experiment_candidate.py \
  /mnt/e/path/to/session \
  --full-scan \
  --tabletop-mask \
  --tabletop-plane-points 80 380 550 380 80 450 550 450 \
  --tabletop-mask-roi 180 100 450 370 \
  --output results/session_quality.json
```

The scan is read-only with respect to the session. Its exit codes are
`0=PASS`, `1=WARN`, and `2=REJECT`. A recording integrity pass does not by
itself establish experimental pose eligibility.

### Estimate a pair with SIFT

The backward-compatible frame-directory form is:

```bash
python scripts/run_pair.py \
  data/object/view_000 data/object/view_001 \
  --backend sift \
  --output results/pair.json
```

For a D435i sequence, select two frame indices:

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend sift \
  --output results/pair.json \
  --evidence-dir results/pair_evidence
```

The evidence directory contains RGB/depth previews and match images. A
rectangular ROI, polygon, or precomputed mask can be passed to restrict
feature extraction. These inputs do not modify the RGB-D session.

For example, a polygon is supplied as a flat `x y` sequence:

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend sift \
  --reference-polygon 258 129 369 119 391 298 273 327 258 303 \
  --query-polygon 300 117 408 150 415 323 331 341 210 249 254 166 \
  --output results/pair_polygon.json
```

The tabletop mask helper fits a plane from manually selected table pixels and
writes derived masks outside the session:

```bash
python scripts/generate_tabletop_mask.py \
  /mnt/e/path/to/session \
  --frame-index 80 \
  --plane-points 80 380 550 380 80 450 550 450 \
  --object-roi 180 100 450 370 \
  --output results/tabletop_mask/frame_080
```

Pass its `mask.png` outputs with `--reference-mask` and `--query-mask`. This
is a tabletop foreground heuristic, not general object segmentation.

### Optional DINOv2 backend

The DINOv2 path uses the same image-space ROI, polygon, and mask inputs. It is
experimental and should be treated as a feature-matching diagnostic:

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend dino \
  --dino-model dinov2_vits14 \
  --dino-device cuda \
  --output results/dino_pair.json
```

CUDA is explicit; this command does not silently fall back to CPU. The first
run may download the official DINOv2 repository and weights through
`torch.hub`. Do not interpret a successful DINOv2 transform as pose accuracy
without independent reference motion.

## Useful tools

| Tool | Purpose |
|---|---|
| `inspect_realsense_sequence.py` | Decode one frame and save previews |
| `validate_experiment_candidate.py` | Check recording or pair quality |
| `export_motion_review.py` | Export sampled RGB/depth review sheets |
| `approve_motion_review.py` | Record a human-reviewed frame pair |
| `generate_tabletop_mask.py` | Create a scene-specific depth-plane mask |
| `run_pair.py` | Run SIFT or DINOv2 correspondence and pose estimation |
| `summarize_dino_quality.py` | Summarize an existing DINOv2 result |

Motion scores and mask reports are aids for selecting data. They cannot tell
whether a change came from a hand, object motion, lighting, or camera motion,
and they do not create ground-truth labels.

## Limitations

- The confidence value is a transparent heuristic, not a calibrated probability.
- Manual ROIs and polygons are currently supported; the tabletop mask is
  scene-specific and heuristic.
- Low texture, occlusion, object symmetry, and missing depth can create
  plausible but incorrect correspondences.
- Static-camera recordings are useful for parsing and smoke checks, not by
  themselves for a viewpoint-change benchmark.
- No independent object pose reference is currently available.
- The weighted-vs-unweighted path is diagnostic code; this repository does not
  claim a measured advantage for either estimator.
- DINOv2 is experimental. DINOv3, large-scale benchmarking, Open3D, ROS,
  Gazebo, and Isaac Sim are intentionally deferred.

## Future plan

The current manual-rotation collection is paused. The existing recordings are
useful for validating parsing, synchronization, depth, masking, and feature
matching, but they do not establish pose accuracy because the nominal rotation
was not independently measured.

When a better debugging setup is available, the next controlled experiment is:

1. use a rigid fixture or turntable so the object motion is repeatable;
2. record the intended angle and, if possible, an independent reference pose;
3. keep any ArUco/AprilTag or fixture reference in a separate evaluation path,
   rather than using it as a matching feature;
4. compare the estimated relative `SE(3)` against that independent reference;
5. only then draw conclusions about pose accuracy or weighted-vs-unweighted
   performance.

A single planar marker can disappear at large rotations, so the reference
should be attached to a fixture or arranged with enough visible geometry for
the required motion range.
