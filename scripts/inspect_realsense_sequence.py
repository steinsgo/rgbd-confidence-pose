#!/usr/bin/env python3
"""Inspect one frame from the recorded D435i sequence format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.realsense_sequence import load_realsense_sequence_frame


def _save_depth_preview(depth_m: np.ndarray, valid: np.ndarray, path: Path) -> None:
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    values = depth_m[valid]
    if values.size:
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.round(scaled[valid] * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_realsense_sequence_frame(args.session, args.frame_index)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_path = args.output / "rgb.png"
    depth_preview_path = args.output / "depth_preview.png"
    summary_path = args.output / "summary.json"
    Image.fromarray(frame.rgb, mode="RGB").save(rgb_path)
    _save_depth_preview(frame.depth_m, frame.valid_depth_mask, depth_preview_path)

    valid_depth = frame.depth_m[frame.valid_depth_mask]
    summary = {
        "session": str(Path(args.session).resolve()),
        "frame_index": frame.frame_index,
        "rgb_shape": list(frame.rgb.shape),
        "rgb_dtype": str(frame.rgb.dtype),
        "depth_m_shape": list(frame.depth_m.shape),
        "depth_m_dtype": str(frame.depth_m.dtype),
        "valid_depth_count": int(valid_depth.size),
        "invalid_depth_count": frame.invalid_depth_count,
        "invalid_depth_ratio": frame.invalid_depth_ratio,
        "valid_depth_min_m": float(valid_depth.min()) if valid_depth.size else None,
        "valid_depth_median_m": float(np.median(valid_depth)) if valid_depth.size else None,
        "valid_depth_max_m": float(valid_depth.max()) if valid_depth.size else None,
        "intrinsics": frame.intrinsics.tolist(),
        "rgb_timestamp_ms": frame.rgb_timestamp_ms,
        "depth_timestamp_ms": frame.depth_timestamp_ms,
        "timestamp_delta_ms": frame.timestamp_delta_ms,
        "timestamp_difference_ms": frame.timestamp_difference_ms,
        "depth_scale_meters_per_unit": frame.depth_scale_meters_per_unit,
        "source_paths": {key: str(value) for key, value in frame.source_paths.items()},
        "rgb_preview": str(rgb_path),
        "depth_preview": str(depth_preview_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
