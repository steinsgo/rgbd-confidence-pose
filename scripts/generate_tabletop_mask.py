#!/usr/bin/env python3
"""Generate a read-only depth-based foreground mask for one RGB-D frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.plane_mask import fit_plane_from_pixels, tabletop_foreground_mask
from rgbd_pose.realsense_sequence import load_realsense_sequence_frame


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_m[valid], [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.rint(scaled[valid] * 255.0).astype(np.uint8)
    return preview


def _parse_points(values: list[float]) -> np.ndarray:
    if len(values) < 6 or len(values) % 2:
        raise ValueError("--plane-points requires at least three x,y pairs")
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("--plane-points must contain finite coordinates")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a tabletop plane and save a binary foreground mask"
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--plane-points", nargs="+", type=float, required=True)
    parser.add_argument(
        "--object-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="optional ROI applied after tabletop foreground selection",
    )
    parser.add_argument("--min-height-m", type=float, default=0.015)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        plane_points = _parse_points(args.plane_points)
        frame = load_realsense_sequence_frame(args.session, args.frame_index)
        plane = fit_plane_from_pixels(
            plane_points, frame.depth_m, frame.intrinsics
        )
        mask = tabletop_foreground_mask(
            frame.depth_m,
            frame.intrinsics,
            plane,
            min_height_m=args.min_height_m,
            roi_xyxy=args.object_roi,
            min_component_area=args.min_component_area,
            close_kernel=args.close_kernel,
        )
    except (FileNotFoundError, IndexError, NotADirectoryError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    mask_path = args.output / "mask.png"
    overlay_path = args.output / "overlay.png"
    depth_path = args.output / "depth_preview.png"
    summary_path = args.output / "summary.json"

    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    Image.fromarray(_depth_preview(frame.depth_m), mode="L").save(depth_path)

    overlay = frame.rgb.copy()
    if np.any(mask):
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        overlay[mask] = np.rint(
            0.45 * overlay[mask].astype(np.float32)
            + 0.55 * green[mask].astype(np.float32)
        ).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    valid_depth = frame.valid_depth_mask
    mask_count = int(mask.sum())
    valid_count = int(valid_depth.sum())
    summary = {
        "session": str(args.session.resolve()),
        "frame_index": frame.frame_index,
        "source_paths": {key: str(value) for key, value in frame.source_paths.items()},
        "rgb_shape": list(frame.rgb.shape),
        "rgb_dtype": str(frame.rgb.dtype),
        "depth_shape": list(frame.depth_m.shape),
        "depth_dtype": str(frame.depth_m.dtype),
        "depth_scale_meters_per_unit": frame.depth_scale_meters_per_unit,
        "invalid_depth_count": frame.invalid_depth_count,
        "invalid_depth_ratio": frame.invalid_depth_ratio,
        "valid_depth_count": valid_count,
        "plane_reference_pixels_uv": plane_points.tolist(),
        "plane_normal": plane.normal.tolist(),
        "plane_offset": plane.offset,
        "plane_sample_count": plane.sample_count,
        "plane_rms_residual_m": plane.rms_residual_m,
        "min_height_m": args.min_height_m,
        "object_roi_xyxy": args.object_roi,
        "min_component_area": args.min_component_area,
        "close_kernel": args.close_kernel,
        "mask_pixel_count": mask_count,
        "mask_pixel_ratio": float(mask_count / mask.size),
        "mask_ratio_of_valid_depth": (
            float(mask_count / valid_count) if valid_count else 0.0
        ),
        "rgb_timestamp_ms": frame.rgb_timestamp_ms,
        "depth_timestamp_ms": frame.depth_timestamp_ms,
        "timestamp_difference_ms": frame.timestamp_difference_ms,
        "outputs": {
            "mask": str(mask_path.resolve()),
            "overlay": str(overlay_path.resolve()),
            "depth_preview": str(depth_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
