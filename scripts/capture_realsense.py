#!/usr/bin/env python3
"""Capture one color-aligned RGB-D frame from an Intel RealSense camera."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.io import save_rgbd_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("Install pyrealsense2 on the D435i acquisition machine") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        frames = None
        for _ in range(args.warmup):
            frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        rgb = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()
        save_rgbd_frame(args.output, rgb, depth_m, k)
        print(f"Saved RGB-D frame to {args.output}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
