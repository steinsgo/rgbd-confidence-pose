#!/usr/bin/env python3
"""Export visual review artifacts from a sequence-quality motion report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.quality_review import export_motion_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export read-only RGB/depth candidate-frame review artifacts."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    manifest = export_motion_review(
        args.session,
        args.quality_report,
        args.output,
        max_frames=args.max_frames,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
