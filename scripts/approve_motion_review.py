#!/usr/bin/env python3
"""Explicitly approve one reviewed pair in a motion-review manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.quality_review import approve_review_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mark one manually reviewed reference/query pair as approved."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--reference-index", type=int, required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--note", required=True, help="short visual-review decision note")
    args = parser.parse_args()

    try:
        manifest = approve_review_manifest(
            args.manifest,
            args.reference_index,
            args.query_index,
            args.note,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
