#!/usr/bin/env python3
"""Screen a recorded RealSense session for experiment eligibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.sequence_quality import screen_realsense_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RGB-D recording quality gate. Exit codes: "
            "0=PASS, 1=WARN, 2=REJECT."
        )
    )
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--reference-index",
        type=int,
        help="reference frame for the optional SIFT/3-D pair check",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        help="query frame for the optional SIFT/3-D pair check",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--backend", choices=["sift"], default="sift")
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--top-fraction", type=float, default=0.50)
    parser.add_argument("--ransac-threshold", type=float, default=0.03)
    parser.add_argument(
        "--reference-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="restrict pair-check SIFT features to this reference rectangle",
    )
    parser.add_argument(
        "--query-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="restrict pair-check SIFT features to this query rectangle",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="decode every RGB/depth pair; default checks all references and decodes samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment_candidate_quality.json"),
        help="JSON report path; the input session is never written",
    )
    args = parser.parse_args()

    report = screen_realsense_session(
        args.session,
        reference_index=args.reference_index,
        query_index=args.query_index,
        sample_count=args.sample_count,
        backend=args.backend,
        min_similarity=args.min_similarity,
        top_fraction=args.top_fraction,
        ransac_threshold_m=args.ransac_threshold,
        full_scan=args.full_scan,
        reference_roi=args.reference_roi,
        query_roi=args.query_roi,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return {"PASS": 0, "WARN": 1, "REJECT": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
