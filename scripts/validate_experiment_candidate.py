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
        "--reference-polygon",
        nargs="+",
        type=float,
        metavar="COORD",
        help="restrict pair-check SIFT features to reference polygon x0 y0 x1 y1 ...",
    )
    parser.add_argument(
        "--query-polygon",
        nargs="+",
        type=float,
        metavar="COORD",
        help="restrict pair-check SIFT features to query polygon x0 y0 x1 y1 ...",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="decode every RGB/depth pair; default checks all references and decodes samples",
    )
    parser.add_argument(
        "--motion-scan",
        action="store_true",
        help="scan all frames for candidate motion and stable frame segments",
    )
    parser.add_argument(
        "--motion-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="optional rectangle used for the motion scan",
    )
    parser.add_argument(
        "--motion-window",
        type=int,
        default=5,
        help="frame separation for motion differences (default: 5)",
    )
    parser.add_argument(
        "--motion-quantile",
        type=float,
        default=0.95,
        help="relative motion-score quantile used as a trigger (default: 0.95)",
    )
    parser.add_argument(
        "--motion-min-score",
        type=float,
        default=0.01,
        help="minimum normalized grayscale change for a motion trigger",
    )
    parser.add_argument(
        "--min-stable-frames",
        type=int,
        default=15,
        help="minimum length of a reported stable segment (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment_candidate_quality.json"),
        help="JSON report path; the input session is never written",
    )
    parser.add_argument(
        "--review-manifest",
        type=Path,
        help="require the selected pair to be approved in this review manifest",
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
        full_scan=args.full_scan or args.motion_scan,
        motion_scan=args.motion_scan,
        motion_roi=args.motion_roi,
        motion_window=args.motion_window,
        motion_quantile=args.motion_quantile,
        motion_min_score=args.motion_min_score,
        min_stable_frames=args.min_stable_frames,
        reference_roi=args.reference_roi,
        query_roi=args.query_roi,
        reference_polygon=args.reference_polygon,
        query_polygon=args.query_polygon,
        review_manifest=args.review_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return {"PASS": 0, "WARN": 1, "REJECT": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
