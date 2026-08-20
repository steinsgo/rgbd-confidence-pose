#!/usr/bin/env python3
"""Estimate relative pose between two saved RGB-D frames or sequence frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.features import (
    build_feature_mask,
    extract_dinov2_patch_features,
    extract_opencv_features,
)
from rgbd_pose.geometry import backproject_pixels, rotation_error_deg
from rgbd_pose.io import load_rgbd_frame
from rgbd_pose.matching import combine_confidence, mutual_nearest_matches
from rgbd_pose.pose import estimate_pose_ransac
from rgbd_pose.quality_review import validate_approved_pair
from rgbd_pose.realsense_sequence import RealSenseFrame, load_realsense_sequence_frame


def _save_depth_preview(depth_m: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    values = depth_m[valid]
    if values.size:
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.round(scaled[valid] * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def _draw_matches(
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_uv: np.ndarray,
    query_uv: np.ndarray,
    highlight: np.ndarray,
    label: str,
    path: Path,
) -> None:
    import cv2

    reference_bgr = cv2.cvtColor(np.asarray(reference_rgb), cv2.COLOR_RGB2BGR)
    query_bgr = cv2.cvtColor(np.asarray(query_rgb), cv2.COLOR_RGB2BGR)
    canvas = np.hstack((reference_bgr, query_bgr))
    width = reference_bgr.shape[1]
    for index, (reference_point, query_point) in enumerate(
        zip(reference_uv, query_uv)
    ):
        color = (0, 220, 0) if bool(highlight[index]) else (0, 0, 230)
        reference_xy = tuple(np.rint(reference_point).astype(int))
        query_xy = tuple(np.rint(query_point).astype(int) + np.array([width, 0]))
        cv2.line(canvas, reference_xy, query_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, reference_xy, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, query_xy, 3, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write match visualization: {path}")


def _frame_payload(frame: RealSenseFrame) -> dict[str, object]:
    return {
        "frame_index": frame.frame_index,
        "rgb_timestamp_ms": frame.rgb_timestamp_ms,
        "depth_timestamp_ms": frame.depth_timestamp_ms,
        "timestamp_delta_ms": frame.timestamp_delta_ms,
        "timestamp_difference_ms": frame.timestamp_difference_ms,
        "invalid_depth_count": frame.invalid_depth_count,
        "invalid_depth_ratio": frame.invalid_depth_ratio,
        "source_paths": {key: str(value) for key, value in frame.source_paths.items()},
    }


def _add_sequence_metadata(
    payload: dict[str, object],
    session: Path | None,
    reference_frame: RealSenseFrame | None,
    query_frame: RealSenseFrame | None,
) -> None:
    if session is not None and reference_frame is not None and query_frame is not None:
        payload["session"] = str(session.resolve())
        payload["reference"] = _frame_payload(reference_frame)
        payload["query"] = _frame_payload(query_frame)


def _add_review_metadata(
    payload: dict[str, object], review_info: dict[str, object] | None
) -> None:
    if review_info is not None:
        payload["review_manifest"] = review_info


def _add_roi_metadata(
    payload: dict[str, object],
    reference_roi: list[int] | None,
    query_roi: list[int] | None,
    reference_polygon: list[float] | None = None,
    query_polygon: list[float] | None = None,
) -> None:
    if reference_roi is not None:
        payload["reference_roi_xyxy"] = reference_roi
    if query_roi is not None:
        payload["query_roi_xyxy"] = query_roi
    if reference_polygon is not None:
        payload["reference_polygon_xy"] = reference_polygon
    if query_polygon is not None:
        payload["query_polygon_xy"] = query_polygon


def _save_evidence(
    evidence_dir: Path,
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_depth: np.ndarray,
    query_depth: np.ndarray,
    matches_reference_uv: np.ndarray,
    matches_query_uv: np.ndarray,
    valid_3d: np.ndarray,
    inliers: np.ndarray,
    payload: dict[str, object],
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reference_rgb, mode="RGB").save(evidence_dir / "rgb_reference.png")
    Image.fromarray(query_rgb, mode="RGB").save(evidence_dir / "rgb_query.png")
    _save_depth_preview(reference_depth, evidence_dir / "depth_reference_preview.png")
    _save_depth_preview(query_depth, evidence_dir / "depth_query_preview.png")
    _draw_matches(
        reference_rgb,
        query_rgb,
        matches_reference_uv,
        matches_query_uv,
        valid_3d,
        "green=used 3D match, red=not used in 3D",
        evidence_dir / "matches_all.png",
    )
    _draw_matches(
        reference_rgb,
        query_rgb,
        matches_reference_uv[valid_3d],
        matches_query_uv[valid_3d],
        inliers,
        "green=RANSAC inlier, red=3D-valid geometric outlier",
        evidence_dir / "matches_inliers.png",
    )
    (evidence_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", nargs="?", type=Path)
    parser.add_argument("query", nargs="?", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--reference-index", type=int)
    parser.add_argument("--query-index", type=int)
    parser.add_argument("--backend", choices=["sift", "orb", "dino"], default="sift")
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--top-fraction", type=float, default=0.5)
    parser.add_argument("--ransac-threshold", type=float, default=0.03)
    parser.add_argument(
        "--reference-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="restrict OpenCV features to this reference image rectangle",
    )
    parser.add_argument(
        "--query-roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="restrict OpenCV features to this query image rectangle",
    )
    parser.add_argument(
        "--reference-polygon",
        nargs="+",
        type=float,
        metavar="COORD",
        help="restrict OpenCV features to reference polygon x0 y0 x1 y1 ...",
    )
    parser.add_argument(
        "--query-polygon",
        nargs="+",
        type=float,
        metavar="COORD",
        help="restrict OpenCV features to query polygon x0 y0 x1 y1 ...",
    )
    parser.add_argument("--output", type=Path, default=Path("results/pair.json"))
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--review-manifest",
        type=Path,
        help="require this manifest to explicitly approve the session pair",
    )
    args = parser.parse_args()

    if args.session is not None:
        if args.reference is not None or args.query is not None:
            parser.error("--session cannot be combined with positional frame directories")
        if args.reference_index is None or args.query_index is None:
            parser.error("--session requires --reference-index and --query-index")
        try:
            review_info = (
                validate_approved_pair(
                    args.review_manifest,
                    args.reference_index,
                    args.query_index,
                )
                if args.review_manifest is not None
                else None
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        reference_frame = load_realsense_sequence_frame(args.session, args.reference_index)
        query_frame = load_realsense_sequence_frame(args.session, args.query_index)
        ref_rgb, ref_depth, ref_k = (
            reference_frame.rgb,
            reference_frame.depth_m,
            reference_frame.intrinsics,
        )
        query_rgb, query_depth, query_k = (
            query_frame.rgb,
            query_frame.depth_m,
            query_frame.intrinsics,
        )
    else:
        if args.reference is None or args.query is None:
            parser.error("provide two frame directories or --session with two frame indices")
        if args.reference_index is not None or args.query_index is not None:
            parser.error("frame indices require --session")
        if args.review_manifest is not None:
            parser.error("--review-manifest requires --session")
        review_info = None
        reference_frame = None
        query_frame = None
        ref_rgb, ref_depth, ref_k = load_rgbd_frame(args.reference)
        query_rgb, query_depth, query_k = load_rgbd_frame(args.query)

    if args.backend == "dino" and (
        args.reference_roi is not None
        or args.query_roi is not None
        or args.reference_polygon is not None
        or args.query_polygon is not None
    ):
        parser.error(
            "ROI and polygon masks are supported only for SIFT or ORB"
        )
    try:
        reference_roi_mask = build_feature_mask(
            ref_rgb.shape,
            roi_xyxy=args.reference_roi,
            polygon_xy=args.reference_polygon,
        )
        query_roi_mask = build_feature_mask(
            query_rgb.shape,
            roi_xyxy=args.query_roi,
            polygon_xy=args.query_polygon,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    if args.backend == "dino":
        ref_features = extract_dinov2_patch_features(ref_rgb)
        query_features = extract_dinov2_patch_features(query_rgb)
    else:
        ref_features = extract_opencv_features(
            ref_rgb, args.backend, mask=reference_roi_mask
        )
        query_features = extract_opencv_features(
            query_rgb, args.backend, mask=query_roi_mask
        )
    if len(ref_features.uv) < 3 or len(query_features.uv) < 3:
        raise SystemExit("Not enough visual features; use a textured object or a wider crop")

    matches = mutual_nearest_matches(
        ref_features.descriptors,
        query_features.descriptors,
        ref_features.uv,
        query_features.uv,
        min_similarity=args.min_similarity,
        top_fraction=args.top_fraction,
    )
    ref_xyz, ref_valid = backproject_pixels(matches.reference_uv, ref_depth, ref_k)
    query_xyz, query_valid = backproject_pixels(matches.query_uv, query_depth, query_k)
    depth_valid = ref_valid & query_valid
    confidence = combine_confidence(matches.confidence, ref_valid, query_valid)
    valid = confidence > 0
    if valid.sum() < 3:
        failure_reason = "Fewer than three matches have valid depth"
        failure_payload: dict[str, object] = {
            "status": "failed",
            "failure_reason": failure_reason,
            "backend": args.backend,
            "reference_feature_count": int(len(ref_features.uv)),
            "query_feature_count": int(len(query_features.uv)),
            "visual_matches": int(len(matches.confidence)),
            "depth_valid_matches": int(depth_valid.sum()),
            "valid_3d_matches": int(valid.sum()),
            "not_used_3d_matches": int((~valid).sum()),
            "ransac_threshold_m": args.ransac_threshold,
            "runtime_s": time.perf_counter() - started,
        }
        _add_sequence_metadata(failure_payload, args.session, reference_frame, query_frame)
        _add_review_metadata(failure_payload, review_info)
        _add_roi_metadata(
            failure_payload,
            args.reference_roi,
            args.query_roi,
            args.reference_polygon,
            args.query_polygon,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(failure_payload, indent=2) + "\n", encoding="utf-8"
        )
        if args.evidence_dir is not None:
            _save_evidence(
                args.evidence_dir,
                ref_rgb,
                query_rgb,
                ref_depth,
                query_depth,
                matches.reference_uv,
                matches.query_uv,
                valid,
                np.zeros(int(valid.sum()), dtype=bool),
                failure_payload,
            )
        print(json.dumps(failure_payload, indent=2))
        raise SystemExit(failure_reason)
    estimate = estimate_pose_ransac(
        ref_xyz[valid],
        query_xyz[valid],
        confidence[valid],
        threshold_m=args.ransac_threshold,
    )
    runtime_s = time.perf_counter() - started
    payload: dict[str, object] = {
        "backend": args.backend,
        "reference_feature_count": int(len(ref_features.uv)),
        "query_feature_count": int(len(query_features.uv)),
        "visual_matches": int(len(matches.confidence)),
        "depth_valid_matches": int(depth_valid.sum()),
        "valid_3d_matches": int(valid.sum()),
        "not_used_3d_matches": int((~valid).sum()),
        "inliers": int(estimate.inliers.sum()),
        "inlier_ratio": float(estimate.inliers.mean()),
        "weighted_rmse_m": estimate.weighted_rmse_m,
        "ransac_threshold_m": args.ransac_threshold,
        "rotation_deviation_from_identity_deg": rotation_error_deg(
            estimate.transform[:3, :3], np.eye(3)
        ),
        "translation_deviation_from_identity_m": float(
            np.linalg.norm(estimate.transform[:3, 3])
        ),
        "runtime_s": runtime_s,
        "transform_reference_to_query": estimate.transform.tolist(),
    }
    _add_sequence_metadata(payload, args.session, reference_frame, query_frame)
    _add_review_metadata(payload, review_info)
    _add_roi_metadata(
        payload,
        args.reference_roi,
        args.query_roi,
        args.reference_polygon,
        args.query_polygon,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.evidence_dir is not None:
        _save_evidence(
            args.evidence_dir,
            ref_rgb,
            query_rgb,
            ref_depth,
            query_depth,
            matches.reference_uv,
            matches.query_uv,
            valid,
            estimate.inliers,
            payload,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
