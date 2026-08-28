"""Read-only quality screening for recorded RealSense RGB-D sessions.

The screener is deliberately a gate, not a benchmark.  It checks recording
integrity and, when a frame pair is supplied, runs the existing SIFT matching
and 3-D pose path as a small eligibility sanity check.  It never writes to a
session directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image

from .features import build_feature_mask, extract_opencv_features, rectangular_roi_mask
from .geometry import backproject_pixels
from .matching import combine_confidence, mutual_nearest_matches
from .plane_mask import fit_plane_from_pixels, tabletop_foreground_mask
from .pose import estimate_pose_ransac
from .quality_review import validate_approved_pair
from .realsense_sequence import RealSenseFrame, load_realsense_sequence_frame


Status = Literal["PASS", "WARN", "REJECT"]


@dataclass(frozen=True)
class QualityThresholds:
    """Explicit, conservative defaults for the screening gate."""

    timestamp_warn_ms: float = 20.0
    timestamp_reject_ms: float = 50.0
    invalid_depth_warn_ratio: float = 0.50
    invalid_depth_reject_ratio: float = 0.98
    far_depth_warn_m: float = 10.0
    far_depth_warn_ratio: float = 0.01
    min_sift_features_warn: int = 20
    min_sift_features_reject: int = 5
    min_visual_matches: int = 8
    min_valid_3d_matches: int = 6
    min_ransac_inliers: int = 6
    min_inlier_ratio: float = 0.50
    tabletop_plane_residual_warn_m: float = 0.010
    tabletop_plane_residual_reject_m: float = 0.030
    tabletop_mask_ratio_warn_max: float = 0.30


def _issue(
    issues: list[dict[str, Any]],
    severity: Literal["WARN", "REJECT"],
    code: str,
    message: str,
    scope: Literal["recording", "pair"],
    **details: Any,
) -> None:
    item: dict[str, Any] = {
        "severity": severity,
        "scope": scope,
        "code": code,
        "message": message,
    }
    if details:
        item["details"] = details
    issues.append(item)


def _status_for_issues(
    issues: list[dict[str, Any]], scope: Literal["recording", "pair"] | None = None
) -> Status:
    relevant = [item for item in issues if scope is None or item["scope"] == scope]
    if any(item["severity"] == "REJECT" for item in relevant):
        return "REJECT"
    if relevant:
        return "WARN"
    return "PASS"


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _read_frame_rows(session: Path) -> list[dict[str, str]]:
    path = session / "frames.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frames.csv: {path}")
    required = {
        "index",
        "color_file",
        "depth_file",
        "color_timestamp_ms",
        "depth_timestamp_ms",
    }
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("frames.csv has no header")
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"frames.csv is missing fields: {', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read frames.csv: {path}") from exc
    if not rows:
        raise ValueError("frames.csv contains no frame rows")
    return rows


def _parse_indexed_rows(rows: list[dict[str, str]]) -> tuple[list[int], dict[int, dict[str, str]]]:
    indexed: dict[int, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frame index at frames.csv line {line_number}") from exc
        if index in indexed:
            raise ValueError(f"Duplicate frame index {index} in frames.csv")
        indexed[index] = row
    return sorted(indexed), indexed


def _select_sample_indices(indices: list[int], sample_count: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not indices:
        return []
    positions = np.linspace(0, len(indices) - 1, min(sample_count, len(indices)), dtype=int)
    return sorted({indices[int(position)] for position in positions})


def _resolve_referenced_path(session: Path, value: str, field: str) -> Path:
    if not value:
        raise ValueError(f"frames.csv field {field!r} is empty")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"frames.csv field {field!r} must be relative")
    session_resolved = session.resolve()
    resolved = (session / relative).resolve()
    if not resolved.is_relative_to(session_resolved):
        raise ValueError(f"frames.csv field {field!r} escapes the session directory")
    return resolved


def _depth_summary(frame: RealSenseFrame, thresholds: QualityThresholds) -> dict[str, Any]:
    valid_depth = frame.depth_m[frame.valid_depth_mask]
    far = valid_depth > thresholds.far_depth_warn_m
    summary: dict[str, Any] = {
        "frame_index": frame.frame_index,
        "rgb_shape": list(frame.rgb.shape),
        "rgb_dtype": str(frame.rgb.dtype),
        "depth_shape": list(frame.depth_m.shape),
        "depth_dtype": str(frame.depth_m.dtype),
        "valid_depth_count": int(valid_depth.size),
        "invalid_depth_count": frame.invalid_depth_count,
        "invalid_depth_ratio": frame.invalid_depth_ratio,
        "valid_depth_min_m": float(valid_depth.min()) if valid_depth.size else None,
        "valid_depth_median_m": float(np.median(valid_depth)) if valid_depth.size else None,
        "valid_depth_max_m": float(valid_depth.max()) if valid_depth.size else None,
        "far_depth_threshold_m": thresholds.far_depth_warn_m,
        "far_depth_count": int(far.sum()),
        "far_depth_ratio": float(far.mean()) if valid_depth.size else 0.0,
        "rgb_timestamp_ms": frame.rgb_timestamp_ms,
        "depth_timestamp_ms": frame.depth_timestamp_ms,
        "timestamp_difference_ms": frame.timestamp_difference_ms,
    }
    return summary


def _motion_gray_signature(
    frame: RealSenseFrame,
    motion_roi: Sequence[int] | None,
    output_size: tuple[int, int] = (64, 48),
) -> np.ndarray:
    """Return a normalized low-resolution grayscale signature for one frame."""
    if motion_roi is None:
        region = frame.rgb
    else:
        rectangular_roi_mask(frame.rgb.shape, motion_roi)
        x0, y0, x1, y1 = (int(value) for value in motion_roi)
        region = frame.rgb[y0:y1, x0:x1]
    image = Image.fromarray(np.asarray(region, dtype=np.uint8), mode="RGB")
    resized = image.convert("L").resize(output_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _contiguous_ranges(indices: Sequence[int]) -> list[dict[str, int]]:
    """Summarize sorted frame indices as inclusive contiguous ranges."""
    if not indices:
        return []
    values = sorted(set(int(index) for index in indices))
    ranges: list[dict[str, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append(
                {
                    "start_index": start,
                    "end_index": previous,
                    "frame_count": previous - start + 1,
                }
            )
            start = value
        previous = value
    ranges.append(
        {
            "start_index": start,
            "end_index": previous,
            "frame_count": previous - start + 1,
        }
    )
    return ranges


def _motion_profile(
    frames: Sequence[RealSenseFrame],
    *,
    motion_roi: Sequence[int] | None = None,
    comparison_window: int = 5,
    motion_quantile: float = 0.95,
    minimum_score: float = 0.01,
    minimum_stable_frames: int = 15,
) -> dict[str, Any]:
    """Describe candidate motion and stable frame segments without labeling causes.

    Each score compares a frame with the frame ``comparison_window`` samples
    earlier.  The threshold is the larger of a recording-relative quantile and
    an explicit minimum score, so normal sensor/compression noise is not
    automatically called motion.  This is a screening heuristic, not optical
    flow or hand/object segmentation.
    """
    if comparison_window < 1:
        raise ValueError("comparison_window must be positive")
    if not 0.0 < motion_quantile < 1.0:
        raise ValueError("motion_quantile must be between 0 and 1")
    if not np.isfinite(minimum_score) or minimum_score < 0.0:
        raise ValueError("minimum_score must be finite and non-negative")
    if minimum_stable_frames < 1:
        raise ValueError("minimum_stable_frames must be positive")
    ordered = sorted(frames, key=lambda item: item.frame_index)
    if len(ordered) <= comparison_window:
        raise ValueError(
            "motion scan requires more frames than comparison_window"
        )
    signatures = [
        _motion_gray_signature(frame, motion_roi) for frame in ordered
    ]
    available_indices = {int(frame.frame_index) for frame in ordered}
    scores: list[dict[str, Any]] = []
    values: list[float] = []
    for position in range(comparison_window, len(ordered)):
        score = float(
            np.mean(
                np.abs(signatures[position] - signatures[position - comparison_window])
            )
        )
        values.append(score)
        scores.append(
            {
                "frame_index": int(ordered[position].frame_index),
                "reference_frame_index": int(
                    ordered[position - comparison_window].frame_index
                ),
                "mean_abs_gray_change": score,
            }
        )

    score_array = np.asarray(values, dtype=np.float64)
    quantile_score = float(np.quantile(score_array, motion_quantile))
    threshold = max(quantile_score, float(minimum_score))
    trigger_indices = [
        int(item["frame_index"])
        for item in scores
        if float(item["mean_abs_gray_change"]) >= threshold
    ]
    trigger_set = set(trigger_indices)

    # A windowed difference implicates the frames covered by that comparison,
    # not only its final frame. Expand each trigger backward conservatively.
    affected_indices: set[int] = set()
    for item in scores:
        if int(item["frame_index"]) not in trigger_set:
            continue
        current = int(item["frame_index"])
        reference = int(item["reference_frame_index"])
        affected_indices.update(
            index
            for index in range(reference + 1, current + 1)
            if index in available_indices
        )

    all_indices = [int(frame.frame_index) for frame in ordered]
    stable_indices = [index for index in all_indices if index not in affected_indices]
    stable_ranges = [
        item
        for item in _contiguous_ranges(stable_indices)
        if item["frame_count"] >= minimum_stable_frames
    ]
    candidate_ranges = _contiguous_ranges(sorted(affected_indices))
    for item in candidate_ranges:
        item["trigger_count"] = sum(
            index in trigger_set
            for index in range(item["start_index"], item["end_index"] + 1)
        )

    return {
        "comparison_window_frames": comparison_window,
        "motion_quantile": motion_quantile,
        "quantile_score": quantile_score,
        "minimum_score": float(minimum_score),
        "threshold_score": threshold,
        "signature_size": [64, 48],
        "motion_roi_xyxy": (
            [int(value) for value in motion_roi]
            if motion_roi is not None
            else None
        ),
        "score_min": float(score_array.min()),
        "score_median": float(np.median(score_array)),
        "score_max": float(score_array.max()),
        "frame_scores": scores,
        "motion_trigger_frame_indices": trigger_indices,
        "motion_candidate_intervals": candidate_ranges,
        "stable_segments": stable_ranges,
        "stable_segment_interpretation": (
            "low temporal RGB change only; not a hand-free or object-static guarantee"
        ),
        "manual_review_required": bool(trigger_indices),
        "stable_segment_count": len(stable_ranges),
        "longest_stable_segment_frames": max(
            (item["frame_count"] for item in stable_ranges), default=0
        ),
    }


def _frame_error_summary(index: int, exc: Exception) -> dict[str, Any]:
    return {
        "frame_index": index,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _normalize_tabletop_plane_points(
    points_uv: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    try:
        points = np.asarray(points_uv, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tabletop plane points must be numeric") from exc
    if points.ndim == 1:
        if points.size < 6 or points.size % 2:
            raise ValueError(
                "tabletop plane points require at least three x,y pairs"
            )
        points = points.reshape(-1, 2)
    elif points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "tabletop plane points must have shape (N, 2) or be a flat x,y sequence"
        )
    if len(points) < 3:
        raise ValueError("tabletop plane points require at least three samples")
    if not np.isfinite(points).all():
        raise ValueError("tabletop plane points must be finite")
    return points


def _mask_component_summary(mask: np.ndarray) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for tabletop mask connected-component screening"
        ) from exc
    mask_array = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_array.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "largest_component_bbox_xyxy": None,
            "component_areas_descending": [],
        }
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    order = np.argsort(areas)[::-1]
    largest_index = int(order[0]) + 1
    largest_area = int(areas[order[0]])
    total_area = int(areas.sum())
    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    width = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    return {
        "component_count": int(len(areas)),
        "largest_component_area": largest_area,
        "largest_component_ratio": (
            float(largest_area / total_area) if total_area else 0.0
        ),
        "largest_component_bbox_xyxy": [x, y, x + width, y + height],
        "component_areas_descending": [int(areas[index]) for index in order[:5]],
    }


def _tabletop_mask_scan(
    frame_cache: dict[int, RealSenseFrame],
    frame_indices: Sequence[int],
    plane_points_uv: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    mask_roi: Sequence[int] | None,
    min_height_m: float,
    min_component_area: int,
    close_kernel: int,
    thresholds: QualityThresholds,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Screen a small set of frames with a read-only tabletop foreground mask."""

    scan_issues: list[dict[str, Any]] = []

    def add_issue(
        severity: Literal["WARN", "REJECT"],
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        _issue(scan_issues, severity, code, message, "recording", **details)
        _issue(issues, severity, code, message, "recording", **details)

    try:
        plane_points = _normalize_tabletop_plane_points(plane_points_uv)
    except ValueError as exc:
        add_issue("REJECT", "tabletop_plane_points_invalid", str(exc))
        return {
            "enabled": True,
            "status": "REJECT",
            "frame_indices": [int(index) for index in frame_indices],
            "plane_reference_pixels_uv": None,
            "frames": [],
            "issues": scan_issues,
        }

    if not frame_indices:
        add_issue(
            "REJECT",
            "tabletop_mask_no_frames",
            "No frames were available for tabletop mask screening",
        )
        return {
            "enabled": True,
            "status": "REJECT",
            "frame_indices": [],
            "plane_reference_pixels_uv": plane_points.tolist(),
            "frames": [],
            "issues": scan_issues,
        }

    frame_reports: list[dict[str, Any]] = []
    for index in sorted(set(int(value) for value in frame_indices)):
        frame = frame_cache.get(index)
        if frame is None:
            add_issue(
                "REJECT",
                "tabletop_mask_frame_unavailable",
                "A requested tabletop mask frame was not decoded",
                frame_index=index,
            )
            frame_reports.append(
                {"frame_index": index, "status": "REJECT", "error": "frame unavailable"}
            )
            continue
        try:
            plane = fit_plane_from_pixels(
                plane_points, frame.depth_m, frame.intrinsics
            )
            mask = tabletop_foreground_mask(
                frame.depth_m,
                frame.intrinsics,
                plane,
                min_height_m=min_height_m,
                roi_xyxy=mask_roi,
                min_component_area=min_component_area,
                close_kernel=close_kernel,
            )
            components = _mask_component_summary(mask)
        except (
            FileNotFoundError,
            IndexError,
            NotADirectoryError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            add_issue(
                "REJECT",
                "tabletop_mask_frame_invalid",
                f"Tabletop mask screening failed for frame {index}: {exc}",
                frame_index=index,
            )
            frame_reports.append(
                {"frame_index": index, "status": "REJECT", "error": str(exc)}
            )
            continue

        valid_depth_count = int(frame.valid_depth_mask.sum())
        mask_count = int(mask.sum())
        mask_ratio = float(mask_count / mask.size)
        frame_report: dict[str, Any] = {
            "frame_index": index,
            "status": "PASS",
            "rgb_depth_timestamp_difference_ms": frame.timestamp_difference_ms,
            "invalid_depth_ratio": frame.invalid_depth_ratio,
            "valid_depth_count": valid_depth_count,
            "plane_normal": plane.normal.tolist(),
            "plane_offset": plane.offset,
            "plane_sample_count": plane.sample_count,
            "plane_rms_residual_m": plane.rms_residual_m,
            "mask_pixel_count": mask_count,
            "mask_pixel_ratio": mask_ratio,
            "mask_ratio_of_valid_depth": (
                float(mask_count / valid_depth_count) if valid_depth_count else 0.0
            ),
            **components,
        }
        if plane.rms_residual_m > thresholds.tabletop_plane_residual_reject_m:
            frame_report["status"] = "REJECT"
            add_issue(
                "REJECT",
                "tabletop_plane_residual_too_high",
                "The tabletop reference samples do not fit a sufficiently stable plane",
                frame_index=index,
                rms_residual_m=plane.rms_residual_m,
                threshold_m=thresholds.tabletop_plane_residual_reject_m,
            )
        elif plane.rms_residual_m > thresholds.tabletop_plane_residual_warn_m:
            frame_report["status"] = "WARN"
            add_issue(
                "WARN",
                "tabletop_plane_residual_high",
                "The tabletop plane fit residual is higher than the warning threshold",
                frame_index=index,
                rms_residual_m=plane.rms_residual_m,
                threshold_m=thresholds.tabletop_plane_residual_warn_m,
            )
        if mask_count == 0:
            frame_report["status"] = "REJECT"
            add_issue(
                "REJECT",
                "tabletop_mask_empty",
                "The tabletop foreground mask contains no pixels",
                frame_index=index,
            )
        elif mask_ratio > thresholds.tabletop_mask_ratio_warn_max:
            if frame_report["status"] == "PASS":
                frame_report["status"] = "WARN"
            add_issue(
                "WARN",
                "tabletop_mask_too_broad",
                "The tabletop foreground mask covers an unusually large image area",
                frame_index=index,
                mask_pixel_ratio=mask_ratio,
                threshold=thresholds.tabletop_mask_ratio_warn_max,
            )
        frame_reports.append(frame_report)

    return {
        "enabled": True,
        "status": _status_for_issues(scan_issues),
        "frame_indices": [int(index) for index in sorted(set(frame_indices))],
        "plane_reference_pixels_uv": plane_points.tolist(),
        "mask_roi_xyxy": list(mask_roi) if mask_roi is not None else None,
        "min_height_m": min_height_m,
        "min_component_area": min_component_area,
        "close_kernel": close_kernel,
        "thresholds": {
            "plane_residual_warn_m": thresholds.tabletop_plane_residual_warn_m,
            "plane_residual_reject_m": thresholds.tabletop_plane_residual_reject_m,
            "mask_ratio_warn_max": thresholds.tabletop_mask_ratio_warn_max,
        },
        "frames": frame_reports,
        "issues": scan_issues,
    }


def _pair_check(
    reference: RealSenseFrame,
    query: RealSenseFrame,
    backend: str,
    min_similarity: float,
    top_fraction: float,
    ransac_threshold_m: float,
    thresholds: QualityThresholds,
    issues: list[dict[str, Any]],
    reference_roi: Sequence[int] | None = None,
    query_roi: Sequence[int] | None = None,
    reference_polygon: Sequence[Sequence[float]] | Sequence[float] | None = None,
    query_polygon: Sequence[Sequence[float]] | Sequence[float] | None = None,
) -> dict[str, Any]:
    if backend != "sift":
        raise ValueError("The sequence quality gate currently supports only the SIFT backend")

    try:
        reference_mask = build_feature_mask(
            reference.rgb.shape,
            roi_xyxy=reference_roi,
            polygon_xy=reference_polygon,
        )
        query_mask = build_feature_mask(
            query.rgb.shape,
            roi_xyxy=query_roi,
            polygon_xy=query_polygon,
        )
    except ValueError as exc:
        mask_error_code = (
            "feature_mask_invalid"
            if reference_roi is not None
            and query_roi is not None
            and (reference_polygon is not None or query_polygon is not None)
            else "polygon_invalid"
            if reference_polygon is not None or query_polygon is not None
            else "roi_invalid"
        )
        _issue(
            issues,
            "REJECT",
            mask_error_code,
            f"The selected pair feature mask is invalid: {exc}",
            "pair",
        )
        result: dict[str, Any] = {
            "status": "REJECT",
            "backend": backend,
            "reference_index": reference.frame_index,
            "query_index": query.frame_index,
            "error": str(exc),
        }
        if reference_roi is not None:
            result["reference_roi_xyxy"] = list(reference_roi)
        if query_roi is not None:
            result["query_roi_xyxy"] = list(query_roi)
        if reference_polygon is not None:
            result["reference_polygon_xy"] = list(reference_polygon)
        if query_polygon is not None:
            result["query_polygon_xy"] = list(query_polygon)
        return result

    try:
        reference_features = extract_opencv_features(
            reference.rgb, backend, mask=reference_mask
        )
        query_features = extract_opencv_features(
            query.rgb, backend, mask=query_mask
        )
    except (ImportError, RuntimeError) as exc:
        _issue(
            issues,
            "REJECT",
            "feature_backend_unavailable",
            str(exc),
            "pair",
        )
        return {
            "status": "REJECT",
            "backend": backend,
            "error": str(exc),
        }

    result: dict[str, Any] = {
        "status": "PASS",
        "backend": backend,
        "reference_index": reference.frame_index,
        "query_index": query.frame_index,
        "reference_feature_count": int(len(reference_features.uv)),
        "query_feature_count": int(len(query_features.uv)),
        "min_similarity": min_similarity,
        "top_fraction": top_fraction,
        "ransac_threshold_m": ransac_threshold_m,
    }
    if reference_roi is not None:
        result["reference_roi_xyxy"] = list(reference_roi)
    if query_roi is not None:
        result["query_roi_xyxy"] = list(query_roi)
    if reference_polygon is not None:
        result["reference_polygon_xy"] = list(reference_polygon)
    if query_polygon is not None:
        result["query_polygon_xy"] = list(query_polygon)

    if min(len(reference_features.uv), len(query_features.uv)) < thresholds.min_sift_features_reject:
        _issue(
            issues,
            "REJECT",
            "too_few_sift_features",
            "The selected frame pair has too few SIFT features for a geometric check",
            "pair",
            reference_features=result["reference_feature_count"],
            query_features=result["query_feature_count"],
            minimum=thresholds.min_sift_features_reject,
        )
        result["status"] = "REJECT"
        result["visual_matches"] = 0
        result["valid_3d_matches"] = 0
        return result

    matches = mutual_nearest_matches(
        reference_features.descriptors,
        query_features.descriptors,
        reference_features.uv,
        query_features.uv,
        min_similarity=min_similarity,
        top_fraction=top_fraction,
    )
    reference_xyz, reference_valid = backproject_pixels(
        matches.reference_uv, reference.depth_m, reference.intrinsics
    )
    query_xyz, query_valid = backproject_pixels(
        matches.query_uv, query.depth_m, query.intrinsics
    )
    depth_valid = reference_valid & query_valid
    confidence = combine_confidence(matches.confidence, reference_valid, query_valid)
    valid = confidence > 0
    result.update(
        {
            "visual_matches": int(len(matches.confidence)),
            "depth_valid_matches": int(depth_valid.sum()),
            "valid_3d_matches": int(valid.sum()),
        }
    )

    if len(matches.confidence) < thresholds.min_visual_matches:
        _issue(
            issues,
            "REJECT",
            "too_few_visual_matches",
            "The selected frame pair has too few accepted visual matches",
            "pair",
            visual_matches=int(len(matches.confidence)),
            minimum=thresholds.min_visual_matches,
        )
        result["status"] = "REJECT"
        return result
    if valid.sum() < thresholds.min_valid_3d_matches:
        _issue(
            issues,
            "REJECT",
            "too_few_valid_3d_matches",
            "The selected frame pair does not provide enough valid 3-D correspondences",
            "pair",
            valid_3d_matches=int(valid.sum()),
            minimum=thresholds.min_valid_3d_matches,
        )
        result["status"] = "REJECT"
        return result

    try:
        estimate = estimate_pose_ransac(
            reference_xyz[valid],
            query_xyz[valid],
            confidence[valid],
            threshold_m=ransac_threshold_m,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        _issue(
            issues,
            "REJECT",
            "pose_estimation_failed",
            f"The selected frame pair failed geometric estimation: {exc}",
            "pair",
        )
        result["status"] = "REJECT"
        result["pose_error"] = str(exc)
        return result

    inlier_count = int(estimate.inliers.sum())
    inlier_ratio = float(estimate.inliers.mean())
    result.update(
        {
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "weighted_rmse_m": estimate.weighted_rmse_m,
            "transform_reference_to_query": estimate.transform.tolist(),
        }
    )
    if inlier_count < thresholds.min_ransac_inliers:
        _issue(
            issues,
            "REJECT",
            "too_few_ransac_inliers",
            "The selected frame pair has too few geometric inliers",
            "pair",
            inliers=inlier_count,
            minimum=thresholds.min_ransac_inliers,
        )
    if inlier_ratio < thresholds.min_inlier_ratio:
        _issue(
            issues,
            "REJECT",
            "low_ransac_inlier_ratio",
            "The selected frame pair has a low geometric inlier ratio",
            "pair",
            inlier_ratio=inlier_ratio,
            minimum=thresholds.min_inlier_ratio,
        )
    result["status"] = _status_for_issues(issues, "pair")
    return result


def _minimal_report(session: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    status = _status_for_issues(issues)
    return {
        "schema_version": 1,
        "session": str(session.resolve()),
        "status": status,
        "eligible_for_experiment": status == "PASS",
        "recording_status": _status_for_issues(issues, "recording"),
        "pair_status": "NOT_RUN",
        "issues": issues,
    }


def screen_realsense_session(
    session_dir: str | Path,
    *,
    reference_index: int | None = None,
    query_index: int | None = None,
    sample_count: int = 3,
    backend: str = "sift",
    min_similarity: float = 0.55,
    top_fraction: float = 0.50,
    ransac_threshold_m: float = 0.03,
    thresholds: QualityThresholds | None = None,
    check_features: bool = True,
    full_scan: bool = False,
    motion_scan: bool = False,
    motion_roi: Sequence[int] | None = None,
    motion_window: int = 5,
    motion_quantile: float = 0.95,
    motion_min_score: float = 0.01,
    min_stable_frames: int = 15,
    reference_roi: Sequence[int] | None = None,
    query_roi: Sequence[int] | None = None,
    reference_polygon: Sequence[Sequence[float]] | Sequence[float] | None = None,
    query_polygon: Sequence[Sequence[float]] | Sequence[float] | None = None,
    review_manifest: str | Path | None = None,
    tabletop_plane_points: Sequence[Sequence[float]] | Sequence[float] | None = None,
    tabletop_mask_frames: Sequence[int] | None = None,
    tabletop_mask_roi: Sequence[int] | None = None,
    tabletop_min_height_m: float = 0.015,
    tabletop_min_component_area: int = 0,
    tabletop_close_kernel: int = 0,
) -> dict[str, Any]:
    """Screen one recorded session without modifying it.

    ``reference_index`` and ``query_index`` are optional.  Without both, the
    report intentionally says that the pair check was not run; a recording
    integrity pass alone is not treated as experimental eligibility.
    """

    session = Path(session_dir)
    thresholds = thresholds or QualityThresholds()
    issues: list[dict[str, Any]] = []
    decode_all = full_scan or motion_scan
    tabletop_indices: list[int] = []
    if not session.is_dir():
        _issue(
            issues,
            "REJECT",
            "session_not_found",
            f"Session directory not found: {session}",
            "recording",
        )
        return _minimal_report(session, issues)

    try:
        metadata = _read_json_object(session / "metadata.json", "metadata.json")
    except (FileNotFoundError, ValueError) as exc:
        metadata = {}
        _issue(issues, "REJECT", "metadata_invalid", str(exc), "recording")

    try:
        rows = _read_frame_rows(session)
        frame_indices, indexed_rows = _parse_indexed_rows(rows)
    except (FileNotFoundError, ValueError) as exc:
        return _minimal_report(session, issues + [{
            "severity": "REJECT",
            "scope": "recording",
            "code": "frames_invalid",
            "message": str(exc),
        }])

    if tabletop_plane_points is not None:
        if tabletop_mask_frames is None:
            tabletop_indices = _select_sample_indices(frame_indices, sample_count)
        else:
            tabletop_indices = [
                int(index) for index in tabletop_mask_frames
            ]
        tabletop_indices.extend(
            int(index)
            for index in (reference_index, query_index)
            if index is not None
        )
        tabletop_indices = sorted(set(tabletop_indices))
    elif tabletop_mask_frames is not None:
        _issue(
            issues,
            "REJECT",
            "tabletop_mask_config_invalid",
            "tabletop_mask_frames requires tabletop_plane_points",
            "recording",
        )

    if metadata.get("status") != "complete":
        _issue(
            issues,
            "REJECT",
            "recording_incomplete",
            "metadata.status is not 'complete'",
            "recording",
            status=metadata.get("status"),
        )
    writer_errors = metadata.get("writer_errors", [])
    if writer_errors:
        _issue(
            issues,
            "REJECT",
            "writer_errors",
            "The recorder reported writer errors",
            "recording",
            count=len(writer_errors) if isinstance(writer_errors, list) else None,
        )

    counts = metadata.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        _issue(issues, "REJECT", "counts_invalid", "metadata.counts must be an object", "recording")
    video_received = counts.get("video_received")
    video_saved = counts.get("video_saved")
    if isinstance(video_received, (int, float)) and video_received != len(rows):
        _issue(
            issues,
            "REJECT",
            "video_received_count_mismatch",
            "metadata video_received does not match frames.csv",
            "recording",
            metadata_count=video_received,
            frame_rows=len(rows),
        )
    if isinstance(video_saved, (int, float)) and video_saved != len(rows):
        _issue(
            issues,
            "REJECT",
            "video_saved_count_mismatch",
            "metadata video_saved does not match frames.csv",
            "recording",
            metadata_count=video_saved,
            frame_rows=len(rows),
        )
    if not isinstance(video_received, (int, float)) or not isinstance(video_saved, (int, float)):
        _issue(
            issues,
            "REJECT",
            "video_counts_missing",
            "metadata.counts must include numeric video_received and video_saved",
            "recording",
        )
    queue_drops = counts.get("video_queue_drops", 0)
    if queue_drops != 0:
        _issue(
            issues,
            "REJECT",
            "video_queue_drops",
            "The recorder reported dropped video frames",
            "recording",
            video_queue_drops=queue_drops,
        )

    sorted_indices = frame_indices
    expected_indices = list(range(sorted_indices[0], sorted_indices[-1] + 1))
    missing_indices = sorted(set(expected_indices) - set(sorted_indices))
    if missing_indices:
        _issue(
            issues,
            "REJECT",
            "frame_index_gaps",
            "frames.csv has gaps in its frame indices",
            "recording",
            missing_count=len(missing_indices),
            first_missing=missing_indices[:5],
        )

    color_timestamps: list[float] = []
    depth_timestamps: list[float] = []
    timestamp_differences: list[float] = []
    color_domains: set[str] = set()
    depth_domains: set[str] = set()
    for row in rows:
        try:
            color_timestamp = float(row["color_timestamp_ms"])
            depth_timestamp = float(row["depth_timestamp_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            _issue(issues, "REJECT", "timestamp_invalid", str(exc), "recording")
            continue
        if not np.isfinite(color_timestamp) or not np.isfinite(depth_timestamp):
            _issue(
                issues,
                "REJECT",
                "timestamp_nonfinite",
                "frames.csv contains a non-finite RGB-D timestamp",
                "recording",
            )
            continue
        color_timestamps.append(color_timestamp)
        depth_timestamps.append(depth_timestamp)
        timestamp_differences.append(abs(color_timestamp - depth_timestamp))
        if row.get("color_timestamp_domain"):
            color_domains.add(row["color_timestamp_domain"])
        if row.get("depth_timestamp_domain"):
            depth_domains.add(row["depth_timestamp_domain"])

    if len(color_timestamps) == len(rows) and len(color_timestamps) > 1:
        if np.any(np.diff(color_timestamps) < 0):
            _issue(issues, "REJECT", "color_timestamps_not_monotonic", "RGB timestamps are not monotonic", "recording")
        if np.any(np.diff(depth_timestamps) < 0):
            _issue(issues, "REJECT", "depth_timestamps_not_monotonic", "Depth timestamps are not monotonic", "recording")
    max_timestamp_difference = max(timestamp_differences, default=None)
    if max_timestamp_difference is not None:
        if max_timestamp_difference > thresholds.timestamp_reject_ms:
            _issue(
                issues,
                "REJECT",
                "timestamp_delta_too_large",
                "The RGB-depth timestamp difference exceeds the rejection threshold",
                "recording",
                maximum_ms=max_timestamp_difference,
                threshold_ms=thresholds.timestamp_reject_ms,
            )
        elif max_timestamp_difference > thresholds.timestamp_warn_ms:
            _issue(
                issues,
                "WARN",
                "timestamp_delta_high",
                "The RGB-depth timestamp difference exceeds the warning threshold",
                "recording",
                maximum_ms=max_timestamp_difference,
                threshold_ms=thresholds.timestamp_warn_ms,
            )
    if len(color_domains) > 1 or len(depth_domains) > 1 or color_domains != depth_domains:
        _issue(
            issues,
            "WARN",
            "timestamp_domains_inconsistent",
            "RGB and depth timestamp domains are not uniform",
            "recording",
            color_domains=sorted(color_domains),
            depth_domains=sorted(depth_domains),
        )

    try:
        calibration = _read_json_object(session / "calibration.json", "calibration.json")
        scale = float(calibration["depth_scale_meters_per_unit"])
        color_intrinsics = calibration["color_intrinsics"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("depth_scale_meters_per_unit must be finite and positive")
        if not isinstance(color_intrinsics, dict):
            raise ValueError("color_intrinsics must be an object")
        calibration_summary = {
            "depth_scale_meters_per_unit": scale,
            "color_intrinsics": {
                key: color_intrinsics.get(key)
                for key in ("width", "height", "fx", "fy", "ppx", "ppy", "distortion_model")
            },
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        calibration_summary = None
        _issue(issues, "REJECT", "calibration_invalid", str(exc), "recording")

    color_files = sorted((session / "color").glob("*.png")) if (session / "color").is_dir() else []
    depth_files = sorted((session / "depth").glob("*.png")) if (session / "depth").is_dir() else []
    if len(color_files) != len(rows) or len(depth_files) != len(rows):
        _issue(
            issues,
            "REJECT",
            "stored_file_count_mismatch",
            "Stored RGB/depth PNG counts do not match frames.csv",
            "recording",
            color_files=len(color_files),
            depth_files=len(depth_files),
            frame_rows=len(rows),
        )

    frame_cache: dict[int, RealSenseFrame] = {}
    frame_errors: list[dict[str, Any]] = []
    missing_references: list[dict[str, Any]] = []
    for index in frame_indices:
        row = indexed_rows[index]
        try:
            rgb_path = _resolve_referenced_path(session, row["color_file"], "color_file")
            depth_path = _resolve_referenced_path(session, row["depth_file"], "depth_file")
            if not rgb_path.is_file() or not depth_path.is_file():
                missing_references.append(
                    {
                        "frame_index": index,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "rgb_exists": rgb_path.is_file(),
                        "depth_exists": depth_path.is_file(),
                    }
                )
        except (KeyError, TypeError, ValueError) as exc:
            missing_references.append(_frame_error_summary(index, exc))
    if missing_references:
        _issue(
            issues,
            "REJECT",
            "referenced_file_missing",
            "frames.csv references missing or invalid RGB/depth files",
            "recording",
            error_count=len(missing_references),
            examples=missing_references[:5],
        )

    invalid_ratios: list[float] = []
    far_ratios: list[float] = []
    decode_indices = frame_indices if decode_all else _select_sample_indices(frame_indices, sample_count)
    decode_indices = sorted(
        set(
            decode_indices
            + [index for index in (reference_index, query_index) if index is not None]
            + tabletop_indices
        )
    )
    for index in decode_indices:
        try:
            frame_cache[index] = load_realsense_sequence_frame(session, index)
            summary = _depth_summary(frame_cache[index], thresholds)
            invalid_ratios.append(float(summary["invalid_depth_ratio"]))
            far_ratios.append(float(summary["far_depth_ratio"]))
        except (FileNotFoundError, IndexError, NotADirectoryError, TypeError, ValueError) as exc:
            frame_errors.append(_frame_error_summary(index, exc))
    if frame_errors:
        _issue(
            issues,
            "REJECT",
            "frame_validation_failed",
            "One or more recorded frames failed RGB-D decoding or validation",
            "recording",
            error_count=len(frame_errors),
            examples=frame_errors[:5],
        )

    invalid_bad = [ratio for ratio in invalid_ratios if ratio > thresholds.invalid_depth_warn_ratio]
    invalid_reject = [ratio for ratio in invalid_ratios if ratio > thresholds.invalid_depth_reject_ratio]
    far_bad = [ratio for ratio in far_ratios if ratio > thresholds.far_depth_warn_ratio]
    depth_summary: dict[str, Any] = {
        "decode_mode": "full" if decode_all else "sampled",
        "frames_checked": len(invalid_ratios),
        "frames_available": len(frame_indices),
        "invalid_depth_ratio_min": min(invalid_ratios) if invalid_ratios else None,
        "invalid_depth_ratio_median": float(np.median(invalid_ratios)) if invalid_ratios else None,
        "invalid_depth_ratio_max": max(invalid_ratios) if invalid_ratios else None,
        "frames_above_invalid_warn_ratio": len(invalid_bad),
        "frames_above_invalid_reject_ratio": len(invalid_reject),
        "far_depth_warn_m": thresholds.far_depth_warn_m,
        "far_depth_ratio_max": max(far_ratios) if far_ratios else None,
        "frames_above_far_depth_warn_ratio": len(far_bad),
    }
    if invalid_reject:
        _issue(
            issues,
            "REJECT",
            "invalid_depth_ratio_too_high",
            "Some frames contain almost no valid depth",
            "recording",
            frames=len(invalid_reject),
            threshold=thresholds.invalid_depth_reject_ratio,
        )
    elif invalid_bad:
        _issue(
            issues,
            "WARN",
            "invalid_depth_ratio_high",
            "Some frames contain a high proportion of invalid depth",
            "recording",
            frames=len(invalid_bad),
            threshold=thresholds.invalid_depth_warn_ratio,
        )
    if far_bad:
        _issue(
            issues,
            "WARN",
            "far_depth_values_present",
            "Some valid depth values exceed the screening range; inspect depth noise or scene range",
            "recording",
            frames=len(far_bad),
            threshold_m=thresholds.far_depth_warn_m,
            ratio_threshold=thresholds.far_depth_warn_ratio,
        )

    motion_report: dict[str, Any] | None = None
    if motion_scan:
        if frame_errors or len(frame_cache) != len(frame_indices):
            _issue(
                issues,
                "REJECT",
                "motion_scan_unavailable",
                "Motion scan requires every recorded frame to decode successfully",
                "recording",
                decoded_frames=len(frame_cache),
                expected_frames=len(frame_indices),
            )
        else:
            try:
                motion_report = _motion_profile(
                    [frame_cache[index] for index in frame_indices],
                    motion_roi=motion_roi,
                    comparison_window=motion_window,
                    motion_quantile=motion_quantile,
                    minimum_score=motion_min_score,
                    minimum_stable_frames=min_stable_frames,
                )
            except (TypeError, ValueError) as exc:
                _issue(issues, "REJECT", "motion_scan_invalid", str(exc), "recording")
            else:
                if not motion_report["stable_segments"]:
                    _issue(
                        issues,
                        "WARN",
                        "stable_segment_not_found",
                        "Motion scan did not find a stable frame segment of the requested length",
                        "recording",
                        minimum_stable_frames=min_stable_frames,
                    )

    tabletop_mask_report: dict[str, Any] | None = None
    if tabletop_plane_points is not None:
        tabletop_mask_report = _tabletop_mask_scan(
            frame_cache,
            tabletop_indices,
            tabletop_plane_points,
            mask_roi=tabletop_mask_roi,
            min_height_m=tabletop_min_height_m,
            min_component_area=tabletop_min_component_area,
            close_kernel=tabletop_close_kernel,
            thresholds=thresholds,
            issues=issues,
        )

    sample_indices = _select_sample_indices(frame_indices, sample_count)
    sample_indices = sorted(set(sample_indices + [index for index in (reference_index, query_index) if index is not None]))
    samples: list[dict[str, Any]] = []
    feature_counts: dict[int, int] = {}
    if check_features:
        try:
            for index in sample_indices:
                frame = frame_cache.get(index)
                if frame is None:
                    continue
                feature_counts[index] = int(len(extract_opencv_features(frame.rgb, "sift").uv))
        except (ImportError, RuntimeError) as exc:
            _issue(issues, "REJECT", "sift_check_unavailable", str(exc), "recording")
    for index in sample_indices:
        frame = frame_cache.get(index)
        if frame is None:
            continue
        item = _depth_summary(frame, thresholds)
        if check_features:
            item["sift_feature_count"] = feature_counts.get(index)
            count = feature_counts.get(index, 0)
            if count < thresholds.min_sift_features_reject:
                _issue(
                    issues,
                    "REJECT",
                    "sample_too_few_sift_features",
                    "A sampled frame has too few SIFT features for matching",
                    "recording",
                    frame_index=index,
                    feature_count=count,
                    minimum=thresholds.min_sift_features_reject,
                )
            elif count < thresholds.min_sift_features_warn:
                _issue(
                    issues,
                    "WARN",
                    "sample_low_sift_features",
                    "A sampled frame has few SIFT features",
                    "recording",
                    frame_index=index,
                    feature_count=count,
                    warning_threshold=thresholds.min_sift_features_warn,
                )
        samples.append(item)

    pair_check: dict[str, Any] | None = None
    if (reference_index is None) != (query_index is None):
        _issue(
            issues,
            "REJECT",
            "pair_indices_incomplete",
            "Provide both reference_index and query_index for a pair check",
            "pair",
        )
        pair_status = "REJECT"
    elif reference_index is None:
        if review_manifest is not None:
            _issue(
                issues,
                "REJECT",
                "review_requires_pair",
                "A review manifest requires both reference_index and query_index",
                "pair",
            )
            pair_status = "REJECT"
            pair_check = {
                "status": "REJECT",
                "error": "review manifest requires an explicit frame pair",
                "review_manifest": str(Path(review_manifest).resolve()),
            }
            # Do not also report the generic pair-not-run warning.
            report_pair_not_run = False
        else:
            report_pair_not_run = True
        if report_pair_not_run:
            _issue(
                issues,
                "WARN",
                "pair_check_not_run",
                "No reference/query frame pair was supplied; recording integrity alone is not experimental eligibility",
                "pair",
            )
            pair_status = "NOT_RUN"
    elif reference_index == query_index:
        _issue(
            issues,
            "REJECT",
            "pair_indices_identical",
            "Reference and query frame indices must differ",
            "pair",
        )
        pair_status = "REJECT"
    elif reference_index not in frame_cache or query_index not in frame_cache:
        _issue(
            issues,
            "REJECT",
            "pair_frame_unavailable",
            "The requested reference/query frame could not be decoded",
            "pair",
            reference_index=reference_index,
            query_index=query_index,
        )
        pair_status = "REJECT"
    else:
        if review_manifest is not None:
            try:
                review_info = validate_approved_pair(
                    review_manifest, reference_index, query_index
                )
            except (OSError, ValueError) as exc:
                _issue(
                    issues,
                    "REJECT",
                    "review_not_approved",
                    str(exc),
                    "pair",
                    review_manifest=str(Path(review_manifest).resolve()),
                )
                pair_check = {
                    "status": "REJECT",
                    "backend": backend,
                    "reference_index": reference_index,
                    "query_index": query_index,
                    "review_manifest": str(Path(review_manifest).resolve()),
                    "error": str(exc),
                }
            else:
                pair_check = _pair_check(
                    frame_cache[reference_index],
                    frame_cache[query_index],
                    backend,
                    min_similarity,
                    top_fraction,
                    ransac_threshold_m,
                    thresholds,
                    issues,
                    reference_roi=reference_roi,
                    query_roi=query_roi,
                    reference_polygon=reference_polygon,
                    query_polygon=query_polygon,
                )
                pair_check["review_manifest"] = review_info
        else:
            pair_check = _pair_check(
                frame_cache[reference_index],
                frame_cache[query_index],
                backend,
                min_similarity,
                top_fraction,
                ransac_threshold_m,
                thresholds,
                issues,
                reference_roi=reference_roi,
                query_roi=query_roi,
                reference_polygon=reference_polygon,
                query_polygon=query_polygon,
            )
        pair_status = str(pair_check["status"])

    report: dict[str, Any] = {
        "schema_version": 1,
        "session": str(session.resolve()),
        "status": _status_for_issues(issues),
        "eligible_for_experiment": _status_for_issues(issues) == "PASS",
        "manual_review_required": bool(
            motion_report is not None and motion_report["manual_review_required"]
        ),
        "recording_status": _status_for_issues(issues, "recording"),
        "pair_status": pair_status,
        "issues": issues,
        "metadata": {
            "status": metadata.get("status"),
            "sequence_name": metadata.get("sequence_name"),
            "device": metadata.get("device"),
            "counts": counts,
            "writer_errors": writer_errors,
        },
        "recording": {
            "frame_count": len(rows),
            "first_frame_index": frame_indices[0],
            "last_frame_index": frame_indices[-1],
            "color_png_count": len(color_files),
            "depth_png_count": len(depth_files),
            "color_timestamp_domains": sorted(color_domains),
            "depth_timestamp_domains": sorted(depth_domains),
            "max_timestamp_difference_ms": max_timestamp_difference,
            "calibration": calibration_summary,
            "depth": depth_summary,
            "samples": samples,
        },
        "motion_scan": motion_report,
        "tabletop_mask_scan": tabletop_mask_report,
        "pair_check": pair_check,
        "thresholds": {
            key: value for key, value in vars(thresholds).items()
        },
    }
    return report
