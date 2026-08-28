"""Descriptive quality metadata for DINOv2 RGB-D pair estimates.

The report deliberately does not declare pose accuracy.  It records mask and
patch coverage, matching consensus, and warnings about likely background
leakage so that a human can compare runs without changing the estimator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .features import ImageFeatures


def feature_mask_kind(
    roi_xyxy: Sequence[int] | None,
    polygon_xy: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
    mask_path: str | None = None,
) -> str:
    """Return the user-specified mask kind without interpreting its geometry."""

    parts: list[str] = []
    if roi_xyxy is not None:
        parts.append("roi")
    if polygon_xy is not None:
        parts.append("polygon")
    if mask_path is not None:
        parts.append("mask_file")
    return "_and_".join(parts) if parts else "none"


def _feature_mask_summary(
    features: ImageFeatures,
    mask: np.ndarray | None,
    roi_xyxy: Sequence[int] | None,
    polygon_xy: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
    mask_path: str | None,
) -> dict[str, object]:
    kind = feature_mask_kind(roi_xyxy, polygon_xy, mask_path)
    candidate_count = (
        int(features.candidate_count)
        if features.candidate_count is not None
        else int(len(features.uv))
    )
    selected_count = int(len(features.uv))
    summary: dict[str, object] = {
        "mask_type": kind,
        "candidate_patch_count": candidate_count,
        "selected_patch_count": selected_count,
        "selected_patch_ratio": (
            float(selected_count / candidate_count) if candidate_count else 0.0
        ),
        "patch_grid": (
            list(features.grid_shape) if features.grid_shape is not None else None
        ),
    }
    if mask_path is not None:
        summary["mask_path"] = mask_path
    if mask is None:
        summary["mask_pixel_area_ratio"] = 1.0
    else:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise ValueError("DINO quality mask must be a two-dimensional array")
        summary["mask_pixel_area_ratio"] = float(mask_array.mean())
    return summary


def _mask_warning(mask_kind: str) -> tuple[str, str]:
    if "mask_file" in mask_kind:
        return (
            "reduced_but_unverified",
            "mask file supplied; inspect its boundary and source procedure for leakage",
        )
    if mask_kind == "none":
        return (
            "high",
            "no feature mask supplied; fixed background may dominate matches",
        )
    if mask_kind == "roi":
        return (
            "elevated",
            "rectangle ROI only; background patch leakage remains possible",
        )
    if mask_kind == "polygon":
        return (
            "reduced_but_unverified",
            "manual polygon supplied; automatic object segmentation is not verified",
        )
    return (
        "reduced_but_unverified",
        "manual polygon and ROI supplied; automatic object segmentation is not verified",
    )


def summarize_dino_quality(
    *,
    reference_features: ImageFeatures,
    query_features: ImageFeatures,
    reference_mask: np.ndarray | None,
    query_mask: np.ndarray | None,
    reference_roi: Sequence[int] | None,
    query_roi: Sequence[int] | None,
    reference_polygon: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
    query_polygon: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
    reference_mask_path: str | None = None,
    query_mask_path: str | None = None,
    depth_valid_matches: int,
    valid_3d_matches: int,
    not_used_3d_matches: int,
    inliers: int | None = None,
    inlier_ratio: float | None = None,
    weighted_rmse_m: float | None = None,
    ransac_threshold_m: float | None = None,
    rotation_deviation_from_identity_deg: float | None = None,
    translation_deviation_from_identity_m: float | None = None,
) -> dict[str, object]:
    """Build a descriptive, JSON-serializable DINOv2 pair-quality report."""

    reference_kind = feature_mask_kind(
        reference_roi, reference_polygon, reference_mask_path
    )
    query_kind = feature_mask_kind(query_roi, query_polygon, query_mask_path)
    reference_risk, reference_warning = _mask_warning(reference_kind)
    query_risk, query_warning = _mask_warning(query_kind)

    geometry: dict[str, object] = {
        "depth_valid_matches": int(depth_valid_matches),
        "valid_3d_matches": int(valid_3d_matches),
        "not_used_3d_matches": int(not_used_3d_matches),
    }
    if inliers is not None:
        geometry["inliers"] = int(inliers)
    if inlier_ratio is not None:
        geometry["inlier_ratio"] = float(inlier_ratio)
    if weighted_rmse_m is not None:
        geometry["weighted_rmse_m"] = float(weighted_rmse_m)
    if ransac_threshold_m is not None:
        geometry["ransac_threshold_m"] = float(ransac_threshold_m)
        if weighted_rmse_m is not None and ransac_threshold_m > 0:
            geometry["rmse_to_ransac_threshold_ratio"] = float(
                weighted_rmse_m / ransac_threshold_m
            )
    if rotation_deviation_from_identity_deg is not None:
        geometry["rotation_deviation_from_identity_deg"] = float(
            rotation_deviation_from_identity_deg
        )
    if translation_deviation_from_identity_m is not None:
        geometry["translation_deviation_from_identity_m"] = float(
            translation_deviation_from_identity_m
        )

    return {
        "schema_version": 1,
        "interpretation": (
            "descriptive_only; no automatic segmentation or ground-truth pose "
            "accuracy is inferred"
        ),
        "reference": {
            **_feature_mask_summary(
                reference_features,
                reference_mask,
                reference_roi,
                reference_polygon,
                reference_mask_path,
            ),
            "background_leakage_risk": reference_risk,
        },
        "query": {
            **_feature_mask_summary(
                query_features,
                query_mask,
                query_roi,
                query_polygon,
                query_mask_path,
            ),
            "background_leakage_risk": query_risk,
        },
        "geometry": geometry,
        "warnings": [reference_warning, query_warning],
    }
