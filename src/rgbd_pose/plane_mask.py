"""Depth-based foreground masks for a tabletop scene.

This module assumes that the supplied reference pixels belong to one physical
table plane.  It is intentionally a foreground-mask helper, not a general
object detector or a pose correction method.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .features import rectangular_roi_mask
from .geometry import backproject_pixels


@dataclass(frozen=True)
class PlaneModel:
    """Plane in camera coordinates: ``normal @ point + offset = 0``."""

    normal: np.ndarray
    offset: float
    sample_count: int
    rms_residual_m: float


def fit_plane_from_pixels(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> PlaneModel:
    """Fit a plane to valid RGB-D samples at image pixels.

    The caller must provide at least three non-collinear table samples.  The
    fit is least-squares; callers should choose reference pixels away from the
    object and inspect the reported residual.
    """

    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 3:
        raise ValueError("pixels_uv must have shape (N, 2) with N >= 3")
    points, valid = backproject_pixels(pixels, depth_m, intrinsics)
    points = points[valid]
    if len(points) < 3:
        raise ValueError("fewer than three plane reference pixels have valid depth")
    centered = points - points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]), 1.0)
    if len(singular_values) < 2 or singular_values[-2] <= scale * 1e-5:
        raise ValueError("plane reference pixels must not be collinear")
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ points.mean(axis=0))
    residuals = points @ normal + offset
    return PlaneModel(
        normal=normal.astype(np.float64),
        offset=offset,
        sample_count=int(len(points)),
        rms_residual_m=float(np.sqrt(np.mean(residuals**2))),
    )


def tabletop_foreground_mask(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    plane: PlaneModel,
    *,
    min_height_m: float = 0.015,
    roi_xyxy: Sequence[int] | None = None,
    min_component_area: int = 0,
    close_kernel: int = 0,
) -> np.ndarray:
    """Return valid pixels at least ``min_height_m`` from the table plane.

    Distance is absolute because the fitted normal has no required
    orientation.  An optional ROI and connected-component area filter make
    the result suitable for a tabletop containing one central object.
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("depth_m must be HxW and intrinsics must be 3x3")
    normal = np.asarray(plane.normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("plane.normal must be a finite length-3 vector")
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 0:
        raise ValueError("plane.normal must have non-zero length")
    if not np.isfinite(plane.offset):
        raise ValueError("plane.offset must be finite")
    normalized_normal = normal / normal_norm
    normalized_offset = float(plane.offset) / normal_norm
    if not np.isfinite(min_height_m) or min_height_m <= 0:
        raise ValueError("min_height_m must be finite and positive")
    if min_component_area < 0:
        raise ValueError("min_component_area must be non-negative")
    if close_kernel < 0:
        raise ValueError("close_kernel must be non-negative")

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics must contain finite positive focal lengths")
    points = np.stack(
        [
            (xx - cx) * depth / fx,
            (yy - cy) * depth / fy,
            depth,
        ],
        axis=-1,
    )
    signed_distance = points @ normalized_normal + normalized_offset
    mask = valid & np.isfinite(signed_distance) & (np.abs(signed_distance) >= min_height_m)
    if roi_xyxy is not None:
        roi_mask = rectangular_roi_mask(depth.shape, roi_xyxy)
        assert roi_mask is not None
        mask &= roi_mask

    if close_kernel:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python for mask morphology") from exc
        kernel_size = int(close_kernel)
        if kernel_size % 2 == 0:
            raise ValueError("close_kernel must be odd when non-zero")
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

    if min_component_area:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python for connected components") from exc
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_component_area)
        mask = keep[labels]
    return mask.astype(bool)
