from __future__ import annotations

import numpy as np


def backproject_pixels(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image pixels to 3D camera coordinates.

    Returns points and a validity mask. Pixel coordinates use (u, v) order.
    """
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv must have shape (N, 2)")
    if depth.ndim != 2 or k.shape != (3, 3):
        raise ValueError("depth_m must be HxW and intrinsics must be 3x3")

    uv = np.rint(pixels).astype(np.int64)
    h, w = depth.shape
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    z = np.full(len(uv), np.nan, dtype=np.float64)
    z[in_bounds] = depth[uv[in_bounds, 1], uv[in_bounds, 0]]
    valid = in_bounds & np.isfinite(z) & (z > 0)

    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    points = np.full((len(uv), 3), np.nan, dtype=np.float64)
    points[valid, 0] = (pixels[valid, 0] - cx) * z[valid] / fx
    points[valid, 1] = (pixels[valid, 1] - cy) * z[valid] / fy
    points[valid, 2] = z[valid]
    return points, valid


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("Expected points Nx3 and transform 4x4")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_error_deg(r_est: np.ndarray, r_gt: np.ndarray) -> float:
    relative = np.asarray(r_est) @ np.asarray(r_gt).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
