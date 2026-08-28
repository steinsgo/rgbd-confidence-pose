from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import transform_points


@dataclass(frozen=True)
class PoseEstimate:
    transform: np.ndarray
    inliers: np.ndarray
    residuals_m: np.ndarray
    weighted_rmse_m: float


def summarize_pose_estimate(
    estimate: PoseEstimate,
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    """Return comparable geometric metrics for one pose estimate.

    ``weights=None`` gives ordinary uniform RMSE.  Supplying confidence values
    additionally reports confidence-weighted RMSE over that estimate's own
    inliers.  The estimate itself is never modified.
    """
    residuals = np.asarray(estimate.residuals_m, dtype=np.float64)
    inliers = np.asarray(estimate.inliers, dtype=bool)
    if residuals.ndim != 1 or inliers.shape != residuals.shape:
        raise ValueError("estimate residuals and inliers must have matching 1-D shapes")
    if not np.any(inliers):
        raise ValueError("pose estimate must contain at least one inlier")

    if weights is None:
        effective_weights = np.ones(residuals.shape, dtype=np.float64)
    else:
        effective_weights = np.asarray(weights, dtype=np.float64)
        if effective_weights.shape != residuals.shape:
            raise ValueError("weights must match the number of residuals")
        effective_weights = np.maximum(effective_weights, 0.0)
    inlier_weights = effective_weights[inliers]
    if inlier_weights.sum() <= 0:
        raise ValueError("inlier weights must have positive sum")

    inlier_residuals = residuals[inliers]
    return {
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "rmse_m": float(np.sqrt(np.mean(inlier_residuals**2))),
        "weighted_rmse_m": float(
            np.sqrt(np.average(inlier_residuals**2, weights=inlier_weights))
        ),
        "transform_reference_to_query": estimate.transform.tolist(),
    }


def weighted_rigid_transform(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Kabsch estimate for target ~= T * source."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")
    w = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0)
    if w.shape != (len(source),) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    w /= w.sum()
    source_center = np.sum(source * w[:, None], axis=0)
    target_center = np.sum(target * w[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * w[:, None]).T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def estimate_pose_ransac(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray | None = None,
    threshold_m: float = 0.025,
    iterations: int = 1500,
    seed: int = 0,
) -> PoseEstimate:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("At least three paired 3D points are required")
    conf = np.ones(len(source)) if confidence is None else np.asarray(confidence, float)
    conf = np.clip(conf, 1e-6, None)
    sampling_probability = conf / conf.sum()
    rng = np.random.default_rng(seed)

    best_score = -np.inf
    best_inliers = None
    for _ in range(iterations):
        chosen = rng.choice(len(source), 3, replace=False, p=sampling_probability)
        try:
            transform = weighted_rigid_transform(source[chosen], target[chosen], conf[chosen])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        inliers = residuals < threshold_m
        if inliers.sum() < 3:
            continue
        score = float(conf[inliers].sum()) - 0.05 * float(residuals[inliers].mean() / threshold_m)
        if score > best_score:
            best_score, best_inliers = score, inliers

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a valid pose")
    transform = weighted_rigid_transform(source[best_inliers], target[best_inliers], conf[best_inliers])
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    inliers = residuals < threshold_m
    rmse = float(np.sqrt(np.average(residuals[inliers] ** 2, weights=conf[inliers])))
    return PoseEstimate(transform, inliers, residuals, rmse)


def estimate_pose_comparison(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    threshold_m: float = 0.025,
    iterations: int = 1500,
    seed: int = 0,
) -> tuple[PoseEstimate, PoseEstimate]:
    """Estimate weighted and uniform poses from identical correspondences.

    Both estimates use the same threshold, iteration count, and random seed.
    The weighted estimate uses ``confidence`` for sampling, scoring, and its
    final rigid fit.  The baseline uses uniform sampling, scoring, and fit by
    passing ``confidence=None``.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim != 1 or confidence.shape[0] != len(source):
        raise ValueError("confidence must contain one value per correspondence")
    weighted = estimate_pose_ransac(
        source,
        target,
        confidence,
        threshold_m=threshold_m,
        iterations=iterations,
        seed=seed,
    )
    unweighted = estimate_pose_ransac(
        source,
        target,
        None,
        threshold_m=threshold_m,
        iterations=iterations,
        seed=seed,
    )
    return weighted, unweighted
