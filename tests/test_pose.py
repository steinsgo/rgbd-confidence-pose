import numpy as np

from rgbd_pose.geometry import backproject_pixels, rotation_error_deg
from rgbd_pose.pose import (
    estimate_pose_comparison,
    estimate_pose_ransac,
    summarize_pose_estimate,
    weighted_rigid_transform,
)


def test_backprojection_center_pixel():
    depth = np.ones((3, 3), dtype=float) * 2
    k = np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]], dtype=float)
    points, valid = backproject_pixels(np.array([[1, 1], [2, 1]]), depth, k)
    assert valid.all()
    np.testing.assert_allclose(points, [[0, 0, 2], [0.02, 0, 2]])


def test_weighted_rigid_transform():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    angle = np.deg2rad(25)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translation = np.array([0.2, -0.1, 0.3])
    target = source @ rotation.T + translation
    transform = weighted_rigid_transform(source, target)
    assert rotation_error_deg(transform[:3, :3], rotation) < 1e-5
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-8)


def test_confidence_ransac_rejects_outliers():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(80, 3))
    target = source + np.array([0.1, 0.0, -0.05])
    confidence = np.ones(80)
    target[:25] = rng.normal(size=(25, 3))
    confidence[:25] = 0.03
    result = estimate_pose_ransac(source, target, confidence, threshold_m=0.01, iterations=400)
    assert result.inliers.sum() == 55
    np.testing.assert_allclose(result.transform[:3, 3], [0.1, 0, -0.05], atol=1e-8)


def test_pose_comparison_uses_same_correspondences_and_uniform_baseline():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(30, 3))
    target = source + np.array([0.04, -0.02, 0.08])
    confidence = np.linspace(0.2, 1.0, len(source))

    weighted, unweighted = estimate_pose_comparison(
        source,
        target,
        confidence,
        threshold_m=0.001,
        iterations=100,
        seed=11,
    )
    direct_uniform = estimate_pose_ransac(
        source,
        target,
        None,
        threshold_m=0.001,
        iterations=100,
        seed=11,
    )

    np.testing.assert_allclose(unweighted.transform, direct_uniform.transform)
    assert weighted.inliers.all()
    assert unweighted.inliers.all()
    weighted_metrics = summarize_pose_estimate(weighted, confidence)
    unweighted_metrics = summarize_pose_estimate(unweighted)
    assert weighted_metrics["inliers"] == 30
    assert unweighted_metrics["inliers"] == 30
    np.testing.assert_allclose(
        weighted_metrics["rmse_m"], unweighted_metrics["rmse_m"], atol=1e-12
    )
