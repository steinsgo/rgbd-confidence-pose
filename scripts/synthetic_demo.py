#!/usr/bin/env python3
"""Deterministic end-to-end test with outlier correspondences."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd_pose.geometry import rotation_error_deg
from rgbd_pose.pose import estimate_pose_comparison, estimate_pose_ransac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-unweighted",
        action="store_true",
        help="also report the uniform baseline on the same synthetic matches",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    source = rng.uniform([-0.3, -0.2, 0.7], [0.3, 0.2, 1.3], size=(160, 3))
    angle = np.deg2rad(18)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([0.08, -0.03, 0.05])
    target = source @ rotation.T + translation + rng.normal(0, 0.002, source.shape)
    confidence = rng.uniform(0.75, 1.0, len(source))
    outliers = rng.choice(len(source), 55, replace=False)
    target[outliers] = rng.uniform([-0.5, -0.5, 0.5], [0.5, 0.5, 1.6], size=(len(outliers), 3))
    confidence[outliers] = rng.uniform(0.01, 0.15, len(outliers))

    if args.compare_unweighted:
        estimate, unweighted = estimate_pose_comparison(
            source,
            target,
            confidence,
            threshold_m=0.015,
        )
    else:
        estimate = estimate_pose_ransac(source, target, confidence, threshold_m=0.015)
        unweighted = None
    rot_error = rotation_error_deg(estimate.transform[:3, :3], rotation)
    trans_error = np.linalg.norm(estimate.transform[:3, 3] - translation)
    print(f"inliers={estimate.inliers.sum()}/{len(source)}")
    print(f"weighted_rmse_m={estimate.weighted_rmse_m:.6f}")
    print(f"rotation_error_deg={rot_error:.4f}")
    print(f"translation_error_m={trans_error:.6f}")
    if rot_error > 0.5 or trans_error > 0.01:
        raise SystemExit("Synthetic pose recovery failed")

    if unweighted is not None:
        unweighted_rot_error = rotation_error_deg(
            unweighted.transform[:3, :3], rotation
        )
        unweighted_trans_error = np.linalg.norm(
            unweighted.transform[:3, 3] - translation
        )
        print(f"unweighted_inliers={unweighted.inliers.sum()}/{len(source)}")
        print(
            "unweighted_rmse_m="
            f"{np.sqrt(np.mean(unweighted.residuals_m[unweighted.inliers] ** 2)):.6f}"
        )
        print(f"unweighted_rotation_error_deg={unweighted_rot_error:.4f}")
        print(f"unweighted_translation_error_m={unweighted_trans_error:.6f}")


if __name__ == "__main__":
    main()
