import numpy as np
import pytest

from rgbd_pose.dino_quality import feature_mask_kind, summarize_dino_quality
from rgbd_pose.features import ImageFeatures


def _features(selected: int = 3, candidates: int = 12) -> ImageFeatures:
    return ImageFeatures(
        np.zeros((selected, 2), dtype=np.float32),
        np.ones((selected, 4), dtype=np.float32),
        candidate_count=candidates,
        grid_shape=(3, 4),
    )


def test_feature_mask_kind_distinguishes_roi_and_polygon():
    assert feature_mask_kind(None, None) == "none"
    assert feature_mask_kind((0, 0, 2, 2), None) == "roi"
    assert feature_mask_kind(None, [0, 0, 2, 0, 1, 2]) == "polygon"
    assert feature_mask_kind((0, 0, 2, 2), [0, 0, 2, 0, 1, 2]) == "roi_and_polygon"


def test_dino_quality_reports_patch_coverage_and_geometry():
    reference_mask = np.zeros((4, 4), dtype=bool)
    reference_mask[:2, :2] = True
    query_mask = np.ones((4, 4), dtype=bool)

    report = summarize_dino_quality(
        reference_features=_features(),
        query_features=_features(6),
        reference_mask=reference_mask,
        query_mask=query_mask,
        reference_roi=None,
        query_roi=(0, 0, 4, 4),
        reference_polygon=[0, 0, 3, 0, 3, 3],
        query_polygon=None,
        depth_valid_matches=8,
        valid_3d_matches=7,
        not_used_3d_matches=1,
        inliers=6,
        inlier_ratio=6 / 7,
        weighted_rmse_m=0.006,
        ransac_threshold_m=0.03,
    )

    assert report["reference"]["mask_type"] == "polygon"
    assert report["query"]["mask_type"] == "roi"
    assert report["reference"]["candidate_patch_count"] == 12
    assert report["reference"]["selected_patch_count"] == 3
    assert report["reference"]["selected_patch_ratio"] == pytest.approx(0.25)
    assert report["reference"]["mask_pixel_area_ratio"] == pytest.approx(0.25)
    assert report["query"]["mask_pixel_area_ratio"] == pytest.approx(1.0)
    assert report["geometry"]["rmse_to_ransac_threshold_ratio"] == pytest.approx(0.2)
    assert report["query"]["background_leakage_risk"] == "elevated"


def test_dino_quality_rejects_non_2d_mask():
    with pytest.raises(ValueError, match="two-dimensional"):
        summarize_dino_quality(
            reference_features=_features(),
            query_features=_features(),
            reference_mask=np.ones((2, 2, 1), dtype=bool),
            query_mask=None,
            reference_roi=(0, 0, 2, 2),
            query_roi=None,
            reference_polygon=None,
            query_polygon=None,
            depth_valid_matches=0,
            valid_3d_matches=0,
            not_used_3d_matches=0,
        )
