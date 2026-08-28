import numpy as np
import pytest

from rgbd_pose.features import (
    build_feature_mask,
    extract_opencv_features,
    polygon_mask,
    rectangular_roi_mask,
)


def test_rectangular_roi_mask_has_expected_bounds():
    mask = rectangular_roi_mask((20, 30, 3), (5, 4, 17, 15))

    assert mask.shape == (20, 30)
    assert mask.dtype == bool
    assert mask.sum() == (17 - 5) * (15 - 4)
    assert mask[4, 5]
    assert not mask[15, 5]
    assert not mask[4, 17]


def test_sift_keypoints_stay_inside_roi():
    rng = np.random.default_rng(12)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[25:100, 35:130] = rng.integers(0, 256, (75, 95, 3), dtype=np.uint8)
    roi = (35, 25, 130, 100)
    features = extract_opencv_features(
        image, "sift", mask=rectangular_roi_mask(image.shape, roi)
    )

    assert len(features.uv) > 0
    assert np.all(features.uv[:, 0] >= roi[0])
    assert np.all(features.uv[:, 0] < roi[2])
    assert np.all(features.uv[:, 1] >= roi[1])
    assert np.all(features.uv[:, 1] < roi[3])


def test_roi_validation_rejects_empty_or_out_of_bounds_rectangle():
    with pytest.raises(ValueError, match="must lie inside"):
        rectangular_roi_mask((20, 30), (0, 0, 0, 10))
    with pytest.raises(ValueError, match="must lie inside"):
        rectangular_roi_mask((20, 30), (0, 0, 31, 10))


def test_polygon_mask_contains_polygon_and_preserves_image_shape():
    mask = polygon_mask(
        (20, 30, 3),
        [(5, 4), (17, 4), (21, 15), (8, 17)],
    )

    assert mask.shape == (20, 30)
    assert mask.dtype == bool
    assert mask[8, 10]
    assert not mask[2, 10]
    assert not mask[18, 10]
    assert not mask[8, 25]


def test_polygon_mask_accepts_flat_coordinates_and_rejects_invalid_vertices():
    mask = polygon_mask((10, 12), [1, 1, 8, 1, 8, 8, 1, 8])

    assert mask[4, 4]
    with pytest.raises(ValueError, match="at least three"):
        polygon_mask((10, 12), [1, 1, 8, 1])
    with pytest.raises(ValueError, match="inside image bounds"):
        polygon_mask((10, 12), [1, 1, 12, 1, 8, 8])


def test_build_feature_mask_intersects_rectangle_and_polygon():
    mask = build_feature_mask(
        (12, 16),
        roi_xyxy=(3, 2, 13, 10),
        polygon_xy=[1, 1, 8, 1, 8, 8, 1, 8],
    )

    assert mask[5, 5]
    assert not mask[5, 2]
    assert not mask[5, 10]
    assert not mask[1, 5]
