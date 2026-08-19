import numpy as np
import pytest

from rgbd_pose.features import extract_opencv_features, rectangular_roi_mask


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
