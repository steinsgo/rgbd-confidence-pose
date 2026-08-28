import numpy as np
import pytest

from rgbd_pose.features import intersect_feature_masks, load_feature_mask
from rgbd_pose.plane_mask import fit_plane_from_pixels, tabletop_foreground_mask


def _intrinsics() -> np.ndarray:
    return np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_tabletop_plane_mask_keeps_a_point_above_the_plane():
    depth = np.ones((5, 5), dtype=np.float32)
    depth[2, 2] = 0.9
    depth[0, 0] = 0.0
    plane = fit_plane_from_pixels(
        np.array([[4, 0], [0, 4], [4, 4]], dtype=np.float64),
        depth,
        _intrinsics(),
    )

    mask = tabletop_foreground_mask(
        depth, _intrinsics(), plane, min_height_m=0.05
    )

    assert mask.dtype == bool
    assert mask[2, 2]
    assert not mask[0, 0]
    assert int(mask.sum()) == 1


def test_plane_fit_rejects_collinear_reference_pixels():
    depth = np.ones((5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="collinear"):
        fit_plane_from_pixels(
            np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64),
            depth,
            _intrinsics(),
        )


def test_feature_mask_loader_and_intersection(tmp_path):
    from PIL import Image

    first = np.zeros((4, 5), dtype=np.uint8)
    first[1:3, 1:4] = 255
    second = np.zeros((4, 5), dtype=np.uint8)
    second[2:, 2:] = 255
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(first, mode="L").save(first_path)
    Image.fromarray(second, mode="L").save(second_path)

    loaded_first = load_feature_mask(first_path, (4, 5, 3))
    loaded_second = load_feature_mask(second_path, (4, 5, 3))
    combined = intersect_feature_masks(loaded_first, loaded_second)

    assert loaded_first.dtype == bool
    assert combined.sum() == 2
    assert combined[2, 2]
    assert combined[2, 3]
    assert not combined[1, 2]


def test_feature_mask_loader_rejects_resolution_mismatch(tmp_path):
    from PIL import Image

    path = tmp_path / "mask.png"
    Image.fromarray(np.ones((3, 4), dtype=np.uint8), mode="L").save(path)

    with pytest.raises(ValueError, match="resolution mismatch"):
        load_feature_mask(path, (4, 4, 3))
