import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rgbd_pose.realsense_sequence import load_realsense_sequence_frame


FRAME_FIELDS = [
    "index",
    "color_file",
    "depth_file",
    "arrival_time_ns",
    "color_timestamp_ms",
    "depth_timestamp_ms",
    "color_frame_number",
    "depth_frame_number",
    "color_timestamp_domain",
    "depth_timestamp_domain",
]


def _write_session(root: Path) -> Path:
    session = root / "session"
    (session / "color").mkdir(parents=True)
    (session / "depth").mkdir()

    rgb_ten = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [12, 34, 56]]],
        dtype=np.uint8,
    )
    rgb_thirty = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    depth_ten = np.array([[1000, 0], [65535, 2500]], dtype=np.uint16)
    depth_thirty = np.array([[2000, 3000], [4000, 5000]], dtype=np.uint16)
    Image.fromarray(rgb_ten).save(session / "color" / "ten.png")
    Image.fromarray(rgb_thirty).save(session / "color" / "thirty.png")
    Image.fromarray(depth_ten).save(session / "depth" / "ten.png")
    Image.fromarray(depth_thirty).save(session / "depth" / "thirty.png")

    calibration = {
        "depth_scale_meters_per_unit": 0.001,
        "color_intrinsics": {
            "width": 2,
            "height": 2,
            "fx": 100.0,
            "fy": 101.0,
            "ppx": 0.5,
            "ppy": 0.75,
            "distortion_model": "distortion.inverse_brown_conrady",
            "coeffs": [0, 0, 0, 0, 0],
        },
        "aligned_depth": {
            "width": 2,
            "height": 2,
            "pixel_coordinate_system": "color_camera",
            "projection_intrinsics": "color_intrinsics",
            "stored_values": "original uint16 depth units",
        },
    }
    (session / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")

    rows = [
        {
            "index": "10",
            "color_file": "color/ten.png",
            "depth_file": "depth/ten.png",
            "arrival_time_ns": "1000000000",
            "color_timestamp_ms": "100.0",
            "depth_timestamp_ms": "99.5",
            "color_frame_number": "10",
            "depth_frame_number": "11",
            "color_timestamp_domain": "timestamp_domain.system_time",
            "depth_timestamp_domain": "timestamp_domain.system_time",
        },
        {
            "index": "30",
            "color_file": "color/thirty.png",
            "depth_file": "depth/thirty.png",
            "arrival_time_ns": "3000000000",
            "color_timestamp_ms": "300.0",
            "depth_timestamp_ms": "301.25",
            "color_frame_number": "30",
            "depth_frame_number": "31",
            "color_timestamp_domain": "timestamp_domain.system_time",
            "depth_timestamp_domain": "timestamp_domain.system_time",
        },
    ]
    with (session / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return session


def test_loader_uses_csv_pairing_and_parses_rgb_depth(tmp_path):
    session = _write_session(tmp_path)

    frame = load_realsense_sequence_frame(session, 30)

    assert frame.frame_index == 30
    assert frame.rgb.shape == (2, 2, 3)
    assert frame.rgb.dtype == np.uint8
    np.testing.assert_array_equal(frame.rgb[0, 0], [1, 2, 3])
    assert frame.depth_m.shape == (2, 2)
    assert frame.depth_m.dtype == np.float32
    np.testing.assert_allclose(frame.depth_m, [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(
        frame.intrinsics,
        [[100.0, 0.0, 0.5], [0.0, 101.0, 0.75], [0.0, 0.0, 1.0]],
    )
    assert frame.rgb_timestamp_ms == 300.0
    assert frame.depth_timestamp_ms == 301.25
    assert frame.timestamp_delta_ms == -1.25
    assert frame.timestamp_difference_ms == 1.25
    assert frame.rgb_path.name == "thirty.png"
    assert frame.depth_path.name == "thirty.png"


def test_loader_marks_zero_and_uint16_max_as_invalid(tmp_path):
    session = _write_session(tmp_path)

    frame = load_realsense_sequence_frame(session, 10)

    np.testing.assert_allclose(frame.depth_m, [[1.0, 0.0], [0.0, 2.5]])
    np.testing.assert_array_equal(frame.valid_depth_mask, [[True, False], [False, True]])
    assert frame.invalid_depth_count == 2
    assert frame.invalid_depth_ratio == 0.5


def test_loader_rejects_unavailable_frame_index(tmp_path):
    session = _write_session(tmp_path)

    with pytest.raises(IndexError, match="Frame index 11 is unavailable"):
        load_realsense_sequence_frame(session, 11)


def test_loader_reports_missing_referenced_file(tmp_path):
    session = _write_session(tmp_path)
    (session / "depth" / "thirty.png").unlink()

    with pytest.raises(FileNotFoundError, match="Missing depth frame"):
        load_realsense_sequence_frame(session, 30)


def test_loader_rejects_intrinsic_image_resolution_mismatch(tmp_path):
    session = _write_session(tmp_path)
    calibration_path = session / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["color_intrinsics"]["width"] = 3
    calibration["aligned_depth"]["width"] = 3
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match="Decoded image resolution"):
        load_realsense_sequence_frame(session, 30)
