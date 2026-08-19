import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_pose.sequence_quality import screen_realsense_session


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


def _write_quality_session(
    root: Path,
    *,
    video_queue_drops: int = 0,
    metadata_status: str = "complete",
) -> Path:
    session = root / "session"
    (session / "color").mkdir(parents=True)
    (session / "depth").mkdir()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[0, 1] = [0, 255, 0]
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    for index in range(2):
        Image.fromarray(rgb).save(session / "color" / f"{index:06d}.png")
        Image.fromarray(depth).save(session / "depth" / f"{index:06d}.png")

    calibration = {
        "depth_scale_meters_per_unit": 0.001,
        "color_intrinsics": {
            "width": 2,
            "height": 2,
            "fx": 100.0,
            "fy": 100.0,
            "ppx": 0.5,
            "ppy": 0.5,
            "distortion_model": "none",
        },
        "aligned_depth": {
            "width": 2,
            "height": 2,
            "projection_intrinsics": "color_intrinsics",
        },
    }
    (session / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")

    rows = []
    for index in range(2):
        rows.append(
            {
                "index": str(index),
                "color_file": f"color/{index:06d}.png",
                "depth_file": f"depth/{index:06d}.png",
                "arrival_time_ns": str(index * 1_000_000),
                "color_timestamp_ms": str(index * 33.3),
                "depth_timestamp_ms": str(index * 33.3 + 1.0),
                "color_frame_number": str(index),
                "depth_frame_number": str(index),
                "color_timestamp_domain": "system",
                "depth_timestamp_domain": "system",
            }
        )
    with (session / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": "1.0",
        "status": metadata_status,
        "sequence_name": "synthetic",
        "device": {"name": "synthetic"},
        "counts": {
            "video_received": 2,
            "video_saved": 2,
            "video_queue_drops": video_queue_drops,
        },
        "writer_errors": [],
    }
    (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session


def test_quality_report_requires_an_explicit_pair_check(tmp_path):
    session = _write_quality_session(tmp_path)

    report = screen_realsense_session(session, check_features=False)

    assert report["recording_status"] == "PASS"
    assert report["pair_status"] == "NOT_RUN"
    assert report["status"] == "WARN"
    assert report["eligible_for_experiment"] is False
    assert any(issue["code"] == "pair_check_not_run" for issue in report["issues"])


def test_quality_report_rejects_video_queue_drops(tmp_path):
    session = _write_quality_session(tmp_path, video_queue_drops=1)

    report = screen_realsense_session(session, check_features=False)

    assert report["status"] == "REJECT"
    assert report["recording_status"] == "REJECT"
    assert any(issue["code"] == "video_queue_drops" for issue in report["issues"])


def test_quality_report_rejects_incomplete_metadata(tmp_path):
    session = _write_quality_session(tmp_path, metadata_status="aborted")

    report = screen_realsense_session(session, check_features=False)

    assert report["status"] == "REJECT"
    assert any(issue["code"] == "recording_incomplete" for issue in report["issues"])


def test_quality_report_rejects_missing_recorded_file(tmp_path):
    session = _write_quality_session(tmp_path)
    (session / "depth" / "000001.png").unlink()

    report = screen_realsense_session(session, check_features=False)

    assert report["status"] == "REJECT"
    assert any(issue["code"] == "referenced_file_missing" for issue in report["issues"])


def test_quality_report_rejects_invalid_pair_roi(tmp_path):
    session = _write_quality_session(tmp_path)

    report = screen_realsense_session(
        session,
        reference_index=0,
        query_index=1,
        check_features=False,
        reference_roi=(0, 0, 0, 2),
        query_roi=(0, 0, 2, 2),
    )

    assert report["pair_status"] == "REJECT"
    assert any(issue["code"] == "roi_invalid" for issue in report["issues"])


def test_quality_report_records_valid_pair_roi(tmp_path):
    session = _write_quality_session(tmp_path)

    report = screen_realsense_session(
        session,
        reference_index=0,
        query_index=1,
        check_features=False,
        reference_roi=(0, 0, 2, 2),
        query_roi=(0, 0, 2, 2),
    )

    assert report["pair_check"]["reference_roi_xyxy"] == [0, 0, 2, 2]
    assert report["pair_check"]["query_roi_xyxy"] == [0, 0, 2, 2]
