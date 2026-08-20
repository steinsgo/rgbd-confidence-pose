import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rgbd_pose.quality_review import (
    approve_review_manifest,
    export_motion_review,
    select_review_frames,
    validate_approved_pair,
)
from rgbd_pose.sequence_quality import _motion_profile, screen_realsense_session


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


def test_quality_report_records_valid_pair_polygon(tmp_path):
    session = _write_quality_session(tmp_path)

    polygon = [0, 0, 1, 0, 1, 1, 0, 1]
    report = screen_realsense_session(
        session,
        reference_index=0,
        query_index=1,
        check_features=False,
        reference_polygon=polygon,
        query_polygon=polygon,
    )

    assert report["pair_check"]["reference_polygon_xy"] == polygon
    assert report["pair_check"]["query_polygon_xy"] == polygon


def test_quality_report_rejects_invalid_pair_polygon(tmp_path):
    session = _write_quality_session(tmp_path)

    report = screen_realsense_session(
        session,
        reference_index=0,
        query_index=1,
        check_features=False,
        reference_polygon=[0, 0, 1, 0],
        query_polygon=[0, 0, 1, 0, 1, 1],
    )

    assert report["pair_status"] == "REJECT"
    assert any(issue["code"] == "polygon_invalid" for issue in report["issues"])


def test_motion_profile_reports_candidate_changes_and_stable_segments():
    frames = []
    for index in range(12):
        value = 255 if 4 <= index <= 5 else 0
        rgb = np.full((32, 32, 3), value, dtype=np.uint8)
        frames.append(SimpleNamespace(frame_index=index, rgb=rgb))

    profile = _motion_profile(
        frames,
        comparison_window=1,
        motion_quantile=0.5,
        minimum_score=0.01,
        minimum_stable_frames=3,
    )

    assert profile["motion_trigger_frame_indices"] == [4, 6]
    assert profile["manual_review_required"] is True
    assert "not a hand-free" in profile["stable_segment_interpretation"]
    assert profile["motion_candidate_intervals"] == [
        {"start_index": 4, "end_index": 4, "frame_count": 1, "trigger_count": 1},
        {"start_index": 6, "end_index": 6, "frame_count": 1, "trigger_count": 1},
    ]
    assert profile["stable_segments"] == [
        {"start_index": 0, "end_index": 3, "frame_count": 4},
        {"start_index": 7, "end_index": 11, "frame_count": 5},
    ]


def test_quality_report_can_run_motion_scan_without_pair_check(tmp_path):
    session = _write_quality_session(tmp_path)

    report = screen_realsense_session(
        session,
        check_features=False,
        motion_scan=True,
        motion_window=1,
        motion_quantile=0.5,
        min_stable_frames=1,
    )

    assert report["recording"]["depth"]["decode_mode"] == "full"
    assert report["manual_review_required"] is False
    assert report["motion_scan"]["stable_segment_count"] == 1
    assert report["motion_scan"]["motion_candidate_intervals"] == []


def test_review_frame_selection_keeps_pair_and_interval_representatives():
    report = {
        "pair_check": {"reference_index": 10, "query_index": 90},
        "motion_scan": {
            "motion_candidate_intervals": [
                {"start_index": 30, "end_index": 34}
            ],
            "stable_segments": [
                {"start_index": 0, "end_index": 20}
            ],
        },
    }

    selected = select_review_frames(report)

    assert [item["frame_index"] for item in selected] == [0, 10, 20, 30, 32, 34, 90]
    assert "pair_reference" in selected[1]["roles"]
    assert "motion_0_middle" in selected[4]["roles"]


def test_motion_review_export_writes_manifest_and_contact_sheets(tmp_path):
    session = _write_quality_session(tmp_path)
    report_path = tmp_path / "quality.json"
    report_path.write_text(
        json.dumps(
            {
                "session": str(session),
                "motion_scan": {
                    "motion_candidate_intervals": [],
                    "stable_segments": [
                        {"start_index": 0, "end_index": 1}
                    ],
                },
                "pair_check": None,
            }
        ),
        encoding="utf-8",
    )

    manifest = export_motion_review(session, report_path, tmp_path / "review")

    assert manifest["review_status"] == "PENDING_MANUAL_REVIEW"
    assert manifest["selection"]["exported_frames"] == 2
    assert (tmp_path / "review" / "rgb_contact_sheet.png").is_file()
    assert (tmp_path / "review" / "depth_contact_sheet.png").is_file()
    assert (tmp_path / "review" / "review_manifest.json").is_file()


def test_review_manifest_requires_explicit_approval(tmp_path):
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "review_status": "PENDING_MANUAL_REVIEW",
                "approved_frame_indices": [],
                "frames": [{"frame_index": 0}, {"frame_index": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not APPROVED"):
        validate_approved_pair(manifest_path, 0, 1)

    approved = approve_review_manifest(
        manifest_path, 0, 1, "visually checked; no hand in either selected frame"
    )
    validated = validate_approved_pair(manifest_path, 0, 1)

    assert approved["review_status"] == "APPROVED"
    assert validated["reference_index"] == 0
    assert validated["query_index"] == 1


def test_quality_report_rejects_pending_review_manifest(tmp_path):
    session = _write_quality_session(tmp_path)
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "review_status": "PENDING_MANUAL_REVIEW",
                "approved_frame_indices": [],
                "frames": [{"frame_index": 0}, {"frame_index": 1}],
            }
        ),
        encoding="utf-8",
    )

    report = screen_realsense_session(
        session,
        reference_index=0,
        query_index=1,
        check_features=False,
        review_manifest=manifest_path,
    )

    assert report["pair_status"] == "REJECT"
    assert any(issue["code"] == "review_not_approved" for issue in report["issues"])
