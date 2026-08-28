"""Read-only visual review artifacts for sequence-quality reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .realsense_sequence import load_realsense_sequence_frame


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse review manifest: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("review manifest must contain a JSON object")
    return value


def _manifest_frame_indices(manifest: Mapping[str, Any]) -> set[int]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("review manifest must contain a frames list")
    indices: set[int] = set()
    for item in frames:
        if not isinstance(item, Mapping) or item.get("frame_index") is None:
            raise ValueError("review manifest contains an invalid frame entry")
        try:
            indices.add(int(item["frame_index"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("review manifest contains a non-integer frame index") from exc
    return indices


def validate_approved_pair(
    manifest_path: str | Path,
    reference_index: int,
    query_index: int,
) -> dict[str, Any]:
    """Require an explicitly approved pair in a review manifest."""
    if reference_index == query_index:
        raise ValueError("reference and query frame indices must differ")
    manifest = _read_manifest(manifest_path)
    if manifest.get("review_status") != "APPROVED":
        raise ValueError(
            "review manifest is not APPROVED; visual review is required before "
            "using this pair"
        )
    available = _manifest_frame_indices(manifest)
    requested = {int(reference_index), int(query_index)}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"approved review manifest does not contain frame(s): {missing}"
        )
    approved = manifest.get("approved_frame_indices")
    if not isinstance(approved, list):
        raise ValueError("approved review manifest must contain approved_frame_indices")
    try:
        approved_set = {int(index) for index in approved}
    except (TypeError, ValueError) as exc:
        raise ValueError("approved_frame_indices must contain integers") from exc
    missing_approval = sorted(requested - approved_set)
    if missing_approval:
        raise ValueError(
            f"requested frame(s) are not approved: {missing_approval}"
        )
    selected_pair = manifest.get("selected_pair")
    if selected_pair is not None:
        if not isinstance(selected_pair, Mapping):
            raise ValueError("selected_pair must be an object when present")
        if (
            int(selected_pair.get("reference_index", -1)) != int(reference_index)
            or int(selected_pair.get("query_index", -1)) != int(query_index)
        ):
            raise ValueError("requested pair does not match manifest.selected_pair")
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "review_status": "APPROVED",
        "reference_index": int(reference_index),
        "query_index": int(query_index),
        "decision_notes": manifest.get("decision_notes", ""),
    }


def approve_review_manifest(
    manifest_path: str | Path,
    reference_index: int,
    query_index: int,
    decision_notes: str,
) -> dict[str, Any]:
    """Mark one reviewed frame pair as approved in a generated manifest."""
    if not decision_notes.strip():
        raise ValueError("decision_notes must not be empty")
    if reference_index == query_index:
        raise ValueError("reference and query frame indices must differ")
    manifest = _read_manifest(manifest_path)
    available = _manifest_frame_indices(manifest)
    missing = sorted({int(reference_index), int(query_index)} - available)
    if missing:
        raise ValueError(
            f"cannot approve frame(s) not present in the manifest: {missing}"
        )
    manifest["review_status"] = "APPROVED"
    manifest["approved_frame_indices"] = [int(reference_index), int(query_index)]
    manifest["selected_pair"] = {
        "reference_index": int(reference_index),
        "query_index": int(query_index),
    }
    manifest["decision_notes"] = decision_notes.strip()
    manifest["approved_at_utc"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _add_role(selected: dict[int, list[str]], index: int, role: str) -> None:
    selected.setdefault(int(index), [])
    if role not in selected[int(index)]:
        selected[int(index)].append(role)


def _add_interval_representatives(
    selected: dict[int, list[str]],
    interval: Mapping[str, Any],
    prefix: str,
) -> None:
    try:
        start = int(interval["start_index"])
        end = int(interval["end_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quality report contains an invalid frame interval") from exc
    if start > end:
        raise ValueError("quality report contains a reversed frame interval")
    middle = (start + end) // 2
    _add_role(selected, start, f"{prefix}_start")
    _add_role(selected, middle, f"{prefix}_middle")
    _add_role(selected, end, f"{prefix}_end")


def select_review_frames(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select deterministic representative frames from a motion report.

    Pair endpoints are retained when available.  Every motion candidate and
    temporal-stability interval contributes its first, middle, and last frame.
    The result contains frame indices and semantic roles only; it does not
    load or modify any session files.
    """
    motion = report.get("motion_scan")
    if not isinstance(motion, Mapping):
        raise ValueError("quality report does not contain a motion_scan object")
    selected: dict[int, list[str]] = {}
    pair = report.get("pair_check")
    if isinstance(pair, Mapping):
        if pair.get("reference_index") is not None:
            _add_role(selected, int(pair["reference_index"]), "pair_reference")
        if pair.get("query_index") is not None:
            _add_role(selected, int(pair["query_index"]), "pair_query")

    for position, interval in enumerate(motion.get("motion_candidate_intervals", [])):
        if not isinstance(interval, Mapping):
            raise ValueError("quality report contains an invalid motion interval")
        _add_interval_representatives(selected, interval, f"motion_{position}")
    for position, interval in enumerate(motion.get("stable_segments", [])):
        if not isinstance(interval, Mapping):
            raise ValueError("quality report contains an invalid stable interval")
        _add_interval_representatives(selected, interval, f"stable_{position}")

    return [
        {"frame_index": index, "roles": roles}
        for index, roles in sorted(selected.items())
    ]


def _limit_review_frames(
    frames: list[dict[str, Any]], max_frames: int
) -> list[dict[str, Any]]:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if len(frames) <= max_frames:
        return frames
    required = [
        item
        for item in frames
        if any(role.startswith("pair_") for role in item["roles"])
    ]
    optional = [item for item in frames if item not in required]
    remaining = max_frames - len(required)
    if remaining <= 0:
        return sorted(required[:max_frames], key=lambda item: item["frame_index"])
    positions = np.linspace(
        0, len(optional) - 1, min(remaining, len(optional)), dtype=int
    )
    chosen = required + [optional[int(position)] for position in positions]
    return sorted(
        {item["frame_index"]: item for item in chosen}.values(),
        key=lambda item: item["frame_index"],
    )


def _depth_preview(depth_m: np.ndarray, valid: np.ndarray) -> Image.Image:
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    values = depth_m[valid]
    if values.size:
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.round(scaled[valid] * 255.0).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _role_color(roles: list[str]) -> tuple[int, int, int]:
    if any(role.startswith("motion_") for role in roles):
        return (220, 60, 60)
    if any(role.startswith("stable_") for role in roles):
        return (50, 175, 75)
    return (55, 110, 220)


def _contact_sheet(
    items: list[dict[str, Any]], output_dir: Path, kind: str
) -> Path:
    tile_width, tile_height = 340, 280
    columns = 4
    rows = max(1, (len(items) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202020")
    draw = ImageDraw.Draw(sheet)
    for position, item in enumerate(items):
        image = Image.open(output_dir / item[f"{kind}_path"]).convert("RGB")
        image.thumbnail((320, 235), Image.Resampling.LANCZOS)
        border = ImageOps.expand(image, border=4, fill=_role_color(item["roles"]))
        x = (position % columns) * tile_width + 10
        y = (position // columns) * tile_height + 8
        sheet.paste(border, (x, y))
        label = f"{item['frame_index']}: {', '.join(item['roles'][:2])}"
        draw.rectangle((x, y + 243, x + 320, y + 269), fill="#000000")
        draw.text((x + 6, y + 248), label, fill="#ffffff")
    path = output_dir / f"{kind}_contact_sheet.png"
    sheet.save(path)
    return path


def export_motion_review(
    session_dir: str | Path,
    quality_report_path: str | Path,
    output_dir: str | Path,
    *,
    max_frames: int = 30,
) -> dict[str, Any]:
    """Export read-only RGB/depth review images and a pending manifest."""
    session = Path(session_dir)
    report_path = Path(quality_report_path)
    output = Path(output_dir)
    if not session.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse quality report: {report_path}") from exc
    if not isinstance(report, dict):
        raise ValueError("quality report must contain a JSON object")

    all_frames = select_review_frames(report)
    selected_frames = _limit_review_frames(all_frames, max_frames)
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frames"
    frame_dir.mkdir(exist_ok=True)
    items: list[dict[str, Any]] = []
    for selected in selected_frames:
        index = int(selected["frame_index"])
        frame = load_realsense_sequence_frame(session, index)
        stem = f"frame_{index:06d}"
        rgb_path = frame_dir / f"{stem}_rgb.png"
        depth_path = frame_dir / f"{stem}_depth_preview.png"
        Image.fromarray(frame.rgb, mode="RGB").save(rgb_path)
        _depth_preview(frame.depth_m, frame.valid_depth_mask).save(depth_path)
        items.append(
            {
                "frame_index": index,
                "roles": selected["roles"],
                "rgb_path": str(rgb_path.relative_to(output)),
                "depth_path": str(depth_path.relative_to(output)),
                "timestamp_difference_ms": frame.timestamp_difference_ms,
                "invalid_depth_ratio": frame.invalid_depth_ratio,
                "source_rgb": str(frame.rgb_path),
                "source_depth": str(frame.depth_path),
            }
        )

    rgb_sheet = _contact_sheet(items, output, "rgb")
    depth_sheet = _contact_sheet(items, output, "depth")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "session": str(session.resolve()),
        "quality_report": str(report_path.resolve()),
        "review_status": "PENDING_MANUAL_REVIEW",
        "approved_frame_indices": [],
        "rejected_frame_indices": [],
        "selection": {
            "available_selected_frames": len(all_frames),
            "exported_frames": len(items),
            "omitted_frames": max(0, len(all_frames) - len(items)),
            "max_frames": max_frames,
        },
        "contact_sheets": {
            "rgb": str(rgb_sheet.relative_to(output)),
            "depth": str(depth_sheet.relative_to(output)),
        },
        "frames": items,
    }
    (output / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
