from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_FRAME_FIELDS = {
    "index",
    "color_file",
    "depth_file",
    "color_timestamp_ms",
    "depth_timestamp_ms",
}
_INTRINSIC_FIELDS = ("width", "height", "fx", "fy", "ppx", "ppy")


@dataclass(frozen=True)
class RealSenseFrame:
    """One paired, aligned RGB-D frame from a recorded D435i sequence.

    RGB images are decoded with Pillow and returned in the order of the
    decoded RGB array. The recorder's ``BGR8`` metadata describes the camera
    capture buffer; it is not an instruction to swap a Pillow-decoded PNG a
    second time. Raw depth values 0 and 65535 are represented as 0.0 metres
    and marked false in ``valid_depth_mask``.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    frame_index: int
    rgb_timestamp_ms: float
    depth_timestamp_ms: float
    timestamp_delta_ms: float
    timestamp_difference_ms: float
    rgb_path: Path
    depth_path: Path
    calibration_path: Path
    frames_csv_path: Path
    valid_depth_mask: np.ndarray
    invalid_depth_count: int
    invalid_depth_ratio: float
    depth_scale_meters_per_unit: float

    @property
    def source_paths(self) -> dict[str, Path]:
        return {
            "rgb": self.rgb_path,
            "depth": self.depth_path,
            "calibration": self.calibration_path,
            "frames": self.frames_csv_path,
        }


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _load_calibration(path: Path) -> tuple[np.ndarray, float]:
    _required_file(path, "calibration.json")
    try:
        calibration: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse calibration JSON: {path}") from exc

    try:
        scale = float(calibration["depth_scale_meters_per_unit"])
        color_intrinsics = calibration["color_intrinsics"]
        aligned_depth = calibration["aligned_depth"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "calibration.json must contain depth_scale_meters_per_unit, "
            "color_intrinsics, and aligned_depth"
        ) from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_meters_per_unit must be finite and positive")
    if not isinstance(color_intrinsics, dict) or not isinstance(aligned_depth, dict):
        raise ValueError("color_intrinsics and aligned_depth must be objects")

    missing = [field for field in _INTRINSIC_FIELDS if field not in color_intrinsics]
    if missing:
        raise ValueError(f"color_intrinsics is missing fields: {', '.join(missing)}")
    try:
        width = int(color_intrinsics["width"])
        height = int(color_intrinsics["height"])
        fx = float(color_intrinsics["fx"])
        fy = float(color_intrinsics["fy"])
        ppx = float(color_intrinsics["ppx"])
        ppy = float(color_intrinsics["ppy"])
    except (TypeError, ValueError) as exc:
        raise ValueError("color_intrinsics contains non-numeric values") from exc
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("color_intrinsics has invalid resolution or focal length")
    if any(not np.isfinite(value) for value in (fx, fy, ppx, ppy)):
        raise ValueError("color_intrinsics contains non-finite values")

    aligned_width = aligned_depth.get("width")
    aligned_height = aligned_depth.get("height")
    if aligned_width != width or aligned_height != height:
        raise ValueError(
            "aligned_depth resolution must match color_intrinsics resolution: "
            f"{aligned_width}x{aligned_height} vs {width}x{height}"
        )
    if aligned_depth.get("projection_intrinsics") != "color_intrinsics":
        raise ValueError("aligned_depth must declare color_intrinsics projection")

    matrix = np.array(
        [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return matrix, scale


def _resolve_referenced_path(session_dir: Path, value: str, field: str) -> Path:
    if not value:
        raise ValueError(f"frames.csv field {field!r} is empty")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"frames.csv field {field!r} must be a relative path")
    session_resolved = session_dir.resolve()
    resolved = (session_dir / relative).resolve()
    if not resolved.is_relative_to(session_resolved):
        raise ValueError(f"frames.csv field {field!r} escapes the session directory")
    return resolved


def _load_frame_rows(path: Path) -> dict[int, dict[str, str]]:
    _required_file(path, "frames.csv")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("frames.csv has no header")
            missing = sorted(_FRAME_FIELDS - set(reader.fieldnames))
            if missing:
                raise ValueError(f"frames.csv is missing fields: {', '.join(missing)}")
            rows: dict[int, dict[str, str]] = {}
            for line_number, row in enumerate(reader, start=2):
                try:
                    index = int(row["index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid frame index at frames.csv line {line_number}"
                    ) from exc
                if index in rows:
                    raise ValueError(f"Duplicate frame index {index} in frames.csv")
                rows[index] = row
    except OSError as exc:
        raise FileNotFoundError(f"Could not read frames.csv: {path}") from exc
    if not rows:
        raise ValueError("frames.csv contains no frame rows")
    return rows


def load_realsense_sequence_frame(
    session_dir: str | Path, frame_index: int
) -> RealSenseFrame:
    """Load one aligned RGB-D frame without modifying the recording.

    ``frames.csv`` is authoritative for the RGB/depth pairing. The returned
    depth array is float32 metres with both raw sentinel values 0 and 65535
    replaced by 0.0; use ``valid_depth_mask`` to distinguish valid samples.
    """

    session = Path(session_dir)
    if not session.is_dir():
        raise NotADirectoryError(f"RealSense session directory not found: {session}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
        raise TypeError("frame_index must be an integer")
    frame_index = int(frame_index)

    calibration_path = session / "calibration.json"
    frames_csv_path = session / "frames.csv"
    intrinsics, depth_scale = _load_calibration(calibration_path)
    rows = _load_frame_rows(frames_csv_path)
    if frame_index not in rows:
        available = f"{min(rows)}..{max(rows)}"
        raise IndexError(f"Frame index {frame_index} is unavailable; available range is {available}")
    row = rows[frame_index]
    rgb_path = _resolve_referenced_path(session, row["color_file"], "color_file")
    depth_path = _resolve_referenced_path(session, row["depth_file"], "depth_file")
    _required_file(rgb_path, "RGB frame")
    _required_file(depth_path, "depth frame")

    with Image.open(rgb_path) as image:
        if image.mode != "RGB":
            raise ValueError(
                f"RGB frame must decode as an RGB PNG: {rgb_path} (mode={image.mode})"
            )
        rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(
            f"Decoded RGB frame must be HxWx3 uint8: {rgb_path} "
            f"(shape={rgb.shape}, dtype={rgb.dtype})"
        )

    with Image.open(depth_path) as image:
        raw_depth = np.asarray(image)
    if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
        raise ValueError(
            f"Decoded depth frame must be a 2-D uint16 PNG: {depth_path} "
            f"(shape={raw_depth.shape}, dtype={raw_depth.dtype})"
        )
    if raw_depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"RGB/depth resolution mismatch: {rgb.shape[:2]} vs {raw_depth.shape}"
        )

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    color_intrinsics = calibration["color_intrinsics"]
    expected_width = int(color_intrinsics["width"])
    expected_height = int(color_intrinsics["height"])
    if rgb.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "Decoded image resolution does not match color_intrinsics: "
            f"{rgb.shape[1]}x{rgb.shape[0]} vs {expected_width}x{expected_height}"
        )

    invalid_raw = (raw_depth == 0) | (raw_depth == np.iinfo(np.uint16).max)
    depth_m = raw_depth.astype(np.float32) * np.float32(depth_scale)
    depth_m[invalid_raw] = 0.0
    valid_depth_mask = ~invalid_raw
    invalid_count = int(invalid_raw.sum())
    invalid_ratio = float(invalid_count / raw_depth.size)

    try:
        rgb_timestamp_ms = float(row["color_timestamp_ms"])
        depth_timestamp_ms = float(row["depth_timestamp_ms"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamps for frame {frame_index}") from exc
    if not np.isfinite(rgb_timestamp_ms) or not np.isfinite(depth_timestamp_ms):
        raise ValueError(f"Non-finite timestamps for frame {frame_index}")
    timestamp_delta_ms = rgb_timestamp_ms - depth_timestamp_ms

    return RealSenseFrame(
        rgb=np.ascontiguousarray(rgb),
        depth_m=np.ascontiguousarray(depth_m),
        intrinsics=intrinsics,
        frame_index=frame_index,
        rgb_timestamp_ms=rgb_timestamp_ms,
        depth_timestamp_ms=depth_timestamp_ms,
        timestamp_delta_ms=float(timestamp_delta_ms),
        timestamp_difference_ms=float(abs(timestamp_delta_ms)),
        rgb_path=rgb_path,
        depth_path=depth_path,
        calibration_path=calibration_path.resolve(),
        frames_csv_path=frames_csv_path.resolve(),
        valid_depth_mask=np.ascontiguousarray(valid_depth_mask),
        invalid_depth_count=invalid_count,
        invalid_depth_ratio=invalid_ratio,
        depth_scale_meters_per_unit=depth_scale,
    )
