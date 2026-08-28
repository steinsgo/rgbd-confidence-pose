#!/usr/bin/env python3
"""Summarize DINOv2 pair-quality metadata without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON run output: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run output must contain a JSON object: {path}")
    if payload.get("backend") != "dino":
        raise ValueError(f"run output is not a DINOv2 result: {path}")
    quality = payload.get("dino_quality")
    if not isinstance(quality, dict):
        raise ValueError(
            f"run output has no dino_quality metadata; rerun with the current code: {path}"
        )
    return payload


def _pair_label(payload: dict[str, object]) -> str:
    reference = payload.get("reference")
    query = payload.get("query")
    if isinstance(reference, dict) and isinstance(query, dict):
        return f"{reference.get('frame_index')}->{query.get('frame_index')}"
    return "custom-frames"


def _patch_label(side: dict[str, object]) -> str:
    return (
        f"{side.get('selected_patch_count')}/"
        f"{side.get('candidate_patch_count')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    for path in args.run_json:
        payload = _load_report(path)
        quality = payload["dino_quality"]
        assert isinstance(quality, dict)
        geometry = quality.get("geometry", {})
        assert isinstance(geometry, dict)
        runs.append(
            {
                "source": str(path.resolve()),
                "pair": _pair_label(payload),
                "reference": quality.get("reference"),
                "query": quality.get("query"),
                "geometry": geometry,
                "warnings": quality.get("warnings", []),
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "interpretation": (
            "descriptive_only; mask risk is heuristic and no ground-truth pose "
            "accuracy is inferred"
        ),
        "runs": runs,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print("pair mask(ref/query) patches(ref/query) valid3d inliers ratio rmse_m warnings")
    for item in runs:
        reference = item["reference"]
        query = item["query"]
        geometry = item["geometry"]
        assert isinstance(reference, dict)
        assert isinstance(query, dict)
        assert isinstance(geometry, dict)
        warnings = "; ".join(str(value) for value in item["warnings"])
        print(
            f"{item['pair']} "
            f"{reference.get('mask_type')}/{query.get('mask_type')} "
            f"{_patch_label(reference)}/{_patch_label(query)} "
            f"{geometry.get('valid_3d_matches')} "
            f"{geometry.get('inliers', '-') } "
            f"{geometry.get('inlier_ratio', '-') } "
            f"{geometry.get('weighted_rmse_m', '-') } "
            f"{warnings}"
        )


if __name__ == "__main__":
    main()
