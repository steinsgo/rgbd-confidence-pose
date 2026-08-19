from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Matches2D:
    reference_uv: np.ndarray
    query_uv: np.ndarray
    confidence: np.ndarray


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return a @ b.T


def mutual_nearest_matches(
    ref_features: np.ndarray,
    query_features: np.ndarray,
    ref_uv: np.ndarray,
    query_uv: np.ndarray,
    min_similarity: float = 0.55,
    top_fraction: float = 0.35,
) -> Matches2D:
    """Match features with mutual-NN and a similarity-derived confidence."""
    sim = cosine_similarity_matrix(ref_features, query_features)
    ref_to_query = np.argmax(sim, axis=1)
    query_to_ref = np.argmax(sim, axis=0)
    ref_index = np.arange(len(ref_features))
    mutual = query_to_ref[ref_to_query] == ref_index
    score = sim[ref_index, ref_to_query]
    keep = mutual & (score >= min_similarity)
    selected = np.flatnonzero(keep)
    if len(selected) == 0:
        return Matches2D(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

    count = max(3, int(np.ceil(len(selected) * np.clip(top_fraction, 0.0, 1.0))))
    selected = selected[np.argsort(score[selected])[-count:]]
    raw = score[selected]
    confidence = np.clip((raw - min_similarity) / max(1.0 - min_similarity, 1e-6), 0, 1)
    return Matches2D(
        np.asarray(ref_uv)[selected],
        np.asarray(query_uv)[ref_to_query[selected]],
        confidence,
    )


def combine_confidence(
    appearance: np.ndarray,
    ref_depth_valid: np.ndarray,
    query_depth_valid: np.ndarray,
) -> np.ndarray:
    validity = np.asarray(ref_depth_valid, bool) & np.asarray(query_depth_valid, bool)
    return np.asarray(appearance, np.float64) * validity.astype(np.float64)
