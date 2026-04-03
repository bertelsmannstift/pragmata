"""Deduplication helpers for synthetic query blueprints."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pragmata.core.schemas.querygen_plan import QueryBlueprint

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_BLUEPRINT_FIELD_ORDER: tuple[str, ...] = (
    "domain",
    "role",
    "language",
    "topic",
    "intent",
    "task",
    "difficulty",
    "format",
    "user_scenario",
    "information_need",
)


def _serialize_blueprint_content(candidate: QueryBlueprint) -> str:
    """Build a deterministic content-only serialization for a blueprint."""
    payload = {field: getattr(candidate, field) for field in _BLUEPRINT_FIELD_ORDER}
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def _blueprint_content_key(candidate: QueryBlueprint) -> str:
    """Return a deterministic exact-duplicate key for a blueprint."""
    serialized = _serialize_blueprint_content(candidate)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _select_non_duplicate_indices(
    similarities: NDArray[np.float32],
    threshold: float = 0.95,
) -> list[int]:
    """Select non-duplicate indices deterministically from a similarity matrix."""
    matrix = np.asarray(similarities, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarities must be a square 2D matrix")

    retained_indices: list[int] = []
    removed_indices: set[int] = set()

    for index in range(matrix.shape[0]):
        if index in removed_indices:
            continue

        retained_indices.append(index)

        for later_index in range(index + 1, matrix.shape[0]):
            if matrix[index, later_index] >= threshold:
                removed_indices.add(later_index)

    return retained_indices


@lru_cache(maxsize=None)
def _load_embedding_model(checkpoint: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model used for semantic blueprint deduplication."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for blueprint deduplication. Install pragmata with the 'querygen' extra."
        ) from exc

    return SentenceTransformer(checkpoint)


def _embed_blueprints(candidates: list[QueryBlueprint]) -> NDArray[np.float32]:
    """Embed blueprint content in one normalized batch."""
    if not candidates:
        return np.empty((0, 0), dtype=np.float32)

    serialized_candidates = [_serialize_blueprint_content(candidate) for candidate in candidates]
    model = _load_embedding_model()
    embeddings = model.encode(
        serialized_candidates,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def deduplicate_blueprints(candidates: list[QueryBlueprint]) -> list[QueryBlueprint]:
    """Remove exact and semantic near-duplicate blueprints deterministically."""
    if not candidates:
        return []

    exact_deduplicated: list[QueryBlueprint] = []
    seen_content_keys: set[str] = set()

    for candidate in candidates:
        content_key = _blueprint_content_key(candidate)
        if content_key in seen_content_keys:
            continue

        seen_content_keys.add(content_key)
        exact_deduplicated.append(candidate)

    if len(exact_deduplicated) <= 1:
        return exact_deduplicated

    embeddings = _embed_blueprints(exact_deduplicated)
    model = _load_embedding_model()
    similarities = np.asarray(
        model.similarity(embeddings, embeddings),
        dtype=np.float32,
    )
    retained_indices = _select_non_duplicate_indices(similarities)

    return [exact_deduplicated[index] for index in retained_indices]
