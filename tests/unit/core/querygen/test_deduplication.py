"""Unit tests for synthetic query blueprint deduplication helpers."""

import hashlib
import json

import numpy as np
import pytest

from pragmata.core.querygen.deduplication import (
    _blueprint_content_key,
    _select_non_duplicate_indices,
    deduplicate_blueprints,
)
from pragmata.core.schemas.querygen_plan import QueryBlueprint


@pytest.fixture()
def make_blueprint() -> callable:
    def _make_blueprint(
        *,
        candidate_id: str = "c001",
        domain: str = "education policy",
        role: str = "policy analyst",
        language: str = "en",
        topic: str = "teacher shortages",
        intent: str = "find evidence",
        task: str = "literature search",
        difficulty: str | None = "medium",
        format: str | None = "bullet list",
        user_scenario: str = "I am preparing a short policy memo.",
        information_need: str = "I need evidence on teacher shortages.",
    ) -> QueryBlueprint:
        return QueryBlueprint.model_validate(
            {
                "candidate_id": candidate_id,
                "domain": domain,
                "role": role,
                "language": language,
                "topic": topic,
                "intent": intent,
                "task": task,
                "difficulty": difficulty,
                "format": format,
                "user_scenario": user_scenario,
                "information_need": information_need,
            }
        )

    return _make_blueprint


def test_blueprint_content_key_is_deterministic_and_ignores_candidate_id(
    make_blueprint: callable,
) -> None:
    blueprint_a = make_blueprint(candidate_id="c001")
    blueprint_b = make_blueprint(candidate_id="c999")

    expected_payload = {
        "domain": "education policy",
        "role": "policy analyst",
        "language": "en",
        "topic": "teacher shortages",
        "intent": "find evidence",
        "task": "literature search",
        "difficulty": "medium",
        "format": "bullet list",
        "user_scenario": "I am preparing a short policy memo.",
        "information_need": "I need evidence on teacher shortages.",
    }
    expected_serialized = json.dumps(
        expected_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    expected_key = hashlib.sha256(expected_serialized.encode("utf-8")).hexdigest()

    assert _blueprint_content_key(blueprint_a) == expected_key
    assert _blueprint_content_key(blueprint_b) == expected_key


@pytest.mark.parametrize(
    ("similarities", "threshold", "expected"),
    [
        (
            np.array(
                [
                    [1.00, 0.96, 0.20],
                    [0.96, 1.00, 0.30],
                    [0.20, 0.30, 1.00],
                ],
                dtype=np.float32,
            ),
            0.95,
            [0, 2],
        ),
        (
            np.array(
                [
                    [1.00, 0.94, 0.96, 0.10],
                    [0.94, 1.00, 0.20, 0.97],
                    [0.96, 0.20, 1.00, 0.98],
                    [0.10, 0.97, 0.98, 1.00],
                ],
                dtype=np.float32,
            ),
            0.95,
            [0, 1],
        ),
        (
            np.eye(3, dtype=np.float32),
            0.95,
            [0, 1, 2],
        ),
    ],
)
def test_select_non_duplicate_indices_returns_first_occurrence_ordered_selection(
    similarities: np.ndarray,
    threshold: float,
    expected: list[int],
) -> None:
    assert _select_non_duplicate_indices(similarities, threshold=threshold) == expected


def test_select_non_duplicate_indices_rejects_non_square_matrices() -> None:
    similarities = np.array([[1.0, 0.9, 0.8]], dtype=np.float32)

    with pytest.raises(ValueError, match="similarities must be a square 2D matrix"):
        _select_non_duplicate_indices(similarities)


def test_deduplicate_blueprints_removes_exact_duplicates_and_preserves_order(
    make_blueprint: callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        make_blueprint(candidate_id="c001", topic="teacher shortages"),
        make_blueprint(candidate_id="c002", topic="teacher shortages"),
        make_blueprint(candidate_id="c003", topic="school meals"),
    ]

    monkeypatch.setattr(
        "pragmata.core.querygen.deduplication._embed_blueprints",
        lambda blueprints: np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    class _FakeModel:
        def similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.eye(len(left), dtype=np.float32)

    monkeypatch.setattr(
        "pragmata.core.querygen.deduplication._load_embedding_model",
        lambda checkpoint="all-MiniLM-L6-v2": _FakeModel(),
    )

    deduplicated = deduplicate_blueprints(candidates)

    assert [blueprint.candidate_id for blueprint in deduplicated] == ["c001", "c003"]


def test_deduplicate_blueprints_applies_near_duplicate_selection_in_original_order(
    make_blueprint: callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        make_blueprint(candidate_id="c001", topic="teacher shortages"),
        make_blueprint(candidate_id="c002", topic="school meals"),
        make_blueprint(candidate_id="c003", topic="digital learning"),
        make_blueprint(candidate_id="c004", topic="adult education"),
    ]

    fake_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        "pragmata.core.querygen.deduplication._embed_blueprints",
        lambda blueprints: fake_embeddings,
    )

    class _FakeModel:
        def similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    [1.00, 0.96, 0.20, 0.10],
                    [0.96, 1.00, 0.30, 0.97],
                    [0.20, 0.30, 1.00, 0.40],
                    [0.10, 0.97, 0.40, 1.00],
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "pragmata.core.querygen.deduplication._load_embedding_model",
        lambda checkpoint="all-MiniLM-L6-v2": _FakeModel(),
    )

    deduplicated = deduplicate_blueprints(candidates)

    assert [blueprint.candidate_id for blueprint in deduplicated] == ["c001", "c003", "c004"]


def test_deduplicate_blueprints_returns_empty_list_for_empty_input() -> None:
    assert deduplicate_blueprints([]) == []