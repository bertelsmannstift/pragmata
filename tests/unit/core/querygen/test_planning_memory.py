"""Tests for the synthetic query-generation stage-1 planning memory helpers."""

import hashlib
import json
from collections.abc import Callable

import pytest

from pragmata.core.querygen.planning_memory import (
    _serialize_spec_content,
    fingerprint_querygen_spec,
)
from pragmata.core.schemas.querygen_input import QueryGenSpec


@pytest.fixture()
def make_spec() -> Callable[..., QueryGenSpec]:
    def _make_spec(
        *,
        domains: object = "education policy",
        roles: object = "policy analyst",
        languages: object = "en",
        topics: object = "teacher shortages",
        intents: object = "find evidence",
        tasks: object = "literature search",
        difficulty: object = "medium",
        formats: object = "bullet list",
        disallowed_topics: list[str] | None = None,
    ) -> QueryGenSpec:
        return QueryGenSpec.model_validate(
            {
                "domain_context": {
                    "domains": domains,
                    "roles": roles,
                    "languages": languages,
                },
                "knowledge_scope": {
                    "topics": topics,
                },
                "scenario": {
                    "intents": intents,
                    "tasks": tasks,
                    "difficulty": difficulty,
                },
                "format_requests": {
                    "formats": formats,
                },
                "safety": {
                    "disallowed_topics": disallowed_topics,
                },
            }
        )

    return _make_spec


@pytest.fixture()
def expected_default_payload() -> dict[str, object]:
    return {
        "domain_context": {
            "domains": [{"value": "education policy", "weight": 1.0}],
            "roles": [{"value": "policy analyst", "weight": 1.0}],
            "languages": [{"value": "en", "weight": 1.0}],
        },
        "knowledge_scope": {
            "topics": [{"value": "teacher shortages", "weight": 1.0}],
        },
        "scenario": {
            "intents": [{"value": "find evidence", "weight": 1.0}],
            "tasks": [{"value": "literature search", "weight": 1.0}],
            "difficulty": [{"value": "medium", "weight": 1.0}],
        },
        "format_requests": {
            "formats": [{"value": "bullet list", "weight": 1.0}],
        },
        "safety": {
            "disallowed_topics": None,
        },
    }


def test_serialize_spec_content_returns_expected_canonical_json(
    make_spec: Callable[..., QueryGenSpec],
    expected_default_payload: dict[str, object],
) -> None:
    spec = make_spec()

    expected_serialized = json.dumps(
        expected_default_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert _serialize_spec_content(spec) == expected_serialized


def test_serialize_spec_content_round_trips_to_expected_payload(
    make_spec: Callable[..., QueryGenSpec],
    expected_default_payload: dict[str, object],
) -> None:
    spec = make_spec()

    serialized = _serialize_spec_content(spec)

    assert isinstance(serialized, str)
    assert json.loads(serialized) == expected_default_payload


def test_fingerprint_querygen_spec_is_stable_across_repeated_calls(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec(disallowed_topics=["medical advice"])

    fingerprint = fingerprint_querygen_spec(spec)

    assert fingerprint == fingerprint_querygen_spec(spec)
    assert fingerprint == fingerprint_querygen_spec(spec)
    assert len(fingerprint) == 64
    assert all(char in "0123456789abcdef" for char in fingerprint)


@pytest.mark.parametrize(
    ("field_name", "base_value", "changed_value"),
    [
        ("domains", "education policy", "health policy"),
        ("roles", "policy analyst", "school principal"),
        ("languages", "en", "de"),
        ("topics", "teacher shortages", "school meals"),
        ("intents", "find evidence", "compare options"),
        ("tasks", "literature search", "summarization"),
        ("difficulty", "medium", "hard"),
        ("formats", "bullet list", "table"),
        ("disallowed_topics", ["medical advice"], ["legal advice"]),
    ],
)
def test_fingerprint_querygen_spec_changes_when_any_field_value_changes(
    make_spec: Callable[..., QueryGenSpec],
    field_name: str,
    base_value: object,
    changed_value: object,
) -> None:
    kwargs_base = {field_name: base_value}
    kwargs_changed = {field_name: changed_value}

    spec_a = make_spec(**kwargs_base)
    spec_b = make_spec(**kwargs_changed)

    assert fingerprint_querygen_spec(spec_a) != fingerprint_querygen_spec(spec_b)


def test_fingerprint_querygen_spec_matches_for_equivalent_canonicalized_inputs() -> None:
    scalar_spec = QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": "education policy",
                "roles": "policy analyst",
                "languages": "en",
            },
            "knowledge_scope": {
                "topics": "teacher shortages",
            },
            "scenario": {
                "intents": "find evidence",
                "tasks": "literature search",
                "difficulty": "medium",
            },
            "format_requests": {
                "formats": "bullet list",
            },
            "safety": {
                "disallowed_topics": ["medical advice"],
            },
        }
    )

    weighted_spec = QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": [{"value": "education policy", "weight": 1.0}],
                "roles": [{"value": "policy analyst", "weight": 1.0}],
                "languages": [{"value": "en", "weight": 1.0}],
            },
            "knowledge_scope": {
                "topics": [{"value": "teacher shortages", "weight": 1.0}],
            },
            "scenario": {
                "intents": [{"value": "find evidence", "weight": 1.0}],
                "tasks": [{"value": "literature search", "weight": 1.0}],
                "difficulty": [{"value": "medium", "weight": 1.0}],
            },
            "format_requests": {
                "formats": [{"value": "bullet list", "weight": 1.0}],
            },
            "safety": {
                "disallowed_topics": ["medical advice"],
            },
        }
    )

    assert scalar_spec == weighted_spec
    assert fingerprint_querygen_spec(scalar_spec) == fingerprint_querygen_spec(weighted_spec)


def test_fingerprint_querygen_spec_matches_sha256_of_serialized_content(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec(disallowed_topics=["medical advice"])

    serialized = _serialize_spec_content(spec)
    expected_fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    assert fingerprint_querygen_spec(spec) == expected_fingerprint
