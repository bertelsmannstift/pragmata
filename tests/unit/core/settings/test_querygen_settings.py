"""Unit tests for synthetic query generation run settings."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pragmata.core.settings.querygen_settings import LlmSettings, QueryGenRunSettings


def _valid_spec_payload() -> dict[str, Any]:
    """Return a minimal valid QueryGenSpec payload."""
    return {
        "domain_context": {
            "domains": "public services",
            "roles": "resident",
            "languages": "en",
        },
        "knowledge_scope": {
            "topics": "housing support",
        },
        "scenario": {
            "intents": "information lookup",
            "tasks": "find eligibility requirements",
        },
    }


def test_llm_settings_defaults() -> None:
    """LlmSettings exposes the expected default values."""
    settings = LlmSettings()

    assert settings.model_provider == "mistralai"
    assert settings.planning_model == "magistral-medium-latest"
    assert settings.realization_model == "mistral-medium-latest"
    assert settings.base_url is None
    assert settings.model_kwargs == {}
    assert settings.requests_per_second == 1.0
    assert settings.check_every_n_seconds == 1.0
    assert settings.max_bucket_size == 1


def test_querygen_run_settings_construction_with_defaults() -> None:
    """QueryGenRunSettings applies run-level defaults when spec is provided."""
    settings = QueryGenRunSettings.model_validate({"spec": _valid_spec_payload()})

    assert settings.llm.model_provider == "mistralai"
    assert settings.llm.planning_model == "magistral-medium-latest"
    assert settings.llm.realization_model == "mistral-medium-latest"
    assert settings.llm.model_kwargs == {}
    assert isinstance(settings.base_dir, Path)
    assert settings.run_id
    assert settings.n_queries == 50


def test_querygen_run_settings_resolve_deep_merges_nested_llm_settings() -> None:
    """Resolve deep-merges nested settings layers for llm configuration."""
    resolved = QueryGenRunSettings.resolve(
        config={
            "spec": _valid_spec_payload(),
            "llm": {
                "model_provider": "mistralai",
                "model_kwargs": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            },
            "n_queries": 20,
        },
        overrides={
            "llm": {
                "planning_model": "magistral-small-latest",
                "model_kwargs": {
                    "temperature": 0.7,
                },
            }
        },
    )

    assert resolved.n_queries == 20
    assert resolved.llm.model_provider == "mistralai"
    assert resolved.llm.planning_model == "magistral-small-latest"
    assert resolved.llm.realization_model == "mistral-medium-latest"
    assert resolved.llm.model_kwargs == {
        "temperature": 0.7,
        "top_p": 0.9,
    }


def test_querygen_run_settings_model_kwargs_merge_semantics() -> None:
    """Resolve preserves and overrides individual model_kwargs entries across layers."""
    resolved = QueryGenRunSettings.resolve(
        config={
            "spec": _valid_spec_payload(),
            "llm": {
                "model_kwargs": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                }
            },
        },
        env={
            "llm": {
                "model_kwargs": {
                    "max_tokens": 300,
                }
            },
        },
        overrides={
            "llm": {
                "model_kwargs": {
                    "temperature": 0.7,
                }
            }
        },
    )

    assert resolved.llm.model_kwargs == {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 300,
    }


def test_llm_settings_rejects_invalid_rate_limiter_values() -> None:
    """LlmSettings rejects invalid rate limiter configuration values."""
    invalid_payloads = [
        {"requests_per_second": 0},
        {"requests_per_second": -1.0},
        {"check_every_n_seconds": 0},
        {"check_every_n_seconds": -0.5},
        {"max_bucket_size": 0},
        {"max_bucket_size": -1},
    ]

    for payload in invalid_payloads:
        try:
            LlmSettings.model_validate(payload)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Expected ValidationError for payload: {payload}")
