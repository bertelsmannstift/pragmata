"""Unit tests for shared runtime settings resolution helpers and base model."""

import os
from unittest.mock import patch
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from pydantic import BaseModel, ValidationError

from pragmata.core.settings.settings_base import (
    UNSET,
    MissingSecretError,
    PROVIDER_API_KEY_ENV_VARS,
    ResolveSettings,
    deep_merge,
    load_config_file,
    prune_unset,
    resolve_provider_api_key,
)


class NestedSettings(BaseModel):
    """Nested settings model used to exercise recursive merge and validation."""

    enabled: bool = False
    threshold: float = 0.5


class ExampleSettings(ResolveSettings):
    """Example settings model used to test layered settings resolution."""

    name: str = "default"
    nested: NestedSettings = NestedSettings()
    tags: list[str] = []


class LayerSet(NamedTuple):
    """Container for config, environment, and override resolution layers."""

    config: dict[str, Any]
    env: dict[str, Any]
    overrides: dict[str, Any]


@pytest.fixture
def sample_layers() -> LayerSet:
    """Representative layered settings payload with nested precedence cases."""
    return LayerSet(
        config={
            "name": "from-config",
            "nested": {"enabled": True, "threshold": 0.7},
        },
        env={
            "nested": {"threshold": 0.9},
        },
        overrides={
            "name": "from-overrides",
            "nested": {"enabled": UNSET, "threshold": 1.0},
        },
    )


def test_deep_merge_recursively_replaces_leaf_values_and_preserves_siblings() -> None:
    """deep_merge recursively replaces leaves while preserving sibling keys."""
    base: dict[str, Any] = {
        "llm": {
            "provider": "mistralai",
            "kwargs": {"temperature": 0.2, "top_p": 0.9},
        },
    }
    incoming: dict[str, Any] = {
        "llm": {
            "kwargs": {"temperature": 0.5},
        },
    }

    merged = deep_merge(base, incoming)

    assert merged["llm"]["provider"] == "mistralai"
    assert merged["llm"]["kwargs"] == {"temperature": 0.5, "top_p": 0.9}
    assert base["llm"]["kwargs"]["temperature"] == 0.2


def test_prune_unset_removes_unset_recursively() -> None:
    """prune_unset removes UNSET values recursively from nested structures."""
    value = {
        "a": 1,
        "b": UNSET,
        "c": {"d": UNSET, "e": 2},
        "f": [1, UNSET, {"g": UNSET, "h": 3}],
    }

    assert prune_unset(value) == {
        "a": 1,
        "c": {"e": 2},
        "f": [1, {"h": 3}],
    }


def test_resolve_applies_precedence_correctly(sample_layers: LayerSet) -> None:
    """Resolve applies defaults, config, env, and overrides in precedence order."""
    resolved = ExampleSettings.resolve(
        config=sample_layers.config,
        env=sample_layers.env,
        overrides=sample_layers.overrides,
    )

    assert resolved.name == "from-overrides"
    assert resolved.nested.enabled is True
    assert resolved.nested.threshold == 1.0
    assert resolved.tags == []


def test_resolve_accepts_missing_layers() -> None:
    """Resolve returns the default model when no resolution layers are provided."""
    resolved = ExampleSettings.resolve()

    assert resolved == ExampleSettings()


def test_resolve_rejects_unknown_fields() -> None:
    """Resolve rejects unexpected fields after layered merging and validation."""
    with pytest.raises(ValidationError):
        ExampleSettings.resolve(overrides={"unknown_field": "boom"})


def test_load_config_file_reads_yaml_mapping(tmp_path: Path) -> None:
    """load_config_file parses a YAML mapping into a Python dictionary."""
    path = tmp_path / "config.yml"
    path.write_text(
        "name: constantine\nnested: {enabled: true}",
        encoding="utf-8",
    )

    assert load_config_file(path) == {
        "name": "constantine",
        "nested": {"enabled": True},
    }


def test_load_config_file_returns_empty_dict_for_empty_yaml(tmp_path: Path) -> None:
    """load_config_file returns an empty dict for an empty YAML file."""
    path = tmp_path / "empty.yml"
    path.touch()

    assert load_config_file(path) == {}


def test_load_config_file_raises_for_missing_file(tmp_path: Path) -> None:
    """load_config_file raises FileNotFoundError when the file does not exist."""
    path = tmp_path / "missing.yml"

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        load_config_file(path)


def test_load_config_file_raises_for_non_mapping_root(tmp_path: Path) -> None:
    """load_config_file raises TypeError when the YAML root is not a mapping."""
    path = tmp_path / "invalid.yml"
    path.write_text("- list_not_dict", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        load_config_file(path)


def test_resolve_provider_api_key_reads_supported_provider_secret() -> None:
    """resolve_provider_api_key returns the secret for a supported provider."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-value"}, clear=True):
        assert resolve_provider_api_key("openai") == "secret-value"


def test_resolve_provider_api_key_rejects_unsupported_provider() -> None:
    """resolve_provider_api_key raises for unsupported providers."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            resolve_provider_api_key("azure_openai")

    message = str(exc_info.value)
    assert message.startswith("Unsupported provider: azure_openai.")
    assert "Supported providers:" in message

    for provider in PROVIDER_API_KEY_ENV_VARS:
        assert provider in message


def test_resolve_provider_api_key_raises_when_env_var_is_missing() -> None:
    """resolve_provider_api_key raises when the mapped env var is absent."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(MissingSecretError, match="OPENAI_API_KEY"):
            resolve_provider_api_key("openai")


def test_resolve_provider_api_key_raises_when_env_var_is_empty() -> None:
    """resolve_provider_api_key raises when the mapped env var is empty."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
        with pytest.raises(MissingSecretError, match="OPENAI_API_KEY"):
            resolve_provider_api_key("openai")


def test_provider_api_key_env_vars_maps_supported_querygen_providers() -> None:
    """PROVIDER_API_KEY_ENV_VARS defines the supported provider mappings."""
    assert PROVIDER_API_KEY_ENV_VARS == {
        "mistralai": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google-genai": "GOOGLE_API_KEY",
    }


def test_resolve_provider_api_key_raises_when_env_var_is_whitespace() -> None:
    """resolve_provider_api_key raises when the mapped env var is only whitespace."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "   "}, clear=True):
        with pytest.raises(MissingSecretError, match="OPENAI_API_KEY"):
            resolve_provider_api_key("openai")
