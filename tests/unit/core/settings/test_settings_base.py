"""Unit tests for shared runtime settings resolution helpers and base model."""

from pathlib import Path
from typing import Any, NamedTuple

import pytest
from pydantic import BaseModel, ValidationError

from pragmata.core.settings.settings_base import (
    UNSET,
    ResolveSettings,
    deep_merge,
    load_config_file,
    prune_unset,
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
