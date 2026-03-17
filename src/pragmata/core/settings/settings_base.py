"""Shared settings resolution base for runtime settings."""

import os
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict

UNSET = object()

Provider = Literal[
    "mistralai",
    "cohere",
    "deepseek",
    "openai",
    "anthropic",
    "google-genai",
]

PROVIDER_API_KEY_ENV_VARS: Final[dict[Provider, str]] = {
    "mistralai": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
}


class MissingSecretError(RuntimeError):
    """Raised when a required secret is missing from the environment."""


def resolve_provider_api_key(provider: str) -> str:
    """Resolve a supported provider API key from process environment."""
    env_var = PROVIDER_API_KEY_ENV_VARS.get(provider)

    if env_var is None:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = os.environ.get(env_var)

    if api_key is None or not api_key.strip():
        raise MissingSecretError(f"Missing required secret: {env_var}")

    return api_key


def deep_merge(
    base: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge a higher-precedence settings layer into a base dictionary.

    Args:
        base: Lower-precedence settings layer.
        incoming: Higher-precedence settings layer.

    Returns:
        A new dictionary containing the merged settings.
    """
    merged: dict[str, Any] = dict(base)

    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def prune_unset(
    value: Any,
) -> Any:
    """Recursively remove values marked as UNSET from override data."""
    if isinstance(value, dict):
        return {key: prune_unset(item) for key, item in value.items() if item is not UNSET}

    if isinstance(value, list):
        return [prune_unset(item) for item in value if item is not UNSET]

    return value


def load_config_file(
    path: str | Path,
) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    config_path = Path(path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Config file root must be a mapping, got {type(data).__name__}: {config_path}")

    return data


class ResolveSettings(BaseModel):
    """Shared base model for resolving layered run settings."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def resolve(
        cls,
        *,
        config: dict[str, Any] | None = None,
        env: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Resolve settings from defaults, config, env, and overrides.

        Args:
            config: Config-file settings layer.
            env: Environment-derived settings layer.
            overrides: Explicit call-site overrides.

        Returns:
            A validated settings instance.
        """
        resolved: dict[str, Any] = {}
        resolved = deep_merge(resolved, config or {})
        resolved = deep_merge(resolved, env or {})
        resolved = deep_merge(resolved, prune_unset(overrides or {}))

        return cls.model_validate(resolved)
