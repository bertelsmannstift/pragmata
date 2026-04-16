"""Unit tests for the Argilla client construction helper."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from pragmata.core.settings.settings_base import MissingSecretError


def test_module_imports_without_argilla_at_module_scope() -> None:
    """Importing the helper module must not trigger an argilla import."""
    sys.modules.pop("pragmata.core.annotation.client", None)
    module = importlib.import_module("pragmata.core.annotation.client")
    assert hasattr(module, "resolve_argilla_client")


def test_resolve_argilla_client_passes_both_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both api_url and api_key are forwarded when provided explicitly."""
    import pragmata.core.annotation.client as client_module

    fake_argilla = MagicMock()
    monkeypatch.setitem(sys.modules, "argilla", fake_argilla)

    client_module.resolve_argilla_client("http://host", "secret")

    fake_argilla.Argilla.assert_called_once_with(api_key="secret", api_url="http://host")


def test_resolve_argilla_client_falls_back_to_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_key=None resolves via ARGILLA_API_KEY env var."""
    import pragmata.core.annotation.client as client_module

    fake_argilla = MagicMock()
    monkeypatch.setitem(sys.modules, "argilla", fake_argilla)
    monkeypatch.setenv("ARGILLA_API_KEY", "env-secret")

    client_module.resolve_argilla_client("http://host", None)

    fake_argilla.Argilla.assert_called_once_with(api_key="env-secret", api_url="http://host")


def test_resolve_argilla_client_raises_when_env_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ARGILLA_API_KEY surfaces as MissingSecretError."""
    import pragmata.core.annotation.client as client_module

    fake_argilla = MagicMock()
    monkeypatch.setitem(sys.modules, "argilla", fake_argilla)
    monkeypatch.delenv("ARGILLA_API_KEY", raising=False)

    with pytest.raises(MissingSecretError, match="ARGILLA_API_KEY"):
        client_module.resolve_argilla_client("http://host", None)

    fake_argilla.Argilla.assert_not_called()


def test_resolve_argilla_client_omits_api_url_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_url=None is omitted so the argilla SDK falls back to its own default."""
    import pragmata.core.annotation.client as client_module

    fake_argilla = MagicMock()
    monkeypatch.setitem(sys.modules, "argilla", fake_argilla)

    client_module.resolve_argilla_client(None, "secret")

    fake_argilla.Argilla.assert_called_once_with(api_key="secret")
