"""Unit tests for the api-layer Argilla credential resolution helpers."""

import pytest

from pragmata.api._argilla_creds import resolve_api_key_override, resolve_api_url_override
from pragmata.core.settings.settings_base import UNSET, MissingSecretError


class TestResolveApiUrlOverride:
    def test_kwarg_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARGILLA_API_URL", "http://env")
        assert resolve_api_url_override("http://kwarg") == "http://kwarg"

    def test_env_used_when_kwarg_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARGILLA_API_URL", "http://env")
        assert resolve_api_url_override(UNSET) == "http://env"

    def test_unset_returned_when_kwarg_and_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARGILLA_API_URL", raising=False)
        assert resolve_api_url_override(UNSET) is UNSET

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_env_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
        monkeypatch.setenv("ARGILLA_API_URL", blank)
        assert resolve_api_url_override(UNSET) is UNSET


class TestResolveApiKeyOverride:
    def test_kwarg_returned_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARGILLA_API_KEY", "env-key")
        assert resolve_api_key_override("kwarg-key") == "kwarg-key"

    def test_env_used_when_kwarg_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARGILLA_API_KEY", "env-key")
        assert resolve_api_key_override(UNSET) == "env-key"

    def test_raises_when_kwarg_unset_and_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARGILLA_API_KEY", raising=False)
        with pytest.raises(MissingSecretError, match="ARGILLA_API_KEY"):
            resolve_api_key_override(UNSET)
