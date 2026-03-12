"""Unit tests for setup.py — naming helpers, password generation, result dataclass.

No Argilla server required; these tests exercise pure Python logic only.
"""

from dataclasses import fields

from chatboteval.annotation.setup import SetupResult, _apply_prefix, _generate_password


class TestApplyPrefix:
    def test_empty_prefix_returns_name(self) -> None:
        assert _apply_prefix("", "retrieval_grounding") == "retrieval_grounding"

    def test_non_empty_prefix_prepends(self) -> None:
        assert _apply_prefix("pb", "retrieval_grounding") == "pb_retrieval_grounding"

    def test_non_empty_prefix_any_name(self) -> None:
        assert _apply_prefix("test", "generation") == "test_generation"


class TestGeneratePassword:
    def test_default_length(self) -> None:
        pw = _generate_password()
        assert len(pw) == 16

    def test_custom_length(self) -> None:
        pw = _generate_password(length=24)
        assert len(pw) == 24

    def test_returns_string(self) -> None:
        assert isinstance(_generate_password(), str)

    def test_randomness(self) -> None:
        # Two calls should (almost certainly) differ
        assert _generate_password() != _generate_password()


class TestSetupResult:
    def test_default_lists_are_empty(self) -> None:
        result = SetupResult()
        assert result.created_workspaces == []
        assert result.skipped_workspaces == []
        assert result.created_datasets == []
        assert result.skipped_datasets == []
        assert result.created_users == []
        assert result.skipped_users == []

    def test_default_passwords_is_empty_dict(self) -> None:
        result = SetupResult()
        assert result.generated_passwords == {}

    def test_field_names(self) -> None:
        field_names = {f.name for f in fields(SetupResult)}
        assert field_names == {
            "created_workspaces",
            "skipped_workspaces",
            "created_datasets",
            "skipped_datasets",
            "created_users",
            "skipped_users",
            "generated_passwords",
        }

    def test_mutable_defaults_are_independent(self) -> None:
        r1 = SetupResult()
        r2 = SetupResult()
        r1.created_workspaces.append("ws")
        assert r2.created_workspaces == []
