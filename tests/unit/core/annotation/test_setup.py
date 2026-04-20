"""Unit tests for setup.py — naming helpers, password generation, result dataclass.

No Argilla server required; these tests exercise pure Python logic only.
"""

from dataclasses import fields

from pragmata.core.annotation.argilla_ops import apply_suffix, generate_password
from pragmata.core.annotation.setup import SetupResult


class TestApplySuffix:
    def test_empty_suffix_returns_name(self) -> None:
        assert apply_suffix("retrieval", "") == "retrieval"

    def test_non_empty_suffix_appends(self) -> None:
        assert apply_suffix("retrieval", "run1") == "retrieval_run1"

    def test_non_empty_suffix_any_name(self) -> None:
        assert apply_suffix("generation", "batch2") == "generation_batch2"


class TestGeneratePassword:
    def test_default_length(self) -> None:
        pw = generate_password()
        assert len(pw) == 16

    def test_custom_length(self) -> None:
        pw = generate_password(length=24)
        assert len(pw) == 24

    def test_returns_string(self) -> None:
        assert isinstance(generate_password(), str)

    def test_randomness(self) -> None:
        # Two calls should (almost certainly) differ
        assert generate_password() != generate_password()


class TestSetupResult:
    def test_default_lists_are_empty(self) -> None:
        result = SetupResult()
        assert result.created_workspaces == []
        assert result.skipped_workspaces == []
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
            "created_users",
            "skipped_users",
            "generated_passwords",
        }

    def test_mutable_defaults_are_independent(self) -> None:
        r1 = SetupResult()
        r2 = SetupResult()
        r1.created_workspaces.append("ws")
        assert r2.created_workspaces == []
