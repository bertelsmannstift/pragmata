"""Unit tests for CLI parsing helpers."""

import json
from datetime import datetime
from pathlib import Path

import pytest
import typer

from pragmata.annotation import Task, UserSpec
from pragmata.api import UNSET
from pragmata.cli.parsing import (
    parse_annotator_ids,
    parse_cli_value,
    parse_datetime,
    parse_locale,
    parse_tasks,
    parse_user_specs,
)


class TestParseCliValue:
    def test_none_returns_unset(self) -> None:
        assert parse_cli_value(None) is UNSET

    def test_plain_string_passthrough(self) -> None:
        assert parse_cli_value("hello") == "hello"

    def test_json_list_of_strings(self) -> None:
        assert parse_cli_value('["a", "b"]') == ["a", "b"]

    def test_json_object(self) -> None:
        assert parse_cli_value('{"x": 1}') == {"x": 1}


class TestParseTasks:
    def test_none_returns_none(self) -> None:
        assert parse_tasks(None) is None

    def test_single_task(self) -> None:
        assert parse_tasks("retrieval") == [Task.RETRIEVAL]

    def test_multiple_tasks(self) -> None:
        assert parse_tasks("retrieval,grounding,generation") == [
            Task.RETRIEVAL,
            Task.GROUNDING,
            Task.GENERATION,
        ]

    def test_whitespace_tolerant(self) -> None:
        assert parse_tasks("retrieval, grounding") == [Task.RETRIEVAL, Task.GROUNDING]

    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_tasks("nonexistent")


class TestParseLocale:
    def test_none_returns_none(self) -> None:
        assert parse_locale(None) is None

    def test_normalises_string(self) -> None:
        assert parse_locale("en") == "en"

    def test_whitespace_tolerant(self) -> None:
        assert parse_locale("  en  ") == "en"

    def test_unknown_locale_not_rejected_here(self) -> None:
        assert parse_locale("xx") == "xx"


class TestParseUserSpecs:
    def test_none_returns_none(self) -> None:
        assert parse_user_specs(None) is None

    def test_reads_json_list(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "users.json"
        spec_file.write_text(
            json.dumps(
                [
                    {"username": "alice", "role": "annotator", "workspaces": ["retrieval"]},
                    {"username": "bob", "role": "owner"},
                ]
            )
        )

        specs = parse_user_specs(str(spec_file))

        assert specs is not None
        assert len(specs) == 2
        assert specs[0] == UserSpec(username="alice", role="annotator", workspaces=["retrieval"])
        assert specs[1].username == "bob"
        assert specs[1].role == "owner"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_user_specs(str(tmp_path / "missing.json"))

    def test_malformed_entry_raises(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "users.json"
        spec_file.write_text(json.dumps([{"username": "alice"}]))  # missing role

        with pytest.raises(TypeError):
            parse_user_specs(str(spec_file))


class TestParseDatetime:
    def test_parses_full_iso(self) -> None:
        assert parse_datetime("2026-05-01T12:34:56") == datetime(2026, 5, 1, 12, 34, 56)

    def test_parses_date_only(self) -> None:
        assert parse_datetime("2026-05-01") == datetime(2026, 5, 1)

    def test_invalid_raises_typer_bad_parameter(self) -> None:
        with pytest.raises(typer.BadParameter):
            parse_datetime("not-a-date")

    def test_none_returns_none(self) -> None:
        assert parse_datetime(None) is None


class TestParseAnnotatorIds:
    def test_single_id(self) -> None:
        assert parse_annotator_ids("alice") == ["alice"]

    def test_multiple_ids(self) -> None:
        assert parse_annotator_ids("alice,bob,carol") == ["alice", "bob", "carol"]

    def test_strips_whitespace(self) -> None:
        assert parse_annotator_ids(" alice , bob ") == ["alice", "bob"]

    def test_drops_empty_entries(self) -> None:
        assert parse_annotator_ids("alice,,bob,") == ["alice", "bob"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_annotator_ids("") == []

    def test_none_returns_none(self) -> None:
        assert parse_annotator_ids(None) is None
