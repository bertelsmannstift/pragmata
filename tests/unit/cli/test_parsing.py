"""Unit tests for CLI parsing helpers."""

import json
from pathlib import Path

import pytest

from pragmata.annotation import Task, UserSpec
from pragmata.api import UNSET
from pragmata.cli.parsing import parse_cli_value, parse_tasks, parse_user_specs


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
