"""Tests for the shared atomic text-write helper."""

import json
from pathlib import Path

import pytest

from pragmata.core.atomic_io import atomic_write_json, atomic_write_text


def test_atomic_write_text_writes_and_replaces(tmp_path: Path) -> None:
    """Content written in the block becomes the file's contents on clean exit."""
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")

    with atomic_write_text(target) as handle:
        handle.write("new content")

    assert target.read_text(encoding="utf-8") == "new content"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_text_leaves_original_intact_on_exception(tmp_path: Path) -> None:
    """An exception in the block leaves the original file untouched and no temp behind."""
    target = tmp_path / "out.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_text(target) as handle:
            handle.write("partial")
            raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_text_no_target_left_when_block_fails_for_new_path(tmp_path: Path) -> None:
    """A failed write to a not-yet-existing path leaves no file at all."""
    target = tmp_path / "fresh.json"

    with pytest.raises(ValueError, match="nope"):
        with atomic_write_text(target) as handle:
            handle.write("partial")
            raise ValueError("nope")

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_round_trips_indented_with_trailing_newline(tmp_path: Path) -> None:
    """Data is persisted as indented JSON with a trailing newline and reads back equal."""
    target = tmp_path / "meta.json"
    data = {"b": 1, "a": [2, 3]}

    atomic_write_json(data, target)

    text = target.read_text(encoding="utf-8")
    assert text == json.dumps(data, indent=2) + "\n"
    assert json.loads(text) == data
    assert list(tmp_path.glob("*.tmp")) == []
