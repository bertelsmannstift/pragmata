"""Tests for the shared atomic text-write helper."""

from pathlib import Path

import pytest

from pragmata.core.atomic_io import atomic_write_text


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
