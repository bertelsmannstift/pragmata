"""Round-trip and edge case tests for csv_io helpers."""

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from pragmata.core.csv_io import read_csv, write_csv

_DT = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)


class Colour(StrEnum):
    """Test colour enum."""

    RED = "red"
    GREEN = "green"


class _Row(BaseModel):
    """Test model exercising all csv_io edge cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: Colour
    score: int
    active: bool
    note: str | None = None


@pytest.fixture
def row_with_note() -> _Row:
    """Row with all fields populated."""
    return _Row(id="r1", label=Colour.RED, score=10, active=True, note="hello")


@pytest.fixture
def row_none_note() -> _Row:
    """Row with note=None."""
    return _Row(id="r2", label=Colour.GREEN, score=0, active=False, note=None)


# --- write_csv ---


def test_write_csv_empty_no_file(tmp_path: Path) -> None:
    """Empty list produces no file."""
    out = tmp_path / "out.csv"
    write_csv([], out)
    assert not out.exists()


def test_write_csv_booleans_lowercase(tmp_path: Path, row_with_note: _Row) -> None:
    """Boolean values are written as lowercase true/false."""
    out = tmp_path / "out.csv"
    write_csv([row_with_note], out)
    content = out.read_text()
    assert "true" in content
    assert "True" not in content
    assert "False" not in content


def test_write_csv_none_as_empty_string(tmp_path: Path, row_none_note: _Row) -> None:
    """None values are written as empty strings."""
    import csv

    out = tmp_path / "out.csv"
    write_csv([row_none_note], out)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["note"] == ""


def test_write_csv_enum_as_string(tmp_path: Path, row_with_note: _Row) -> None:
    """StrEnum is written as its string value."""
    import csv

    out = tmp_path / "out.csv"
    write_csv([row_with_note], out)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["label"] == "red"


# --- Round-trips ---


def test_roundtrip_with_note(tmp_path: Path, row_with_note: _Row) -> None:
    """Row with all fields populated survives CSV round-trip."""
    out = tmp_path / "out.csv"
    write_csv([row_with_note], out)
    result = read_csv(out, _Row)
    assert result == [row_with_note]


def test_roundtrip_none_field(tmp_path: Path, row_none_note: _Row) -> None:
    """None optional field survives CSV round-trip."""
    out = tmp_path / "out.csv"
    write_csv([row_none_note], out)
    result = read_csv(out, _Row)
    assert result == [row_none_note]
    assert result[0].note is None


def test_roundtrip_multiple_rows(tmp_path: Path, row_with_note: _Row, row_none_note: _Row) -> None:
    """Multiple rows survive CSV round-trip."""
    out = tmp_path / "out.csv"
    write_csv([row_with_note, row_none_note], out)
    result = read_csv(out, _Row)
    assert result == [row_with_note, row_none_note]


# --- read_csv: extra columns ---


def test_read_csv_extra_column_raises(tmp_path: Path, row_with_note: _Row) -> None:
    """Extra CSV columns cause validation error on read."""
    import csv

    out = tmp_path / "out.csv"
    write_csv([row_with_note], out)
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or []) + ["extra_col"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["extra_col"] = "oops"
            writer.writerow(row)
    with pytest.raises(ValidationError):
        read_csv(out, _Row)


# --- Datetime round-trip ---


class _TimestampRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    created_at: datetime


def test_roundtrip_datetime(tmp_path: Path) -> None:
    """Datetime field survives CSV round-trip."""
    row = _TimestampRow(id="t1", created_at=_DT)
    out = tmp_path / "out.csv"
    write_csv([row], out)
    result = read_csv(out, _TimestampRow)
    assert result == [row]
