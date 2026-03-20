"""Unit tests for format-specific loaders.

No Argilla server required — tests exercise pure Python loading logic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pragmata.core.annotation.loaders import (
    load_csv,
    load_dataframe,
    load_hf_dataset,
    load_json,
    load_jsonl,
    resolve_records,
)

# ---------------------------------------------------------------------------
# Fixtures — canonical record shape for reuse
# ---------------------------------------------------------------------------

_CHUNKS = [
    {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "text": "Berlin is a city."},
    {"chunk_id": "c2", "doc_id": "d1", "chunk_rank": 2, "text": "It is the capital."},
]

_RECORD = {
    "query": "What is the capital?",
    "answer": "Berlin is the capital.",
    "chunks": _CHUNKS,
    "context_set": "ctx-001",
    "language": "de",
}


def _write_json(path: Path, data: list) -> Path:
    path.write_text(json.dumps(data))
    return path


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_loads_array(self, tmp_path: Path) -> None:
        f = _write_json(tmp_path / "data.json", [_RECORD])
        result = load_json(f)
        assert len(result) == 1
        assert result[0]["query"] == "What is the capital?"

    def test_multiple_records(self, tmp_path: Path) -> None:
        f = _write_json(tmp_path / "data.json", [_RECORD, _RECORD])
        assert len(load_json(f)) == 2

    def test_empty_array(self, tmp_path: Path) -> None:
        f = _write_json(tmp_path / "data.json", [])
        assert load_json(f) == []

    def test_rejects_non_array(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"not": "an array"}))
        with pytest.raises(ValueError, match="Expected JSON array"):
            load_json(f)


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_lines(self, tmp_path: Path) -> None:
        f = _write_jsonl(tmp_path / "data.jsonl", [_RECORD])
        result = load_jsonl(f)
        assert len(result) == 1
        assert result[0]["query"] == "What is the capital?"

    def test_multiple_lines(self, tmp_path: Path) -> None:
        f = _write_jsonl(tmp_path / "data.jsonl", [_RECORD, _RECORD])
        assert len(load_jsonl(f)) == 2

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(_RECORD) + "\n\n" + json.dumps(_RECORD) + "\n")
        assert len(load_jsonl(f)) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("")
        assert load_jsonl(f) == []

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON on line 1"):
            load_jsonl(f)

    def test_rejects_non_object_line(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("[1,2,3]\n")
        with pytest.raises(ValueError, match="Expected JSON object on line 1"):
            load_jsonl(f)


# ---------------------------------------------------------------------------
# load_csv — JSON string column
# ---------------------------------------------------------------------------


class TestLoadCsvJsonColumn:
    def _write_csv(self, path: Path, records: list[dict]) -> Path:
        import csv as _csv

        fieldnames = list(records[0].keys())
        with path.open("w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_loads_json_chunks_column(self, tmp_path: Path) -> None:
        row = {
            "query": "What is X?",
            "answer": "X is Y.",
            "chunks": json.dumps(_CHUNKS),
            "context_set": "ctx-001",
        }
        f = self._write_csv(tmp_path / "data.csv", [row])
        result = load_csv(f)
        assert len(result) == 1
        assert isinstance(result[0]["chunks"], list)
        assert result[0]["chunks"][0]["chunk_id"] == "c1"

    def test_multiple_rows(self, tmp_path: Path) -> None:
        row = {
            "query": "Q",
            "answer": "A",
            "chunks": json.dumps(_CHUNKS),
            "context_set": "ctx",
        }
        f = self._write_csv(tmp_path / "data.csv", [row, row])
        assert len(load_csv(f)) == 2

    def test_empty_csv(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("query,answer,chunks\n")
        assert load_csv(f) == []

    def test_invalid_json_in_chunks(self, tmp_path: Path) -> None:
        row = {"query": "Q", "answer": "A", "chunks": "not json", "context_set": "ctx"}
        f = self._write_csv(tmp_path / "data.csv", [row])
        with pytest.raises(ValueError, match="Invalid JSON in 'chunks' column"):
            load_csv(f)


# ---------------------------------------------------------------------------
# load_csv — denormalised rows
# ---------------------------------------------------------------------------


class TestLoadCsvDenormalised:
    def _write_csv(self, path: Path, rows: list[dict]) -> Path:
        import csv as _csv

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_groups_by_record_id(self, tmp_path: Path) -> None:
        rows = [
            {
                "record_id": "1",
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "chunk a",
                "chunk_id": "c1",
                "doc_id": "d1",
                "chunk_rank": "1",
            },
            {
                "record_id": "1",
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "chunk b",
                "chunk_id": "c2",
                "doc_id": "d1",
                "chunk_rank": "2",
            },
            {
                "record_id": "2",
                "query": "Q2",
                "answer": "A2",
                "context_set": "ctx2",
                "chunk_text": "chunk c",
                "chunk_id": "c3",
                "doc_id": "d2",
                "chunk_rank": "1",
            },
        ]
        f = self._write_csv(tmp_path / "data.csv", rows)
        result = load_csv(f)
        assert len(result) == 2
        assert len(result[0]["chunks"]) == 2
        assert len(result[1]["chunks"]) == 1

    def test_groups_by_query_answer_fallback(self, tmp_path: Path) -> None:
        rows = [
            {
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "chunk a",
                "chunk_id": "c1",
                "doc_id": "d1",
                "chunk_rank": "1",
            },
            {
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "chunk b",
                "chunk_id": "c2",
                "doc_id": "d1",
                "chunk_rank": "2",
            },
        ]
        f = self._write_csv(tmp_path / "data.csv", rows)
        result = load_csv(f)
        assert len(result) == 1
        assert len(result[0]["chunks"]) == 2

    def test_chunk_rank_parsed_as_int(self, tmp_path: Path) -> None:
        rows = [
            {
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "text",
                "chunk_id": "c1",
                "doc_id": "d1",
                "chunk_rank": "3",
            },
        ]
        f = self._write_csv(tmp_path / "data.csv", rows)
        result = load_csv(f)
        assert result[0]["chunks"][0]["chunk_rank"] == 3

    def test_group_column_excluded_from_record(self, tmp_path: Path) -> None:
        rows = [
            {
                "record_id": "1",
                "query": "Q",
                "answer": "A",
                "context_set": "ctx",
                "chunk_text": "text",
                "chunk_id": "c1",
                "doc_id": "d1",
                "chunk_rank": "1",
            },
        ]
        f = self._write_csv(tmp_path / "data.csv", rows)
        result = load_csv(f)
        assert "record_id" not in result[0]

    def test_rejects_csv_without_chunks_or_chunk_columns(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("query,answer\nQ,A\n")
        with pytest.raises(ValueError, match="CSV must have either"):
            load_csv(f)


# ---------------------------------------------------------------------------
# load_hf_dataset
# ---------------------------------------------------------------------------


class TestLoadHfDataset:
    def test_uses_to_list(self) -> None:
        mock_ds = MagicMock()
        mock_ds.to_list.return_value = [_RECORD]
        result = load_hf_dataset(mock_ds)
        assert result == [_RECORD]
        mock_ds.to_list.assert_called_once()

    def test_falls_back_to_iteration(self) -> None:
        class IterableDataset:
            def __iter__(self):
                return iter([_RECORD])

        result = load_hf_dataset(IterableDataset())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------


class TestLoadDataframe:
    def test_calls_to_dict_records(self) -> None:
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [_RECORD]
        result = load_dataframe(mock_df)
        assert result == [_RECORD]
        mock_df.to_dict.assert_called_once_with("records")


# ---------------------------------------------------------------------------
# resolve_records — dispatch
# ---------------------------------------------------------------------------


class TestResolveRecords:
    def test_list_passthrough(self) -> None:
        records = [_RECORD]
        assert resolve_records(records) is records

    def test_json_file(self, tmp_path: Path) -> None:
        f = _write_json(tmp_path / "data.json", [_RECORD])
        result = resolve_records(str(f))
        assert len(result) == 1

    def test_jsonl_file(self, tmp_path: Path) -> None:
        f = _write_jsonl(tmp_path / "data.jsonl", [_RECORD])
        result = resolve_records(str(f))
        assert len(result) == 1

    def test_path_object(self, tmp_path: Path) -> None:
        f = _write_json(tmp_path / "data.json", [_RECORD])
        result = resolve_records(f)
        assert len(result) == 1

    def test_format_override(self, tmp_path: Path) -> None:
        # write json content but with .txt extension
        f = tmp_path / "data.txt"
        f.write_text(json.dumps([_RECORD]))
        result = resolve_records(str(f), format="json")
        assert len(result) == 1

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "data.parquet"
        f.write_text("")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            resolve_records(str(f))

    def test_unsupported_format_kwarg(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text("[]")
        with pytest.raises(ValueError, match="Unsupported format"):
            resolve_records(str(f), format="parquet")

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_records("/nonexistent/data.json")

    def test_hf_dataset(self) -> None:
        mock_ds = MagicMock()
        mock_ds.to_list.return_value = [_RECORD]
        type(mock_ds).__name__ = "Dataset"
        result = resolve_records(mock_ds)
        assert result == [_RECORD]

    def test_dataframe(self) -> None:
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [_RECORD]
        type(mock_df).__name__ = "DataFrame"
        result = resolve_records(mock_df)
        assert result == [_RECORD]

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported records type"):
            resolve_records(42)
