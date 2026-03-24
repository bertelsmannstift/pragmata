"""Format-specific loaders — convert files and objects to list[dict].

Each loader reads a source format and returns a list of raw dicts suitable
for passing to validate_records(). Called internally by import_records()
via _resolve_records(); not part of the public API.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file containing an array of records."""
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file — one JSON object per line."""
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {lineno}, got {type(obj).__name__}")
            records.append(obj)
    return records


_CHUNK_COLUMNS = {"chunk_text", "chunk_id", "doc_id", "chunk_rank"}


def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV file — supports JSON string chunks column or denormalised rows."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    if "chunks" in rows[0]:
        return _csv_json_column(rows)
    if _CHUNK_COLUMNS <= rows[0].keys():
        return _csv_denormalised(rows)

    raise ValueError(
        "CSV must have either a 'chunks' column (JSON string) or chunk_text/chunk_id/doc_id/chunk_rank columns"
    )


def _csv_json_column(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse CSV where chunks are a JSON array string in a 'chunks' column."""
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        raw_chunks = row.pop("chunks")
        try:
            chunks = json.loads(raw_chunks)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in 'chunks' column, row {i + 1}: {exc}") from exc
        if not isinstance(chunks, list):
            raise ValueError(f"Expected array in 'chunks' column, row {i + 1}")
        row["chunks"] = chunks
        records.append(row)
    return records


def _csv_denormalised(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group denormalised chunk rows into canonical records."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_group_key = "record_id" in rows[0] or "group" in rows[0]
    group_col = "record_id" if "record_id" in rows[0] else "group"

    for row in rows:
        if has_group_key:
            key = row[group_col]
        else:
            key = f"{row.get('query', '')}\x00{row.get('answer', '')}"
        groups[key].append(row)

    records: list[dict[str, Any]] = []
    for chunk_rows in groups.values():
        first = chunk_rows[0]
        record: dict[str, Any] = {}
        # copy non-chunk columns from first row
        for col in first:
            if col not in _CHUNK_COLUMNS and col not in ("record_id", "group"):
                record[col] = first[col]
        # assemble chunks
        record["chunks"] = [
            {
                "text": r["chunk_text"],
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "chunk_rank": _parse_int(r["chunk_rank"]),
            }
            for r in chunk_rows
        ]
        records.append(record)
    return records


def _parse_int(value: str | int) -> int:
    """Parse a string or int to int — handles CSV string values."""
    if isinstance(value, int):
        return value
    return int(value)


# ---------------------------------------------------------------------------
# Object loaders
# ---------------------------------------------------------------------------


def load_hf_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Convert a HuggingFace Dataset to list[dict]."""
    # datasets.Dataset supports iteration and to_list()
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return [dict(row) for row in dataset]


def load_dataframe(df: Any) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to list[dict]."""
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_EXTENSION_LOADERS = {
    ".json": load_json,
    ".jsonl": load_jsonl,
    ".csv": load_csv,
}


def resolve_records(
    records: Any,
    *,
    format: str = "auto",
) -> list[dict[str, Any]]:
    """Resolve input to list[dict] by dispatching on type.

    Args:
        records: One of list[dict], str/Path (file), Dataset, or DataFrame.
        format: Override extension detection for file paths.

    Returns:
        list[dict] ready for validate_records().

    Raises:
        FileNotFoundError: If file path doesn't exist.
        ValueError: If format is unsupported or file content is invalid.
        TypeError: If records type is not recognised.
    """
    if isinstance(records, list):
        return records

    if isinstance(records, (str, Path)):
        path = Path(records)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if format != "auto":
            loader = _EXTENSION_LOADERS.get(f".{format}")
            if loader is None:
                raise ValueError(f"Unsupported format: {format!r}")
            return loader(path)

        loader = _EXTENSION_LOADERS.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported file extension: {path.suffix!r}")
        return loader(path)

    # HF Dataset — check before DataFrame since Dataset may also have to_dict
    type_name = type(records).__name__
    if type_name == "Dataset":
        return load_hf_dataset(records)
    if type_name == "DataFrame":
        return load_dataframe(records)

    raise TypeError(f"Unsupported records type: {type(records)}")
