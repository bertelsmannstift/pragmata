"""Format-specific loaders — convert files and objects to list[dict].

Each loader reads a source format and returns a list of raw dicts suitable
for passing to validate_records(). Called internally by import_records()
via resolve_records(); not part of the public API.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset

    RecordInput: TypeAlias = list[dict[str, Any]] | str | Path | pd.DataFrame | Dataset
else:
    RecordInput: TypeAlias = Any

# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file containing an array of records."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file — one JSON object per line."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
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


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV file — supports JSON string chunks column or denormalised rows."""
    with path.open(newline="", encoding="utf-8") as f:
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
            key = f"{row.get('query', '')}\x00{row.get('answer', '')}\x00{row.get('context_set', '')}"
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


def _load_hf_dataset(dataset: Dataset) -> list[dict[str, Any]]:
    """Convert a HuggingFace Dataset to list[dict]."""
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return [dict(row) for row in dataset]


def _load_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to list[dict]."""
    return [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_EXTENSION_LOADERS = {
    ".json": _load_json,
    ".jsonl": _load_jsonl,
    ".csv": _load_csv,
}


def resolve_records(
    records: RecordInput,
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
        path = Path(records).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if format != "auto":
            loader = _EXTENSION_LOADERS.get(f".{format}")
            if loader is None:
                raise ValueError(f"Unsupported format: {format!r}")
            result = loader(path)
            logger.info("Loaded %d records from %s (format=%s)", len(result), path, format)
            return result

        loader = _EXTENSION_LOADERS.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported file extension: {path.suffix!r}")
        result = loader(path)
        logger.info("Loaded %d records from %s", len(result), path)
        return result

    # HF Dataset — check before DataFrame since Dataset may also have to_dict
    try:
        from datasets import Dataset as _Dataset

        if isinstance(records, _Dataset):
            return _load_hf_dataset(records)
    except ImportError:
        pass

    try:
        import pandas as _pd

        if isinstance(records, _pd.DataFrame):
            return _load_dataframe(records)
    except ImportError:
        pass

    raise TypeError(f"Unsupported records type: {type(records)}")
