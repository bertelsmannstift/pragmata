"""CSV serialisation helpers for Pydantic models."""

import csv
import typing
from pathlib import Path

from chatboteval.core.types import M


def _to_csv_value(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _is_optional(annotation: object) -> bool:
    """Return True if annotation is X | None (union containing NoneType)."""
    return type(None) in typing.get_args(annotation)


def write_csv(rows: list[M], path: Path) -> None:
    """Write rows to a CSV file; no-op when rows is empty."""
    if not rows:
        return
    headers = list(type(rows[0]).model_fields.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            raw = row.model_dump(mode="json")
            writer.writerow({k: _to_csv_value(raw[k]) for k in headers})


def read_csv(path: Path, model_cls: type[M]) -> list[M]:
    """Read CSV rows and deserialise into typed Pydantic model instances."""
    optional_fields = {name for name, field in model_cls.model_fields.items() if _is_optional(field.annotation)}
    results: list[M] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            processed = {k: (None if k in optional_fields and v == "" else v) for k, v in row.items()}
            results.append(model_cls.model_validate(processed))
    return results
