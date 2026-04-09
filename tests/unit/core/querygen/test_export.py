"""Unit tests for synthetic query export."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from pragmata.core.querygen.export import export_queries
from pragmata.core.schemas.querygen_output import SyntheticQueriesMeta, SyntheticQueryRow


def _make_row(
    *,
    query_id: str = "run123_q1",
    query: str = "How do I apply for housing support?",
    domain: str | None = "public policy",
    role: str | None = "citizen",
    language: str | None = "en",
    topic: str | None = "housing",
    intent: str | None = "information",
    task: str | None = "eligibility",
    difficulty: str | None = "easy",
    format: str | None = "steps",
) -> SyntheticQueryRow:
    """Build a valid SyntheticQueryRow for tests."""
    return SyntheticQueryRow(
        query_id=query_id,
        query=query,
        domain=domain,
        role=role,
        language=language,
        topic=topic,
        intent=intent,
        task=task,
        difficulty=difficulty,
        format=format,
    )


def _make_meta(
    *,
    run_id: str = "run123",
    created_at: datetime = datetime(2026, 3, 9, 10, 30, tzinfo=UTC),
    n_requested_queries: int = 5,
    n_returned_queries: int = 3,
    model_provider: str = "mistralai",
    planning_model: str = "magistral-medium-latest",
    realization_model: str = "mistral-medium-latest",
) -> SyntheticQueriesMeta:
    """Build a valid SyntheticQueriesMeta for tests."""
    return SyntheticQueriesMeta(
        run_id=run_id,
        created_at=created_at,
        n_requested_queries=n_requested_queries,
        n_returned_queries=n_returned_queries,
        model_provider=model_provider,
        planning_model=planning_model,
        realization_model=realization_model,
    )


def test_export_queries_writes_rows_to_csv(tmp_path: Path) -> None:
    """export_queries should write assembled synthetic query rows to CSV."""
    rows = [
        _make_row(),
        _make_row(
            query_id="run123_q2",
            query="What documents do I need?",
            topic="documentation",
        ),
    ]
    meta = _make_meta(n_returned_queries=2)
    queries_path = tmp_path / "synthetic_queries.csv"
    meta_path = tmp_path / "synthetic_queries.meta.json"

    export_queries(rows=rows, meta=meta, queries_path=queries_path, meta_path=meta_path)

    with queries_path.open(newline="", encoding="utf-8") as f:
        written_rows = list(csv.DictReader(f))

    assert written_rows == [
        {
            "query_id": "run123_q1",
            "query": "How do I apply for housing support?",
            "domain": "public policy",
            "role": "citizen",
            "language": "en",
            "topic": "housing",
            "intent": "information",
            "task": "eligibility",
            "difficulty": "easy",
            "format": "steps",
        },
        {
            "query_id": "run123_q2",
            "query": "What documents do I need?",
            "domain": "public policy",
            "role": "citizen",
            "language": "en",
            "topic": "documentation",
            "intent": "information",
            "task": "eligibility",
            "difficulty": "easy",
            "format": "steps",
        },
    ]


def test_export_queries_writes_meta_to_json(tmp_path: Path) -> None:
    """export_queries should write dataset-level metadata to JSON."""
    rows = [_make_row()]
    meta = _make_meta(n_returned_queries=1)
    queries_path = tmp_path / "synthetic_queries.csv"
    meta_path = tmp_path / "synthetic_queries.meta.json"

    export_queries(rows=rows, meta=meta, queries_path=queries_path, meta_path=meta_path)

    assert json.loads(meta_path.read_text(encoding="utf-8")) == meta.model_dump(mode="json")


def test_export_queries_reuses_write_csv_helper(monkeypatch, tmp_path: Path) -> None:
    """export_queries should delegate CSV writing to the shared write_csv helper."""
    rows = [_make_row()]
    meta = _make_meta(n_returned_queries=1)
    queries_path = tmp_path / "synthetic_queries.csv"
    meta_path = tmp_path / "synthetic_queries.meta.json"
    called: dict[str, object] = {}

    def _fake_write_csv(export_rows: list[SyntheticQueryRow], export_path: Path) -> None:
        called["rows"] = export_rows
        called["path"] = export_path

    monkeypatch.setattr("pragmata.core.querygen.export.write_csv", _fake_write_csv)

    export_queries(rows=rows, meta=meta, queries_path=queries_path, meta_path=meta_path)

    assert called == {
        "rows": rows,
        "path": queries_path,
    }


def test_export_queries_serializes_empty_optional_row_fields(tmp_path: Path) -> None:
    """export_queries should write optional empty row fields as empty CSV cells."""
    rows = [
        _make_row(
            role=None,
            language=None,
            difficulty=None,
            format=None,
        )
    ]
    meta = _make_meta(n_returned_queries=1)
    queries_path = tmp_path / "synthetic_queries.csv"
    meta_path = tmp_path / "synthetic_queries.meta.json"

    export_queries(rows=rows, meta=meta, queries_path=queries_path, meta_path=meta_path)

    with queries_path.open(newline="", encoding="utf-8") as f:
        written_rows = list(csv.DictReader(f))

    assert written_rows == [
        {
            "query_id": "run123_q1",
            "query": "How do I apply for housing support?",
            "domain": "public policy",
            "role": "",
            "language": "",
            "topic": "housing",
            "intent": "information",
            "task": "eligibility",
            "difficulty": "",
            "format": "",
        }
    ]


def test_export_queries_serializes_metadata_json_values(tmp_path: Path) -> None:
    """export_queries should serialize metadata using JSON-compatible model_dump output."""
    rows = [_make_row()]
    meta = _make_meta(n_requested_queries=1, n_returned_queries=0)
    queries_path = tmp_path / "synthetic_queries.csv"
    meta_path = tmp_path / "synthetic_queries.meta.json"

    export_queries(rows=rows, meta=meta, queries_path=queries_path, meta_path=meta_path)

    assert json.loads(meta_path.read_text(encoding="utf-8")) == {
        "run_id": "run123",
        "created_at": "2026-03-09T10:30:00Z",
        "n_requested_queries": 1,
        "n_returned_queries": 0,
        "model_provider": "mistralai",
        "planning_model": "magistral-medium-latest",
        "realization_model": "mistral-medium-latest",
    }