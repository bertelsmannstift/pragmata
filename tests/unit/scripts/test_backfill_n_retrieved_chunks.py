"""Unit tests for the n_retrieved_chunks backfill script.

Focus: the K-join via derive_record_uuid (the bit easy to get wrong) and
the dry-run vs apply behaviour. The Argilla write itself is exercised via
``test_metadata_ops`` and ``test_panel_status``; here we mock the dataset.
"""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pragmata.core.annotation.record_builder import derive_record_uuid
from pragmata.core.schemas.annotation_import import QueryResponsePair

_SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
backfill = importlib.import_module("backfill_n_retrieved_chunks")


def _make_pair(*, chunks: int = 5, query: str = "Q?", answer: str = "A.") -> QueryResponsePair:
    return QueryResponsePair.model_validate(
        {
            "query": query,
            "answer": answer,
            "chunks": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "chunk_rank": i + 1, "text": f"t{i}"} for i in range(chunks)
            ],
            "context_set": "ctx-1",
            "language": "de",
        }
    )


def _make_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "src.jsonl"
    with path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


class TestLoadKMap:
    def test_projects_canonical_fields_and_derives_uuid(self, tmp_path: Path) -> None:
        """Source line has extra keys (query_id, domain, ...) - they must be projected away before validation."""
        pair = _make_pair(chunks=5)
        raw = {
            "query": pair.query,
            "answer": pair.answer,
            "chunks": [c.model_dump() for c in pair.chunks],
            "context_set": pair.context_set,
            "language": pair.language,
            "query_id": "should-be-ignored",
            "domain": "x",
            "topic": "y",
        }
        path = _make_jsonl(tmp_path, [raw])

        k_map = backfill.load_k_map([path])

        expected_uuid = derive_record_uuid(pair)
        assert k_map == {expected_uuid: 5}

    def test_skips_lines_with_empty_chunks(self, tmp_path: Path) -> None:
        """no_retrieval.jsonl entries have chunks=[] - should be skipped, not validation-fail."""
        path = _make_jsonl(
            tmp_path,
            [
                {"query": "q", "answer": "a", "chunks": [], "context_set": "ctx", "language": "de"},
            ],
        )

        k_map = backfill.load_k_map([path])

        assert k_map == {}

    def test_skips_lines_missing_canonical_fields(self, tmp_path: Path) -> None:
        """errors.jsonl entries have different schema (no 'chunks' key at all) - skipped."""
        path = _make_jsonl(
            tmp_path,
            [
                {"error_type": "x", "message": "y", "query_id": "z"},
            ],
        )

        k_map = backfill.load_k_map([path])

        assert k_map == {}

    def test_skips_invalid_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "src.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        k_map = backfill.load_k_map([path])
        assert k_map == {}

    def test_missing_file_is_warned_not_raised(self, tmp_path: Path) -> None:
        k_map = backfill.load_k_map([tmp_path / "absent.jsonl"])
        assert k_map == {}

    def test_dedupes_same_pair_across_files(self, tmp_path: Path) -> None:
        pair = _make_pair(chunks=3)
        raw = {
            "query": pair.query,
            "answer": pair.answer,
            "chunks": [c.model_dump() for c in pair.chunks],
            "context_set": pair.context_set,
            "language": pair.language,
        }
        (tmp_path / "a.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")

        k_map = backfill.load_k_map([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])

        assert len(k_map) == 1
        assert next(iter(k_map.values())) == 3


@pytest.fixture(autouse=True)
def _stub_argilla_property(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argilla MetadataProperty constructors require an authenticated client.

    Patch the IntegerMetadataProperty constructor + ensure_metadata_property
    in the backfill module's namespace so unit tests don't need credentials.
    """
    fake_prop = MagicMock()
    fake_prop.name = "n_retrieved_chunks"
    monkeypatch.setattr(backfill.rg, "IntegerMetadataProperty", MagicMock(return_value=fake_prop))


class TestBackfillDataset:
    def _record(self, *, record_id: str, record_uuid: str, n_retrieved_chunks: int | None = None) -> MagicMock:
        metadata: dict[str, object] = {"record_uuid": record_uuid, "chunk_id": f"chunk-{record_id}"}
        if n_retrieved_chunks is not None:
            metadata["n_retrieved_chunks"] = n_retrieved_chunks
        record = MagicMock()
        record.id = record_id
        record.metadata = metadata
        return record

    def _dataset(self, records: list[MagicMock]) -> MagicMock:
        dataset = MagicMock()
        dataset.name = "retrieval_production"
        dataset.records.side_effect = lambda *a, **kw: iter(records)
        dataset.settings.metadata.__getitem__ = MagicMock(return_value=None)
        return dataset

    def test_dry_run_counts_no_writes(self) -> None:
        records = [
            self._record(record_id="r1", record_uuid="u1"),
            self._record(record_id="r2", record_uuid="u2"),
        ]
        dataset = self._dataset(records)

        stats = backfill.backfill_dataset(dataset, {"u1": 5, "u2": 7}, dry_run=True)

        assert stats.n_updated == 2
        assert stats.n_already_correct == 0
        dataset.records.log.assert_not_called()

    def test_apply_writes_and_declares_property(self) -> None:
        records = [self._record(record_id="r1", record_uuid="u1")]
        dataset = self._dataset(records)

        stats = backfill.backfill_dataset(dataset, {"u1": 5}, dry_run=False)

        assert stats.n_updated == 1
        # Property declared once.
        dataset.settings.add.assert_called_once()
        dataset.settings.update.assert_called_once()
        # Record upserted with full metadata (as an rg.Record, not a dict — dicts
        # get their non-flat keys silently dropped by IngestedRecordMapper).
        dataset.records.log.assert_called_once()
        logged = dataset.records.log.call_args[0][0][0]
        import argilla as rg

        assert isinstance(logged, rg.Record)
        assert logged.id == "r1"
        assert dict(logged.metadata)["n_retrieved_chunks"] == 5
        assert dict(logged.metadata)["record_uuid"] == "u1"

    def test_already_correct_skipped(self) -> None:
        records = [self._record(record_id="r1", record_uuid="u1", n_retrieved_chunks=5)]
        dataset = self._dataset(records)

        stats = backfill.backfill_dataset(dataset, {"u1": 5}, dry_run=False)

        assert stats.n_updated == 0
        assert stats.n_already_correct == 1
        dataset.records.log.assert_not_called()

    def test_no_join_skipped_and_counted(self) -> None:
        records = [self._record(record_id="r1", record_uuid="unknown-uuid")]
        dataset = self._dataset(records)

        stats = backfill.backfill_dataset(dataset, {"u1": 5}, dry_run=False)

        assert stats.n_skipped_no_join == 1
        assert stats.n_updated == 0

    def test_orphan_skipped_and_counted(self) -> None:
        rec = MagicMock()
        rec.id = "rec"
        rec.metadata = {"chunk_id": "c1"}  # no record_uuid
        dataset = self._dataset([rec])

        stats = backfill.backfill_dataset(dataset, {"u1": 5}, dry_run=False)

        assert stats.n_skipped_orphan == 1
        assert stats.n_updated == 0


class TestMainExitCode:
    def test_runs_dry_when_no_apply_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Smoke-test main() in dry-run mode against empty inputs - no client needed if it short-circuits."""
        path = _make_jsonl(tmp_path, [])  # empty file

        # Patch the client + settings resolution to avoid touching real env / config.
        monkeypatch.setattr(
            backfill, "resolve_argilla_client", lambda url, key: MagicMock(datasets=lambda *a, **kw: None)
        )
        monkeypatch.setattr(backfill, "resolve_api_key", lambda _name: "k")
        monkeypatch.setenv("ARGILLA_API_URL", "http://localhost:6900")

        rc = backfill.main([str(path), "--base-dir", str(tmp_path)])
        assert rc == 0
