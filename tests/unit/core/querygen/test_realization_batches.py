"""Tests for the Stage 2 realization-batch checkpoint read helper."""

import importlib.metadata
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pragmata.core.querygen.assembly import assemble_realization_batch_artifact
from pragmata.core.querygen.export import export_realization_batch_artifact
from pragmata.core.querygen.realization_batches import read_realization_batch_artifact
from pragmata.core.schemas.querygen_realize import RealizedQuery


def _realized(candidate_id: str) -> RealizedQuery:
    return RealizedQuery(candidate_id=candidate_id, query=f"query for {candidate_id}")


_EXPECTED = {
    "expected_spec_fingerprint": "fp-1",
    "expected_source_run_id": "run-1",
    "expected_n_queries": 10,
    "expected_batch_size": 5,
    "expected_candidate_ids": ["c001", "c002"],
    "expected_llm_fingerprint": "llm-fp-1",
}


def _write_valid(path: Path) -> None:
    artifact = assemble_realization_batch_artifact(
        spec_fingerprint="fp-1",
        llm_fingerprint="llm-fp-1",
        source_run_id="run-1",
        n_queries=10,
        batch_size=5,
        batch_idx=0,
        candidate_ids=["c001", "c002"],
        queries=[_realized("c001"), _realized("c002")],
    )
    export_realization_batch_artifact(artifact=artifact, path=path)


def test_read_realization_batch_artifact_roundtrip_via_export(tmp_path: Path) -> None:
    """A written artifact reads back with matching content."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    artifact = read_realization_batch_artifact(path=path, **_EXPECTED)

    assert artifact is not None
    assert [q.candidate_id for q in artifact.queries] == ["c001", "c002"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_realization_batch_artifact_returns_none_for_missing_path(tmp_path: Path) -> None:
    """A missing checkpoint reads as None."""
    assert read_realization_batch_artifact(path=tmp_path / "absent.json", **_EXPECTED) is None


@pytest.mark.parametrize(
    ("override_key", "override_value"),
    [
        ("expected_spec_fingerprint", "fp-other"),
        ("expected_source_run_id", "run-other"),
        ("expected_n_queries", 20),
        ("expected_batch_size", 7),
        ("expected_candidate_ids", ["c001", "c999"]),
        ("expected_llm_fingerprint", "llm-other"),
    ],
)
def test_read_realization_batch_artifact_returns_none_for_header_mismatch(
    tmp_path: Path,
    override_key: str,
    override_value: object,
) -> None:
    """A mismatch on any validated header field reads as None (drift)."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    expected = {**_EXPECTED, override_key: override_value}
    assert read_realization_batch_artifact(path=path, **expected) is None


def test_read_realization_batch_artifact_returns_none_for_pragmata_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint written by a different pragmata version reads as None."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "pragmata":
            return "0.0.0-other"
        return real_version(name)

    monkeypatch.setattr("pragmata.core.querygen.checkpoint_read.importlib.metadata.version", fake_version)
    assert read_realization_batch_artifact(path=path, **_EXPECTED) is None


def test_read_realization_batch_artifact_raises_for_malformed_json(tmp_path: Path) -> None:
    """Malformed file content raises a JSON decode error."""
    path = tmp_path / "batch_0000.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_realization_batch_artifact(path=path, **_EXPECTED)


def test_realization_batch_artifact_rejects_candidate_query_length_mismatch() -> None:
    """The schema enforces a 1:1 candidate_ids/queries mapping."""
    with pytest.raises(ValidationError):
        assemble_realization_batch_artifact(
            spec_fingerprint="fp-1",
            llm_fingerprint="llm-fp-1",
            source_run_id="run-1",
            n_queries=10,
            batch_size=5,
            batch_idx=0,
            candidate_ids=["c001", "c002"],
            queries=[_realized("c001")],
        )
