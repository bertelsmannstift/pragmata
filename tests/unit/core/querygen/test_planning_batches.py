"""Tests for the Stage 1 planning-batch checkpoint read helper."""

import importlib.metadata
from pathlib import Path

import pytest
from pydantic import ValidationError

from pragmata.core.querygen.assembly import assemble_planning_batch_artifact
from pragmata.core.querygen.export import export_planning_batch_artifact
from pragmata.core.querygen.planning_batches import read_planning_batch_artifact
from pragmata.core.schemas.querygen_plan import QueryBlueprint


def _blueprint(candidate_id: str) -> QueryBlueprint:
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain="d",
        role="r",
        language="german",
        topic="t",
        intent="i",
        task="k",
        user_scenario="scenario",
        information_need="need",
    )


_EXPECTED = {
    "expected_spec_fingerprint": "fp-1",
    "expected_source_run_id": "run-1",
    "expected_n_queries": 10,
    "expected_batch_size": 5,
    "expected_candidate_ids": ["c001", "c002"],
    "expected_enable_planning_memory": True,
}


def _write_valid(path: Path) -> None:
    artifact = assemble_planning_batch_artifact(
        spec_fingerprint="fp-1",
        source_run_id="run-1",
        n_queries=10,
        batch_size=5,
        batch_idx=0,
        candidate_ids=["c001", "c002"],
        blueprints=[_blueprint("c001"), _blueprint("c002")],
        planning_summary_state=None,
        enable_planning_memory=True,
    )
    export_planning_batch_artifact(artifact=artifact, path=path)


def test_read_planning_batch_artifact_roundtrip_via_export(tmp_path: Path) -> None:
    """A written artifact reads back with matching content."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    artifact = read_planning_batch_artifact(path=path, **_EXPECTED)

    assert artifact is not None
    assert [bp.candidate_id for bp in artifact.blueprints] == ["c001", "c002"]
    assert artifact.batch_idx == 0


def test_export_planning_batch_artifact_leaves_no_tmp_on_success(tmp_path: Path) -> None:
    """No uniquified tempfile remains after a successful atomic write."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_planning_batch_artifact_returns_none_for_missing_path(tmp_path: Path) -> None:
    """A missing checkpoint reads as None (not yet written)."""
    assert read_planning_batch_artifact(path=tmp_path / "absent.json", **_EXPECTED) is None


@pytest.mark.parametrize(
    ("override_key", "override_value"),
    [
        ("expected_spec_fingerprint", "fp-other"),
        ("expected_source_run_id", "run-other"),
        ("expected_n_queries", 20),
        ("expected_batch_size", 7),
        ("expected_candidate_ids", ["c001", "c999"]),
        ("expected_enable_planning_memory", False),
    ],
)
def test_read_planning_batch_artifact_returns_none_for_header_mismatch(
    tmp_path: Path,
    override_key: str,
    override_value: object,
) -> None:
    """A mismatch on any validated header field reads as None (drift)."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)

    expected = {**_EXPECTED, override_key: override_value}
    assert read_planning_batch_artifact(path=path, **expected) is None


def test_read_planning_batch_artifact_returns_none_for_pragmata_version_mismatch(
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

    monkeypatch.setattr("pragmata.core.querygen.planning_batches.importlib.metadata.version", fake_version)
    assert read_planning_batch_artifact(path=path, **_EXPECTED) is None


def test_read_planning_batch_artifact_raises_for_extra_field(tmp_path: Path) -> None:
    """An unknown field fails schema validation (extra=forbid)."""
    path = tmp_path / "batch_0000.json"
    _write_valid(path)
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload.replace("{", '{"unexpected": 1,', 1), encoding="utf-8")

    with pytest.raises(ValidationError):
        read_planning_batch_artifact(path=path, **_EXPECTED)


def test_read_planning_batch_artifact_raises_for_malformed_json(tmp_path: Path) -> None:
    """Malformed file content raises a JSON decode error."""
    import json

    path = tmp_path / "batch_0000.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_planning_batch_artifact(path=path, **_EXPECTED)


def test_planning_batch_artifact_rejects_candidate_blueprint_length_mismatch() -> None:
    """The schema enforces a 1:1 candidate_ids/blueprints mapping."""
    with pytest.raises(ValidationError):
        assemble_planning_batch_artifact(
            spec_fingerprint="fp-1",
            source_run_id="run-1",
            n_queries=10,
            batch_size=5,
            batch_idx=0,
            candidate_ids=["c001", "c002"],
            blueprints=[_blueprint("c001")],
            planning_summary_state=None,
            enable_planning_memory=True,
        )
