"""Tests for the frozen Stage 1 result read helper."""

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pragmata.core.querygen.assembly import assemble_selected_blueprints_artifact
from pragmata.core.querygen.export import export_selected_blueprints
from pragmata.core.querygen.selected_blueprints import read_selected_blueprints_artifact
from pragmata.core.schemas.querygen_output import SelectedBlueprintsArtifact
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
    "expected_near_duplicate_tolerance": 0.95,
    "expected_enable_planning_memory": True,
    "expected_llm_fingerprint": "llm-fp-1",
}


def _write_valid(path: Path) -> None:
    artifact = assemble_selected_blueprints_artifact(
        spec_fingerprint="fp-1",
        llm_fingerprint="llm-fp-1",
        source_run_id="run-1",
        n_queries=10,
        batch_size=5,
        near_duplicate_tolerance=0.95,
        enable_planning_memory=True,
        embedding_model="all-MiniLM-L6-v2",
        blueprints=[_blueprint("c001"), _blueprint("c003")],
    )
    export_selected_blueprints(artifact=artifact, path=path)


def test_read_selected_blueprints_roundtrip_via_export(tmp_path: Path) -> None:
    """A written frozen result reads back with matching content and no drift."""
    path = tmp_path / "selected_blueprints.json"
    _write_valid(path)

    artifact, drifted = read_selected_blueprints_artifact(path=path, **_EXPECTED)

    assert drifted == []
    assert artifact is not None
    assert [bp.candidate_id for bp in artifact.blueprints] == ["c001", "c003"]
    assert artifact.embedding_model == "all-MiniLM-L6-v2"


def test_read_selected_blueprints_returns_none_for_missing_path(tmp_path: Path) -> None:
    """A missing frozen result reads as ``(None, [])`` -- nothing to resume, no drift."""
    artifact, drifted = read_selected_blueprints_artifact(path=tmp_path / "absent.json", **_EXPECTED)

    assert artifact is None
    assert drifted == []


@pytest.mark.parametrize(
    ("override_key", "override_value"),
    [
        ("expected_spec_fingerprint", "fp-other"),
        ("expected_source_run_id", "run-other"),
        ("expected_n_queries", 20),
        ("expected_batch_size", 7),
        ("expected_near_duplicate_tolerance", 0.80),
        ("expected_enable_planning_memory", False),
        ("expected_llm_fingerprint", "llm-other"),
    ],
)
def test_read_selected_blueprints_reports_drift_for_header_mismatch(
    tmp_path: Path,
    override_key: str,
    override_value: object,
) -> None:
    """A mismatch on any validated header field is reported as drift, not reused."""
    path = tmp_path / "selected_blueprints.json"
    _write_valid(path)

    expected = {**_EXPECTED, override_key: override_value}
    artifact, drifted = read_selected_blueprints_artifact(path=path, **expected)

    assert artifact is None
    assert override_key.removeprefix("expected_") in {field.field for field in drifted}


def test_read_selected_blueprints_ignores_embedding_model_for_validation(tmp_path: Path) -> None:
    """embedding_model is recorded for provenance but not used to detect drift."""
    path = tmp_path / "selected_blueprints.json"
    artifact = SelectedBlueprintsArtifact(
        spec_fingerprint="fp-1",
        pragmata_version=importlib.metadata.version("pragmata"),
        llm_fingerprint="llm-fp-1",
        source_run_id="run-1",
        n_queries=10,
        batch_size=5,
        near_duplicate_tolerance=0.95,
        enable_planning_memory=True,
        embedding_model="some-other-embedding-model",
        blueprints=[_blueprint("c001")],
        created_at=datetime.now(UTC),
    )
    path.write_text(json.dumps(artifact.model_dump(mode="json")), encoding="utf-8")

    result, drifted = read_selected_blueprints_artifact(path=path, **_EXPECTED)
    assert drifted == []
    assert result is not None
    assert result.embedding_model == "some-other-embedding-model"


def test_read_selected_blueprints_reports_drift_for_pragmata_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen result written by a different pragmata version is reported as drift."""
    path = tmp_path / "selected_blueprints.json"
    _write_valid(path)

    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "pragmata":
            return "0.0.0-other"
        return real_version(name)

    monkeypatch.setattr("pragmata.core.querygen.checkpoint_read.importlib.metadata.version", fake_version)
    artifact, drifted = read_selected_blueprints_artifact(path=path, **_EXPECTED)

    assert artifact is None
    assert "pragmata_version" in {field.field for field in drifted}


def test_read_selected_blueprints_raises_for_malformed_json(tmp_path: Path) -> None:
    """Malformed file content raises a JSON decode error."""
    path = tmp_path / "selected_blueprints.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_selected_blueprints_artifact(path=path, **_EXPECTED)
