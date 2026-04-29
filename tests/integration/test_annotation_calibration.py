"""Integration tests for the calibration / production split against a live Argilla.

Run with: pytest tests/integration/test_annotation_calibration.py -m "integration and annotation" -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

from pathlib import Path

import argilla as rg
import pytest

from pragmata.api.annotation_import import import_records
from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.annotation.setup import setup_workspaces, teardown_resources
from pragmata.core.schemas.annotation_import import PartitionManifest
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

pytestmark = [pytest.mark.integration, pytest.mark.annotation]

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_DATASET_ID = "testcalibration"
_CREDS: dict[str, str] = {"api_url": _API_URL, "api_key": _API_KEY}

_SETTINGS = AnnotationSettings(dataset_id=_DATASET_ID)


def _make_raw(i: int) -> dict:
    return {
        "query": f"Question {i}?",
        "answer": f"Answer {i}.",
        "chunks": [{"chunk_id": f"c{i}", "doc_id": "d1", "chunk_rank": 1, "text": f"Chunk {i}."}],
        "context_set": "ctx-001",
        "language": "en",
    }


@pytest.fixture(scope="module")
def client() -> rg.Argilla:
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


@pytest.fixture(autouse=True, scope="module")
def clean_environment(client: rg.Argilla):
    teardown_resources(client, _SETTINGS)
    setup_workspaces(client, _SETTINGS)
    yield
    teardown_resources(client, _SETTINGS)


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    """Per-test workspace so each test owns its partition manifest."""
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_calibration_fraction_creates_both_datasets(client: rg.Argilla, base_dir: Path) -> None:
    """Default fraction=0.1 routes ~10% to calibration; both datasets exist."""
    teardown_resources(client, _SETTINGS)
    setup_workspaces(client, _SETTINGS)

    records = [_make_raw(i) for i in range(50)]
    result = import_records(records, dataset_id=_DATASET_ID, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)

    # Both production and calibration datasets exist for retrieval
    prod_name = dataset_name(Task.RETRIEVAL, calibration=False, dataset_id=_DATASET_ID)
    cal_name = dataset_name(Task.RETRIEVAL, calibration=True, dataset_id=_DATASET_ID)
    assert client.datasets(prod_name, workspace="retrieval") is not None
    assert client.datasets(cal_name, workspace="retrieval") is not None

    # Both populated
    assert result.calibration_count > 0
    assert result.production_count > 0
    assert result.calibration_count + result.production_count == len(records)


def test_calibration_fraction_zero_skips_calibration_dataset(client: rg.Argilla, base_dir: Path) -> None:
    """fraction=0 routes everything to production; calibration dataset is not created."""
    auto_id = "testcalibrationzero"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        result = import_records(
            [_make_raw(i) for i in range(5)],
            dataset_id=auto_id,
            base_dir=base_dir,
            calibration_fraction=0.0,
            **_CREDS,
        )

        prod_name = dataset_name(Task.RETRIEVAL, calibration=False, dataset_id=auto_id)
        cal_name = dataset_name(Task.RETRIEVAL, calibration=True, dataset_id=auto_id)
        assert client.datasets(prod_name, workspace="retrieval") is not None
        assert client.datasets(cal_name, workspace="retrieval") is None
        assert result.calibration_count == 0
        assert result.production_count == 5
    finally:
        teardown_resources(client, auto_settings)


def test_reimport_locks_partition_assignments(client: rg.Argilla, base_dir: Path) -> None:
    """Re-importing the same batch with a different fraction does not move records."""
    auto_id = "testreimport"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        records = [_make_raw(i) for i in range(20)]

        first = import_records(records, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)
        second = import_records(records, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.0, **_CREDS)

        # Same input + locked manifest = identical assignments
        assert first.calibration_count == second.calibration_count
        assert first.production_count == second.production_count

        # Manifest sidecar exists and has all 20 entries
        manifest_path = base_dir / "annotation" / "imports" / auto_id / "partition.meta.json"
        assert manifest_path.exists()
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())
        assert len(manifest.assignments) == 20
    finally:
        teardown_resources(client, auto_settings)


def test_growing_batch_partitions_new_records_only(client: rg.Argilla, base_dir: Path) -> None:
    """Appending new records on a second import partitions only the new ones."""
    auto_id = "testgrowing"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        first_batch = [_make_raw(i) for i in range(10)]
        first = import_records(first_batch, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)

        # Second import: 10 prior + 10 new
        second_batch = first_batch + [_make_raw(i) for i in range(10, 20)]
        second = import_records(second_batch, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.0, **_CREDS)

        # The first 10 keep their original assignments. The next 10 all go production
        # because fraction=0 on the second import.
        manifest_path = base_dir / "annotation" / "imports" / auto_id / "partition.meta.json"
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())
        assert len(manifest.assignments) == 20

        # second.calibration_count must equal first.calibration_count (no new cal records)
        assert second.calibration_count == first.calibration_count
        assert second.production_count == 20 - first.calibration_count
    finally:
        teardown_resources(client, auto_settings)


def test_records_carry_calibration_metadata_to_argilla(client: rg.Argilla, base_dir: Path) -> None:
    """Imported Argilla records exist in the dataset matching their assignment."""
    auto_id = "testmetadata"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        records = [_make_raw(i) for i in range(10)]
        import_records(records, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)

        manifest_path = base_dir / "annotation" / "imports" / auto_id / "partition.meta.json"
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())

        cal_uuids = {rid for rid, entry in manifest.assignments.items() if entry.calibration}
        prod_uuids = {rid for rid, entry in manifest.assignments.items() if not entry.calibration}

        prod_ds = client.datasets(
            dataset_name(Task.GROUNDING, calibration=False, dataset_id=auto_id),
            workspace="grounding",
        )
        cal_ds = client.datasets(
            dataset_name(Task.GROUNDING, calibration=True, dataset_id=auto_id),
            workspace="grounding",
        )

        assert prod_ds is not None
        prod_record_uuids = {r.metadata["record_uuid"] for r in prod_ds.records}
        assert prod_record_uuids == prod_uuids

        if cal_uuids:
            assert cal_ds is not None
            cal_record_uuids = {r.metadata["record_uuid"] for r in cal_ds.records}
            assert cal_record_uuids == cal_uuids
    finally:
        teardown_resources(client, auto_settings)
