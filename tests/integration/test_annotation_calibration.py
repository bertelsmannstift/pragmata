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
from pragmata.core.paths.annotation_paths import resolve_import_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_import import PartitionManifest
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings, TaskSettings, WorkspaceSettings

pytestmark = [pytest.mark.integration, pytest.mark.annotation]

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_DATASET_ID = "testcalibration"
_CREDS: dict[str, str] = {"api_url": _API_URL, "api_key": _API_KEY}

_SETTINGS = AnnotationSettings(dataset_id=_DATASET_ID)


def _make_raw(i: int, *, n_chunks: int = 1) -> dict:
    return {
        "query": f"Question {i}?",
        "answer": f"Answer {i}.",
        "chunks": [
            {"chunk_id": f"c{i}-{j}", "doc_id": "d1", "chunk_rank": j + 1, "text": f"Chunk {i}-{j}."}
            for j in range(n_chunks)
        ],
        "context_set": "ctx-001",
        "language": "en",
    }


def _make_raw_multi_chunk(i: int, *, n_chunks: int) -> dict:
    return {
        "query": f"Question {i}?",
        "answer": f"Answer {i}.",
        "chunks": [
            {"chunk_id": f"c{i}-{k}", "doc_id": "d1", "chunk_rank": k + 1, "text": f"Chunk {i}-{k}."}
            for k in range(n_chunks)
        ],
        "context_set": "ctx-001",
        "language": "en",
    }


def _manifest_path(base_dir: Path, dataset_id: str) -> Path:
    """Resolve the partition manifest path the same way the import API does."""
    workspace = WorkspacePaths.from_base_dir(base_dir)
    return resolve_import_paths(workspace=workspace, dataset_id=dataset_id).partition_manifest


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


def test_explicit_fraction_creates_both_datasets(client: rg.Argilla, base_dir: Path) -> None:
    """Explicit calibration_fraction=0.5 routes records into both datasets."""
    teardown_resources(client, _SETTINGS)
    setup_workspaces(client, _SETTINGS)

    # Mix single- and multi-chunk records so retrieval (counts chunks) is
    # distinguishable from grounding/generation (count records). With all
    # single-chunk records the two would be indistinguishable.
    records = [_make_raw(i, n_chunks=(3 if i % 5 == 0 else 1)) for i in range(50)]
    total_chunks = sum(len(r["chunks"]) for r in records)
    assert total_chunks > len(records)  # guard: the corpus must actually be multi-chunk

    result = import_records(records, dataset_id=_DATASET_ID, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)

    prod_name = dataset_name(Task.RETRIEVAL, calibration=False, dataset_id=_DATASET_ID)
    cal_name = dataset_name(Task.RETRIEVAL, calibration=True, dataset_id=_DATASET_ID)
    assert client.datasets(prod_name, workspace="retrieval") is not None
    assert client.datasets(cal_name, workspace="retrieval") is not None

    assert all(result.calibration_count[t] > 0 for t in Task)
    assert all(result.production_count[t] > 0 for t in Task)
    # Retrieval totals count chunks; grounding/generation count records. If
    # retrieval were mistakenly counted per-record this would fail.
    assert result.calibration_count[Task.RETRIEVAL] + result.production_count[Task.RETRIEVAL] == total_chunks
    for t in (Task.GROUNDING, Task.GENERATION):
        assert result.calibration_count[t] + result.production_count[t] == len(records)
    assert all(result.calibration_fraction[t] == 0.5 for t in Task)


def test_default_fraction_resolves_from_settings(client: rg.Argilla, base_dir: Path) -> None:
    """Omitted CLI/API kwarg falls through to AnnotationSettings default (0.1)."""
    auto_id = "testdefaultfraction"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        # No calibration_fraction kwarg - settings default (0.1) wins.
        result = import_records([_make_raw(i) for i in range(50)], dataset_id=auto_id, base_dir=base_dir, **_CREDS)
        assert all(result.calibration_fraction[t] == 0.1 for t in Task)
        assert all(result.calibration_count[t] >= 1 for t in Task)  # 0.1 of 50 records ~ 5 expected
    finally:
        teardown_resources(client, auto_settings)


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
        assert all(c == 0 for c in result.calibration_count.values())
        assert all(result.production_count[t] == 5 for t in Task)
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
        manifest_path = _manifest_path(base_dir, auto_id)
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
        manifest_path = _manifest_path(base_dir, auto_id)
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())
        assert len(manifest.assignments) == 20

        # second.calibration_count must equal first.calibration_count (no new cal records)
        assert second.calibration_count == first.calibration_count
        assert all(second.production_count[t] == 20 - first.calibration_count[t] for t in Task)
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

        manifest_path = _manifest_path(base_dir, auto_id)
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())

        cal_uuids = {
            rid
            for rid, entry in manifest.assignments.items()
            if entry.grounding_generation_calibration.get(Task.GROUNDING, False)
        }
        prod_uuids = {
            rid
            for rid, entry in manifest.assignments.items()
            if not entry.grounding_generation_calibration.get(Task.GROUNDING, False)
        }

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


def test_calibration_max_items_caps_grounding(client: rg.Argilla, base_dir: Path) -> None:
    """Per-task cap binds: realised calibration count never exceeds the absolute cap."""
    auto_id = "testcap"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        records = [_make_raw(i) for i in range(30)]
        result = import_records(
            records,
            dataset_id=auto_id,
            base_dir=base_dir,
            calibration_fraction=1.0,
            calibration_max_items=3,
            **_CREDS,
        )

        assert result.calibration_count[Task.GROUNDING] == 3
        assert result.production_count[Task.GROUNDING] == 27
        assert result.calibration_max_items[Task.GROUNDING] == 3

        cal_ds = client.datasets(
            dataset_name(Task.GROUNDING, calibration=True, dataset_id=auto_id),
            workspace="grounding",
        )
        prod_ds = client.datasets(
            dataset_name(Task.GROUNDING, calibration=False, dataset_id=auto_id),
            workspace="grounding",
        )
        assert cal_ds is not None and prod_ds is not None
        assert len(list(cal_ds.records)) == 3
        assert len(list(prod_ds.records)) == 27
    finally:
        teardown_resources(client, auto_settings)


def test_per_chunk_retrieval_routing_to_argilla(client: rg.Argilla, base_dir: Path) -> None:
    """Different chunks of one record can land in different retrieval datasets on Argilla."""
    auto_id = "testperchunk"
    auto_settings = AnnotationSettings(dataset_id=auto_id)
    teardown_resources(client, auto_settings)
    setup_workspaces(client, auto_settings)

    try:
        # 30 records × 4 chunks = 120 retrieval units at fraction=0.5; per-chunk
        # independence means some pairs end up with a mixed retrieval bucket set.
        records = [_make_raw_multi_chunk(i, n_chunks=4) for i in range(30)]
        import_records(records, dataset_id=auto_id, base_dir=base_dir, calibration_fraction=0.5, **_CREDS)

        manifest_path = _manifest_path(base_dir, auto_id)
        manifest = PartitionManifest.model_validate_json(manifest_path.read_text())

        mixed = sum(
            1
            for entry in manifest.assignments.values()
            if 0 < sum(entry.retrieval_chunk_calibration.values()) < len(entry.retrieval_chunk_calibration)
        )
        assert mixed > 0, "per-chunk independence not realised - no pair had a mixed retrieval bucket"

        cal_ds = client.datasets(
            dataset_name(Task.RETRIEVAL, calibration=True, dataset_id=auto_id),
            workspace="retrieval",
        )
        prod_ds = client.datasets(
            dataset_name(Task.RETRIEVAL, calibration=False, dataset_id=auto_id),
            workspace="retrieval",
        )
        assert cal_ds is not None and prod_ds is not None

        expected_cal_chunks = {
            chunk_id
            for entry in manifest.assignments.values()
            for chunk_id, is_cal in entry.retrieval_chunk_calibration.items()
            if is_cal
        }
        expected_prod_chunks = {
            chunk_id
            for entry in manifest.assignments.values()
            for chunk_id, is_cal in entry.retrieval_chunk_calibration.items()
            if not is_cal
        }
        cal_chunk_ids = {r.metadata["chunk_id"] for r in cal_ds.records}
        prod_chunk_ids = {r.metadata["chunk_id"] for r in prod_ds.records}
        assert cal_chunk_ids == expected_cal_chunks
        assert prod_chunk_ids == expected_prod_chunks
        assert cal_chunk_ids.isdisjoint(prod_chunk_ids)
    finally:
        teardown_resources(client, auto_settings)


class TestPerWorkspaceMinSubmitted:
    """Inherited carve-out reaches the live Argilla dataset's distribution config."""

    def test_workspace_carve_out_creates_correct_dataset_min_submitted(
        self, client: rg.Argilla, base_dir: Path
    ) -> None:
        """Per-task ``production_min_submitted`` carve-out flows into the dataset's distribution.

        Deployment default is 1; the retrieval workspace overrides to 2; the
        retrieval task carves out 4. The created Argilla dataset's
        ``distribution.min_submitted`` must reflect the inherited value (4).
        """
        auto_id = "testcarveout"
        carved_settings = AnnotationSettings(
            dataset_id=auto_id,
            production_min_submitted=1,
            workspaces={
                "retrieval": WorkspaceSettings(
                    production_min_submitted=2,
                    tasks={Task.RETRIEVAL: TaskSettings(production_min_submitted=4)},
                ),
                "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        teardown_resources(client, carved_settings)
        setup_workspaces(client, carved_settings)

        try:
            records = [_make_raw(i) for i in range(3)]
            import_records(
                records,
                dataset_id=auto_id,
                base_dir=base_dir,
                calibration_fraction=0.0,
                **_CREDS,
            )

            prod_name = dataset_name(Task.RETRIEVAL, calibration=False, dataset_id=auto_id)
            prod_ds = client.datasets(prod_name, workspace="retrieval")
            assert prod_ds is not None
            # Task-level carve-out (4) wins over workspace (2) and deployment (1).
            assert prod_ds.distribution.min_submitted == 4

            # Sibling workspace inherits deployment default (1).
            grounding_prod = client.datasets(
                dataset_name(Task.GROUNDING, calibration=False, dataset_id=auto_id),
                workspace="grounding",
            )
            assert grounding_prod is not None
            assert grounding_prod.distribution.min_submitted == 1
        finally:
            teardown_resources(client, carved_settings)
