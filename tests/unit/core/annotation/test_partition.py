"""Unit tests for record_builder partition logic and manifest IO."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pragmata.core.annotation.record_builder import (
    _calibration_digest,
    assign_partitions,
    derive_record_uuid,
    load_partition_manifest,
    write_partition_manifest,
)
from pragmata.core.schemas.annotation_import import (
    Chunk,
    PartitionManifest,
    PartitionManifestEntry,
    QueryResponsePair,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import (
    AnnotationSettings,
    TaskSettings,
    WorkspaceSettings,
)


def _make_pair(query: str = "Q?", *, n_chunks: int = 1) -> QueryResponsePair:
    return QueryResponsePair(
        query=query,
        answer="A.",
        chunks=[Chunk(chunk_id=f"c-{query}-{i}", doc_id="d", chunk_rank=i + 1, text=f"t{i}") for i in range(n_chunks)],
        context_set="ctx",
    )


def _default_settings(**overrides) -> AnnotationSettings:
    """Default workspace topology with overridable fields."""
    return AnnotationSettings(
        workspaces={
            "retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
        },
        **overrides,
    )


class TestCalibrationDigest:
    """Per-(task, unit) digest properties — production primitive used by ``assign_partitions``."""

    def test_deterministic_across_calls(self) -> None:
        results = [_calibration_digest("uuid-x", Task.GROUNDING, seed=42) for _ in range(10)]
        assert len(set(results)) == 1

    def test_digest_in_uint32_range(self) -> None:
        for i in range(20):
            d = _calibration_digest(f"uuid-{i}", Task.GROUNDING, seed=0)
            assert 0 <= d < 2**32

    def test_different_seed_changes_digest(self) -> None:
        """At threshold=0.5, two seeds disagree on a non-trivial fraction of uuids."""
        threshold = int(0.5 * (2**32))
        flips = sum(
            1
            for i in range(100)
            if (_calibration_digest(f"uuid-{i}", Task.GROUNDING, seed=0) < threshold)
            != (_calibration_digest(f"uuid-{i}", Task.GROUNDING, seed=5) < threshold)
        )
        assert 25 < flips < 75, f"seeds look correlated: {flips}/100 disagreements"

    def test_per_task_draws_are_independent(self) -> None:
        """Same unit-id at same threshold can land differently per task — proves task is in the hash."""
        threshold = int(0.5 * (2**32))
        agreements = sum(
            1
            for i in range(100)
            if (_calibration_digest(f"uuid-{i}", Task.RETRIEVAL, seed=0) < threshold)
            == (_calibration_digest(f"uuid-{i}", Task.GROUNDING, seed=0) < threshold)
        )
        assert 20 < agreements < 80, f"per-task draws look correlated: {agreements}/100"


class TestAssignPartitions:
    @pytest.fixture
    def empty_manifest(self) -> PartitionManifest:
        now = datetime.now(timezone.utc)
        return PartitionManifest(
            dataset_id="test",
            created_at=now,
            updated_at=now,
            partition_seed=0,
            assignments={},
        )

    def test_full_fraction_all_calibration(self, empty_manifest: PartitionManifest) -> None:
        settings = _default_settings(calibration_fraction=1.0)
        pairs = [_make_pair(f"q{i}") for i in range(5)]
        partition = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        assert len(partition.assignments) == 5
        for entry in partition.assignments.values():
            # Every task: every unit is calibration
            assert entry.grounding_generation_calibration[Task.GROUNDING] is True
            assert entry.grounding_generation_calibration[Task.GENERATION] is True
            assert all(entry.retrieval_chunk_calibration.values())

    def test_zero_fraction_all_production(self, empty_manifest: PartitionManifest) -> None:
        settings = _default_settings(calibration_fraction=0.0)
        pairs = [_make_pair(f"q{i}", n_chunks=3) for i in range(5)]
        partition = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        for entry in partition.assignments.values():
            assert entry.grounding_generation_calibration[Task.GROUNDING] is False
            assert entry.grounding_generation_calibration[Task.GENERATION] is False
            assert not any(entry.retrieval_chunk_calibration.values())

    def test_retrieval_chunks_partitioned_independently(self, empty_manifest: PartitionManifest) -> None:
        """Different chunks of the same record can land in different buckets."""
        settings = _default_settings(calibration_fraction=0.5)
        # 50 pairs × 4 chunks each = 200 retrieval units at fraction 0.5
        pairs = [_make_pair(f"q{i}", n_chunks=4) for i in range(50)]
        partition = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        # Some pairs should have a mixed retrieval calibration set (not all True or all False).
        mixed = sum(
            1
            for entry in partition.assignments.values()
            if 0 < sum(entry.retrieval_chunk_calibration.values()) < len(entry.retrieval_chunk_calibration)
        )
        assert mixed > 0, "no pairs had per-chunk mixing - partition might still be per-record"

    def test_existing_assignments_preserved_across_reimports(self, empty_manifest: PartitionManifest) -> None:
        settings = _default_settings(calibration_fraction=1.0)
        pair = _make_pair("q0")
        rid = derive_record_uuid(pair)

        # Pre-seed manifest with this record assigned to production for grounding only.
        empty_manifest.assignments[rid] = PartitionManifestEntry(
            grounding_generation_calibration={Task.GROUNDING: False, Task.GENERATION: False},
            retrieval_chunk_calibration={},
            import_id="prior",
            calibration_fraction_at_import={t: 0.0 for t in Task},
            assigned_at=datetime.now(timezone.utc),
        )

        partition = assign_partitions([pair], manifest=empty_manifest, settings=settings, import_id="imp2")

        # Existing assignment preserved despite fraction=1.0 in new settings.
        assert partition.assignments[rid].grounding_generation_calibration[Task.GROUNDING] is False
        assert partition.assignments[rid].import_id == "prior"

    def test_new_records_stamp_per_task_fraction(self, empty_manifest: PartitionManifest) -> None:
        """Per-task fractions are stamped on each new entry."""
        settings = AnnotationSettings(
            calibration_fraction=0.5,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_fraction=0.1,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        pair = _make_pair("q0")
        partition = assign_partitions([pair], manifest=empty_manifest, settings=settings, import_id="imp1")
        rid = derive_record_uuid(pair)
        entry = partition.assignments[rid]

        assert entry.calibration_fraction_at_import[Task.RETRIEVAL] == 0.1
        assert entry.calibration_fraction_at_import[Task.GROUNDING] == 0.5


class TestManifestIO:
    def test_load_empty_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "partition.meta.json"
        manifest = load_partition_manifest(path, dataset_id="test", partition_seed=42)

        assert manifest.dataset_id == "test"
        assert manifest.partition_seed == 42
        assert manifest.assignments == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "partition.meta.json"
        original = load_partition_manifest(path, dataset_id="test", partition_seed=7)
        original.assignments["uuid-1"] = PartitionManifestEntry(
            grounding_generation_calibration={Task.GROUNDING: True, Task.GENERATION: False},
            retrieval_chunk_calibration={"chunk-a": True, "chunk-b": False},
            import_id="imp1",
            calibration_fraction_at_import={t: 0.1 for t in Task},
            assigned_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        )

        write_partition_manifest(path, original)
        restored = load_partition_manifest(path, dataset_id="test", partition_seed=7)

        assert restored.dataset_id == original.dataset_id
        assert restored.partition_seed == original.partition_seed
        assert restored.assignments["uuid-1"] == original.assignments["uuid-1"]

    def test_write_requires_existing_parent(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "subdir" / "partition.meta.json"
        manifest = load_partition_manifest(path, dataset_id="x", partition_seed=0)
        with pytest.raises(FileNotFoundError):
            write_partition_manifest(path, manifest)

    def test_load_rejects_dataset_id_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "partition.meta.json"
        original = load_partition_manifest(path, dataset_id="scope-a", partition_seed=0)
        write_partition_manifest(path, original)

        with pytest.raises(ValueError, match="scope-a"):
            load_partition_manifest(path, dataset_id="scope-b", partition_seed=0)
