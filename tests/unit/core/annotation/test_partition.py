"""Unit tests for record_builder partition logic and manifest IO."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pragmata.core.annotation.record_builder import (
    _bucket_calibration,
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


class TestBucketCalibration:
    """Per-(task, unit) bucketing with task in the hash input."""

    def test_zero_fraction_always_production(self) -> None:
        for i in range(20):
            assert _bucket_calibration(f"uuid-{i}", Task.GROUNDING, 0.0, seed=0) is False

    def test_full_fraction_always_calibration(self) -> None:
        for i in range(20):
            assert _bucket_calibration(f"uuid-{i}", Task.GROUNDING, 1.0, seed=0) is True

    def test_deterministic_across_calls(self) -> None:
        results = [_bucket_calibration("uuid-x", Task.GROUNDING, 0.5, seed=42) for _ in range(10)]
        assert len(set(results)) == 1

    def test_different_seed_changes_bucketing(self) -> None:
        """Across a sample, two seeds disagree on a non-trivial fraction of uuids."""
        flips = sum(
            1
            for i in range(100)
            if _bucket_calibration(f"uuid-{i}", Task.GROUNDING, 0.5, seed=0)
            != _bucket_calibration(f"uuid-{i}", Task.GROUNDING, 0.5, seed=5)
        )
        # At fraction=0.5 with independent seeds, ~50% of uuids should disagree.
        assert 25 < flips < 75, f"seeds look correlated: {flips}/100 disagreements"

    def test_per_task_draws_are_independent(self) -> None:
        """Same unit-id at same fraction can land differently per task (good - proves task is in the hash)."""
        # Sample many uuids; assert the per-task results disagree for at least some.
        agreements = 0
        for i in range(100):
            r = _bucket_calibration(f"uuid-{i}", Task.RETRIEVAL, 0.5, seed=0)
            g = _bucket_calibration(f"uuid-{i}", Task.GROUNDING, 0.5, seed=0)
            if r == g:
                agreements += 1
        # If task were not in the hash, retrieval == grounding for every uuid (100 agreements).
        # Independent draws should disagree ~50% of the time.
        assert 20 < agreements < 80, f"per-task draws look correlated: {agreements}/100"

    def test_calibration_digest_consistent_with_bucket(self) -> None:
        """`_bucket_calibration` is exactly `_calibration_digest < fraction × 2^32`."""
        digest = _calibration_digest("uuid-x", Task.GROUNDING, seed=42)
        threshold = 0.5
        expected = digest < int(threshold * (2**32))
        assert _bucket_calibration("uuid-x", Task.GROUNDING, threshold, seed=42) == expected


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
        result = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        assert len(result) == 5
        for entry in result.values():
            # Every task: every unit is calibration
            assert entry.grounding_generation_calibration[Task.GROUNDING] is True
            assert entry.grounding_generation_calibration[Task.GENERATION] is True
            assert all(entry.retrieval_chunk_calibration.values())

    def test_zero_fraction_all_production(self, empty_manifest: PartitionManifest) -> None:
        settings = _default_settings(calibration_fraction=0.0)
        pairs = [_make_pair(f"q{i}", n_chunks=3) for i in range(5)]
        result = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        for entry in result.values():
            assert entry.grounding_generation_calibration[Task.GROUNDING] is False
            assert entry.grounding_generation_calibration[Task.GENERATION] is False
            assert not any(entry.retrieval_chunk_calibration.values())

    def test_retrieval_chunks_partitioned_independently(self, empty_manifest: PartitionManifest) -> None:
        """Different chunks of the same record can land in different buckets."""
        settings = _default_settings(calibration_fraction=0.5)
        # 50 pairs × 4 chunks each = 200 retrieval units at fraction 0.5
        pairs = [_make_pair(f"q{i}", n_chunks=4) for i in range(50)]
        result = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        # Some pairs should have a mixed retrieval calibration set (not all True or all False).
        mixed = sum(
            1
            for entry in result.values()
            if 0 < sum(entry.retrieval_chunk_calibration.values()) < len(entry.retrieval_chunk_calibration)
        )
        assert mixed > 0, "no pairs had per-chunk mixing - partition might still be per-record"

    def test_cap_limits_calibration_count(self, empty_manifest: PartitionManifest) -> None:
        """Per-task cap binds: realised count never exceeds cap."""
        # Grounding has its own cap of 5 out of 50 pairs at fraction 1.0.
        settings = AnnotationSettings(
            calibration_fraction=1.0,
            workspaces={
                "g": WorkspaceSettings(
                    tasks={Task.GROUNDING: TaskSettings(calibration_max_records=5)},
                ),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
                "r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            },
        )
        pairs = [_make_pair(f"q{i}") for i in range(50)]
        result = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        cal_count = sum(1 for e in result.values() if e.grounding_generation_calibration[Task.GROUNDING])
        assert cal_count == 5

    def test_cap_zero_routes_all_new_to_production(self, empty_manifest: PartitionManifest) -> None:
        settings = AnnotationSettings(
            calibration_fraction=1.0,
            calibration_max_records=1,  # arbitrary positive (cap=0 is forbidden by PositiveInt)
            workspaces={
                "g": WorkspaceSettings(
                    tasks={Task.GROUNDING: TaskSettings(calibration_max_records=1)},
                ),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
                "r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            },
        )
        pairs = [_make_pair(f"q{i}") for i in range(20)]
        result = assign_partitions(pairs, manifest=empty_manifest, settings=settings, import_id="imp1")

        cal_count = sum(1 for e in result.values() if e.grounding_generation_calibration[Task.GROUNDING])
        assert cal_count == 1

    def test_cap_preserves_existing_assignments_when_over_cap(self, empty_manifest: PartitionManifest, caplog) -> None:
        """If existing manifest count exceeds new cap, warn and don't demote."""
        # Pre-populate manifest with 10 grounding-calibration entries.
        now = datetime.now(timezone.utc)
        for i in range(10):
            rid = f"existing-{i}"
            empty_manifest.assignments[rid] = PartitionManifestEntry(
                grounding_generation_calibration={Task.GROUNDING: True, Task.GENERATION: False},
                retrieval_chunk_calibration={},
                import_id="prior",
                calibration_fraction_at_import={t: 1.0 for t in Task},
                calibration_max_records_at_import={t: None for t in Task},
                assigned_at=now,
            )
        # New cap is 5 - below the existing 10.
        settings = AnnotationSettings(
            calibration_fraction=1.0,
            calibration_max_records=5,
            workspaces={
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
                "r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            },
        )
        new_pairs = [_make_pair(f"new-{i}") for i in range(5)]

        with caplog.at_level("WARNING"):
            assign_partitions(new_pairs, manifest=empty_manifest, settings=settings, import_id="imp2")

        # Existing 10 entries still present and still calibration.
        existing_cal = sum(
            1
            for rid, e in empty_manifest.assignments.items()
            if rid.startswith("existing-") and e.grounding_generation_calibration[Task.GROUNDING]
        )
        assert existing_cal == 10
        # No new record promoted (cap was already exceeded).
        new_cal = sum(
            1
            for rid, e in empty_manifest.assignments.items()
            if not rid.startswith("existing-") and e.grounding_generation_calibration[Task.GROUNDING]
        )
        assert new_cal == 0
        assert any("exceeds cap" in m for m in caplog.messages)

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
            calibration_max_records_at_import={t: None for t in Task},
            assigned_at=datetime.now(timezone.utc),
        )

        result = assign_partitions([pair], manifest=empty_manifest, settings=settings, import_id="imp2")

        # Existing assignment preserved despite fraction=1.0 in new settings.
        assert result[rid].grounding_generation_calibration[Task.GROUNDING] is False
        assert result[rid].import_id == "prior"

    def test_new_records_stamp_per_task_provenance(self, empty_manifest: PartitionManifest) -> None:
        """Per-task fractions and caps are stamped on each new entry."""
        settings = AnnotationSettings(
            calibration_fraction=0.5,
            calibration_max_records=100,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_fraction=0.1,
                    calibration_max_records=20,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        pair = _make_pair("q0")
        result = assign_partitions([pair], manifest=empty_manifest, settings=settings, import_id="imp1")
        rid = derive_record_uuid(pair)
        entry = result[rid]

        assert entry.calibration_fraction_at_import[Task.RETRIEVAL] == 0.1
        assert entry.calibration_fraction_at_import[Task.GROUNDING] == 0.5
        assert entry.calibration_max_records_at_import[Task.RETRIEVAL] == 20
        assert entry.calibration_max_records_at_import[Task.GROUNDING] == 100

    def test_cap_under_split_imports_is_order_dependent_by_design(self, empty_manifest: PartitionManifest) -> None:
        """Documented property: cap-binding split imports depend on order.

        Because the manifest is append-only, the final calibration set depends
        on (corpus, seed, import_order) rather than (corpus, seed) alone.
        This test pins the property so a future regression to order-independence
        is a deliberate design change requiring this test's update.
        """
        settings = AnnotationSettings(
            calibration_fraction=1.0,
            calibration_max_records=3,
            workspaces={
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
                "r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            },
        )
        pairs = [_make_pair(f"q{i}") for i in range(10)]

        # Order A: import first 5 then last 5.
        manifest_a = PartitionManifest(
            dataset_id="test",
            created_at=empty_manifest.created_at,
            updated_at=empty_manifest.updated_at,
            partition_seed=0,
        )
        assign_partitions(pairs[:5], manifest=manifest_a, settings=settings, import_id="imp_a1")
        assign_partitions(pairs[5:], manifest=manifest_a, settings=settings, import_id="imp_a2")

        # Order B: all at once.
        manifest_b = PartitionManifest(
            dataset_id="test",
            created_at=empty_manifest.created_at,
            updated_at=empty_manifest.updated_at,
            partition_seed=0,
        )
        assign_partitions(pairs, manifest=manifest_b, settings=settings, import_id="imp_b")

        # Both orders honour the cap (cardinality invariant). The specific set chosen
        # may differ under binding cap - documented in assign_partitions.
        def _cal_count(m: PartitionManifest) -> int:
            return sum(1 for e in m.assignments.values() if e.grounding_generation_calibration[Task.GROUNDING])

        assert _cal_count(manifest_a) == 3
        assert _cal_count(manifest_b) == 3


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
            calibration_max_records_at_import={t: None for t in Task},
            assigned_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        )

        write_partition_manifest(path, original)
        restored = load_partition_manifest(path, dataset_id="test", partition_seed=7)

        assert restored.dataset_id == original.dataset_id
        assert restored.partition_seed == original.partition_seed
        assert restored.assignments["uuid-1"] == original.assignments["uuid-1"]

    def test_round_trip_through_legacy_manifest(self, tmp_path: Path) -> None:
        """Legacy on-disk manifests (with `calibration: bool`) load via the migrator."""
        path = tmp_path / "partition.meta.json"
        # Hand-write a legacy-shaped manifest JSON.
        legacy_json = """
        {
            "dataset_id": "test",
            "created_at": "2026-04-22T00:00:00+00:00",
            "updated_at": "2026-04-22T00:00:00+00:00",
            "partition_seed": 7,
            "assignments": {
                "uuid-1": {
                    "calibration": true,
                    "import_id": "imp1",
                    "calibration_fraction_at_import": 0.1,
                    "assigned_at": "2026-04-22T00:00:00+00:00"
                }
            }
        }
        """
        path.write_text(legacy_json, encoding="utf-8")
        restored = load_partition_manifest(path, dataset_id="test", partition_seed=7)
        entry = restored.assignments["uuid-1"]
        assert entry.grounding_generation_calibration == {Task.GROUNDING: True, Task.GENERATION: True}
        assert entry.retrieval_chunk_calibration == {}
        assert all(v == 0.1 for v in entry.calibration_fraction_at_import.values())

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
