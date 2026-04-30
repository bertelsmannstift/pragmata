"""Unit tests for record_builder partition logic and manifest IO."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pragmata.core.annotation.record_builder import (
    _bucket_calibration,
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


def _make_pair(query: str = "Q?") -> QueryResponsePair:
    return QueryResponsePair(
        query=query,
        answer="A.",
        chunks=[Chunk(chunk_id=f"c-{query}", doc_id="d", chunk_rank=1, text="t")],
        context_set="ctx",
    )


class TestBucketCalibration:
    def test_zero_fraction_always_production(self) -> None:
        for i in range(20):
            assert _bucket_calibration(f"uuid-{i}", 0.0, seed=0) is False

    def test_full_fraction_always_calibration(self) -> None:
        for i in range(20):
            assert _bucket_calibration(f"uuid-{i}", 1.0, seed=0) is True

    def test_deterministic_across_calls(self) -> None:
        results = [_bucket_calibration("uuid-x", 0.5, seed=42) for _ in range(10)]
        assert len(set(results)) == 1

    def test_different_seed_different_bucket(self) -> None:
        # Probabilistic: different seeds usually flip at least one of N records
        flips = 0
        for i in range(50):
            uuid = f"uuid-{i}"
            if _bucket_calibration(uuid, 0.5, seed=0) != _bucket_calibration(uuid, 0.5, seed=99):
                flips += 1
        assert flips > 0

    def test_fraction_bounded_for_uniform_uuids(self) -> None:
        # 10% target on 1000 records should be near 100, with wide tolerance for hash variance
        n_cal = sum(1 for i in range(1000) if _bucket_calibration(f"uuid-{i}", 0.1, seed=7))
        assert 50 <= n_cal <= 150


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

    def test_new_records_get_assignments(self, empty_manifest: PartitionManifest) -> None:
        pairs = [_make_pair(f"q{i}") for i in range(10)]
        result = assign_partitions(pairs, manifest=empty_manifest, fraction=1.0, import_id="imp1")

        assert len(result) == 10
        assert all(result.values())  # fraction=1.0 -> all calibration
        assert len(empty_manifest.assignments) == 10

    def test_existing_assignments_respected(self, empty_manifest: PartitionManifest) -> None:
        # Pre-seed manifest with one record assigned to production
        pair = _make_pair("q0")
        rid = derive_record_uuid(pair)
        empty_manifest.assignments[rid] = PartitionManifestEntry(
            calibration=False,
            import_id="prior",
            calibration_fraction_at_import=0.0,
            assigned_at=datetime.now(timezone.utc),
        )

        # Re-import with fraction=1.0 should keep the existing production assignment
        result = assign_partitions([pair], manifest=empty_manifest, fraction=1.0, import_id="imp2")

        assert result[rid] is False
        assert empty_manifest.assignments[rid].import_id == "prior"

    def test_new_records_use_current_fraction(self, empty_manifest: PartitionManifest) -> None:
        pair = _make_pair("q0")
        rid = derive_record_uuid(pair)
        assign_partitions([pair], manifest=empty_manifest, fraction=1.0, import_id="imp1")

        entry = empty_manifest.assignments[rid]
        assert entry.calibration_fraction_at_import == 1.0
        assert entry.import_id == "imp1"

    def test_mixed_existing_and_new_records(self, empty_manifest: PartitionManifest) -> None:
        existing = _make_pair("q-existing")
        existing_rid = derive_record_uuid(existing)
        empty_manifest.assignments[existing_rid] = PartitionManifestEntry(
            calibration=True,
            import_id="prior",
            calibration_fraction_at_import=0.5,
            assigned_at=datetime.now(timezone.utc),
        )

        new_pair = _make_pair("q-new")
        result = assign_partitions([existing, new_pair], manifest=empty_manifest, fraction=0.0, import_id="imp2")

        assert result[existing_rid] is True  # preserved
        assert result[derive_record_uuid(new_pair)] is False  # new, fraction=0


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
            calibration=True,
            import_id="imp1",
            calibration_fraction_at_import=0.1,
            assigned_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        )

        write_partition_manifest(path, original)
        restored = load_partition_manifest(path, dataset_id="test", partition_seed=7)

        assert restored.dataset_id == original.dataset_id
        assert restored.partition_seed == original.partition_seed
        assert restored.assignments["uuid-1"] == original.assignments["uuid-1"]

    def test_atomic_write_creates_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "subdir" / "partition.meta.json"
        manifest = load_partition_manifest(path, dataset_id="x", partition_seed=0)
        write_partition_manifest(path, manifest)
        assert path.exists()
