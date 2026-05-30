"""Unit tests for shared safe metadata ops (used by status --tag-incomplete + backfill)."""

from unittest.mock import MagicMock

import argilla as rg

from pragmata.core.annotation.metadata_ops import (
    ensure_metadata_property,
    upsert_record_metadata,
)


def _mock_dataset(*, existing_metadata_props: list[str] | None = None) -> MagicMock:
    dataset = MagicMock()
    dataset.name = "ds"
    metadata = MagicMock()
    metadata.__getitem__ = MagicMock(side_effect=lambda key: key if key in (existing_metadata_props or []) else None)
    dataset.settings.metadata = metadata
    return dataset


def _mock_record(metadata: dict[str, object], record_id: str = "rec-1") -> MagicMock:
    record = MagicMock()
    record.id = record_id
    record.metadata = metadata
    return record


class TestEnsureMetadataProperty:
    def test_adds_when_absent(self) -> None:
        dataset = _mock_dataset(existing_metadata_props=[])
        # The MagicMock for IntegerMetadataProperty can be unauthenticated; use a real instance
        # via a stand-in MagicMock to avoid the Argilla SDK requiring credentials at construction.
        prop = MagicMock(spec=rg.IntegerMetadataProperty)
        prop.name = "n_retrieved_chunks"

        added = ensure_metadata_property(dataset, prop)

        assert added is True
        dataset.settings.add.assert_called_once_with(prop)
        dataset.settings.update.assert_called_once()

    def test_skips_when_present(self) -> None:
        dataset = _mock_dataset(existing_metadata_props=["n_retrieved_chunks"])
        prop = MagicMock(spec=rg.IntegerMetadataProperty)
        prop.name = "n_retrieved_chunks"

        added = ensure_metadata_property(dataset, prop)

        assert added is False
        dataset.settings.add.assert_not_called()
        dataset.settings.update.assert_not_called()


def _logged_record(dataset: MagicMock) -> "object":
    """Pull the (single) record out of dataset.records.log([rg.Record])."""
    dataset.records.log.assert_called_once()
    payload = dataset.records.log.call_args[0][0]
    assert len(payload) == 1
    return payload[0]


class TestBuildMetadataUpsert:
    def test_returns_record_with_full_merged_metadata(self) -> None:
        from pragmata.core.annotation.metadata_ops import build_metadata_upsert

        record = _mock_record(
            {"record_uuid": "u1", "chunk_id": "c1", "chunk_rank": 3, "doc_id": "d1"},
            record_id="rec-1",
        )

        upsert = build_metadata_upsert(record, {"n_retrieved_chunks": 5})

        assert upsert is not None
        assert upsert.id == "rec-1"
        assert dict(upsert.metadata) == {
            "record_uuid": "u1",
            "chunk_id": "c1",
            "chunk_rank": 3,
            "doc_id": "d1",
            "n_retrieved_chunks": 5,
        }

    def test_returns_none_when_unchanged(self) -> None:
        from pragmata.core.annotation.metadata_ops import build_metadata_upsert

        record = _mock_record({"chunk_id": "c1", "flag": "yes"})
        assert build_metadata_upsert(record, {"flag": "yes"}) is None

    def test_remove_keys_drops_existing(self) -> None:
        from pragmata.core.annotation.metadata_ops import build_metadata_upsert

        record = _mock_record({"chunk_id": "c1", "needs_completion": "true"})
        upsert = build_metadata_upsert(record, {}, remove_keys=["needs_completion"])
        assert upsert is not None
        assert "needs_completion" not in dict(upsert.metadata)
        assert dict(upsert.metadata)["chunk_id"] == "c1"


class TestUpsertRecordMetadata:
    def test_full_dict_merge_preserves_existing_keys(self) -> None:
        """Argilla metadata is REPLACE-not-merge; helper must send the FULL merged dict."""
        dataset = _mock_dataset()
        record = _mock_record(
            {"record_uuid": "u1", "chunk_id": "c1", "chunk_rank": 3, "doc_id": "d1"},
            record_id="rec-1",
        )

        upsert_record_metadata(dataset, record, {"n_retrieved_chunks": 5})

        logged = _logged_record(dataset)
        # Must be an rg.Record, not a dict - dict payloads get their non-flat
        # keys silently dropped by Argilla's IngestedRecordMapper, wiping metadata.
        import argilla as rg

        assert isinstance(logged, rg.Record)
        assert logged.id == "rec-1"
        assert dict(logged.metadata) == {
            "record_uuid": "u1",
            "chunk_id": "c1",
            "chunk_rank": 3,
            "doc_id": "d1",
            "n_retrieved_chunks": 5,
        }

    def test_update_overrides_existing_key(self) -> None:
        dataset = _mock_dataset()
        record = _mock_record({"flag": "old", "other": 1})

        upsert_record_metadata(dataset, record, {"flag": "new"})

        logged = _logged_record(dataset)
        assert dict(logged.metadata)["flag"] == "new"
        assert dict(logged.metadata)["other"] == 1

    def test_remove_keys_clears_existing_tag(self) -> None:
        dataset = _mock_dataset()
        record = _mock_record({"chunk_id": "c1", "needs_completion": "true"})

        upsert_record_metadata(dataset, record, {}, remove_keys=["needs_completion"])

        logged = _logged_record(dataset)
        assert "needs_completion" not in dict(logged.metadata)
        assert dict(logged.metadata)["chunk_id"] == "c1"

    def test_noop_when_metadata_unchanged(self) -> None:
        """No write when updates leave metadata identical (idempotent)."""
        dataset = _mock_dataset()
        record = _mock_record({"chunk_id": "c1", "flag": "yes"})

        upsert_record_metadata(dataset, record, {"flag": "yes"})

        dataset.records.log.assert_not_called()

    def test_noop_when_remove_key_absent(self) -> None:
        dataset = _mock_dataset()
        record = _mock_record({"chunk_id": "c1"})

        upsert_record_metadata(dataset, record, {}, remove_keys=["needs_completion"])

        dataset.records.log.assert_not_called()
