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


class TestUpsertRecordMetadata:
    def test_full_dict_merge_preserves_existing_keys(self) -> None:
        """Argilla metadata is REPLACE-not-merge; helper must send the FULL merged dict."""
        dataset = _mock_dataset()
        record = _mock_record(
            {"record_uuid": "u1", "chunk_id": "c1", "chunk_rank": 3, "doc_id": "d1"},
            record_id="rec-1",
        )

        upsert_record_metadata(dataset, record, {"n_retrieved_chunks": 5})

        dataset.records.log.assert_called_once()
        payload = dataset.records.log.call_args[0][0]
        assert payload == [
            {
                "id": "rec-1",
                "metadata": {
                    "record_uuid": "u1",
                    "chunk_id": "c1",
                    "chunk_rank": 3,
                    "doc_id": "d1",
                    "n_retrieved_chunks": 5,
                },
            }
        ]

    def test_update_overrides_existing_key(self) -> None:
        dataset = _mock_dataset()
        record = _mock_record({"flag": "old", "other": 1})

        upsert_record_metadata(dataset, record, {"flag": "new"})

        payload = dataset.records.log.call_args[0][0]
        assert payload[0]["metadata"]["flag"] == "new"
        assert payload[0]["metadata"]["other"] == 1

    def test_remove_keys_clears_existing_tag(self) -> None:
        dataset = _mock_dataset()
        record = _mock_record({"chunk_id": "c1", "needs_completion": "true"})

        upsert_record_metadata(dataset, record, {}, remove_keys=["needs_completion"])

        payload = dataset.records.log.call_args[0][0]
        assert "needs_completion" not in payload[0]["metadata"]
        assert payload[0]["metadata"]["chunk_id"] == "c1"

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
