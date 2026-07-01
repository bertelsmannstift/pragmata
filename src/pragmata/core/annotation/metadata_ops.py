"""Shared safe metadata operations for live Argilla mutations.

Used by both the ``--tag-partial-panels`` write path in ``panel_status`` and the
one-off backfill script under ``scripts/``. Centralises the two safety
invariants that every metadata write must respect on Argilla v2.8.0:

1. **Argilla metadata is REPLACE, not merge.** Every ``dataset.records.log``
   call replaces the record's metadata wholesale. To avoid clobbering
   existing keys, always fetch the current dict, merge in the update, and
   send the FULL resulting dict.
2. **Property declaration is additive and idempotent.** Adding a metadata
   property to an existing dataset is non-destructive, but the SDK raises
   if the property already exists (with override warning). Skip the add
   when the property is already present.

Writes go via ``rg.Record(id=..., metadata={...})`` rather than a raw dict
payload: the SDK's ``IngestedRecordMapper`` flattens dict keys against the
dataset schema, so a ``{"id": ..., "metadata": {...}}`` shape would treat
"metadata" as an unknown top-level attribute and silently send an empty
metadata dict (wiping the record). Passing an ``rg.Record`` bypasses the
mapper.
"""

import logging
from collections.abc import Iterable, Mapping

import argilla as rg

logger = logging.getLogger(__name__)


def ensure_metadata_property(dataset: rg.Dataset, prop: rg.MetadataType) -> bool:
    """Idempotently declare ``prop`` on ``dataset``.

    Returns True if the property was newly added (and the dataset settings
    pushed to the server), False if it was already present.
    """
    existing = dataset.settings.metadata[prop.name]
    if existing is not None:
        return False
    dataset.settings.add(prop)
    dataset.settings.update()
    logger.info("Declared metadata property %r on dataset %s", prop.name, dataset.name)
    return True


def build_metadata_upsert(
    record: rg.Record,
    updates: Mapping[str, object],
    *,
    remove_keys: Iterable[str] = (),
) -> rg.Record | None:
    """Merge ``updates`` into ``record.metadata`` and return an upsert Record.

    Returns ``None`` when the merge produces no change (idempotent no-op).
    Mutates ``record``'s metadata in place and returns it, so its fields (and
    suggestions) ride along in the upsert payload: Argilla v2.8.0 rejects a
    field-less record with 422 "fields cannot be empty" because the required
    text fields must be present, so ``id`` + metadata alone is not a valid
    upsert.

    Callers batch the returned Records into a single ``dataset.records.log``
    call per dataset to amortise the round-trip.
    """
    current = dict(record.metadata)
    merged = dict(current)
    merged.update(updates)
    for key in remove_keys:
        merged.pop(key, None)
    if merged == current:
        return None
    for key, value in updates.items():
        record.metadata[key] = value
    for key in remove_keys:
        record.metadata.pop(key, None)
    return record
