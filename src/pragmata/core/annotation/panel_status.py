"""Live read-only panel status across prod + cal retrieval datasets.

For each ``record_uuid`` (one query, K chunks) reports two distinct notions
of "complete":

- ``panel_complete`` (metric-facing) = all K chunks have at least one terminal
  (submitted-or-discarded) response. This is what eval scorers care about.
- ``distribution_satisfied`` (operational) = every chunk's submitted-response
  count is >= the per-chunk Argilla ``min_submitted`` threshold (1 for prod,
  3 for cal). A panel can be metric-complete but distribution-unsatisfied,
  or vice versa - don't conflate them.

Headline totals come from ``dataset.progress()`` aggregated across the
walked datasets.

K is computed by COUNTING distinct chunk-records per record_uuid (every
chunk became a record at import; records are never deleted). This is
distinct from the export-time completeness which sources K from the
``n_retrieved_chunks`` metadata; the live K is the ground truth pre-backfill
and the metadata is cross-checked for integrity.

``tag_incomplete_chunks`` is an optional advisory write that stamps a
``needs_completion`` TermsMetadataProperty (visible to annotators) on
chunk-records belonging to incomplete panels that are themselves unresolved,
and idempotently clears the tag from resolved chunks or panels that have
since completed. Never tags a discarded chunk (discarded is resolved).
"""

import logging
from dataclasses import dataclass

import argilla as rg

from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.annotation.export_fetcher import resolve_task_purposes
from pragmata.core.annotation.metadata_ops import (
    ensure_metadata_property,
    upsert_record_metadata,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"submitted", "discarded"})
NEEDS_COMPLETION_KEY = "needs_completion"
NEEDS_COMPLETION_VALUE = "true"


@dataclass(frozen=True)
class PanelStatus:
    """Live status facts for one retrieval panel (one ``record_uuid``)."""

    record_uuid: str
    k_records: int  # distinct chunk-records seen (live K)
    k_metadata: int  # n_retrieved_chunks metadata (0 if missing)
    n_terminal: int  # distinct chunks with >=1 terminal response
    panel_complete: bool  # k_records > 0 and n_terminal == k_records
    distribution_satisfied: bool  # every chunk meets its per-purpose min_submitted
    integrity_ok: bool  # k_records == k_metadata (when metadata present)


@dataclass(frozen=True)
class HeadlineTotals:
    """Aggregate counts from ``dataset.progress()`` across walked datasets."""

    total: int
    completed: int
    pending: int


@dataclass(frozen=True)
class StatusReport:
    """Live per-panel status + headline aggregates."""

    panels: dict[str, PanelStatus]
    headline: HeadlineTotals
    n_panels: int
    n_complete: int
    n_distribution_satisfied: int
    n_integrity_warnings: int
    n_orphans_skipped: int


@dataclass
class _ChunkRecord:
    """Internal: live record snapshot for status + tag passes."""

    record: rg.Record  # argilla record handle (needed for tag write)
    dataset: rg.Dataset  # owning dataset (needed for tag write)
    record_uuid: str
    chunk_id: str
    calibration: bool
    n_retrieved_chunks_metadata: int
    has_terminal: bool
    n_submitted_responses: int


def _collect_records(
    client: rg.Argilla, settings: AnnotationSettings
) -> tuple[list[_ChunkRecord], int, HeadlineTotals]:
    """Single walk across prod + cal retrieval datasets. Returns (records, n_orphans, headline)."""
    workspace_name, purposes = resolve_task_purposes(settings, Task.RETRIEVAL)
    records: list[_ChunkRecord] = []
    n_orphans = 0
    totals = {"total": 0, "completed": 0, "pending": 0}
    for calibration in purposes:
        ds_name = dataset_name(Task.RETRIEVAL, calibration=calibration, dataset_id=settings.dataset_id)
        dataset = client.datasets(ds_name, workspace=workspace_name)
        if dataset is None:
            continue
        progress = dataset.progress()
        for key in totals:
            totals[key] += int(progress.get(key, 0) or 0)
        for record in dataset.records(with_responses=True):
            record_uuid: str = record.metadata.get("record_uuid", "")
            if not record_uuid:
                n_orphans += 1
                continue
            chunk_id: str = record.metadata.get("chunk_id", "")
            k_meta = int(record.metadata.get("n_retrieved_chunks") or 0)
            responses = record.responses or []
            has_terminal = any(r.status in _TERMINAL_STATUSES for r in responses)
            n_submitted = sum(1 for r in responses if r.status == "submitted")
            records.append(
                _ChunkRecord(
                    record=record,
                    dataset=dataset,
                    record_uuid=record_uuid,
                    chunk_id=chunk_id,
                    calibration=calibration,
                    n_retrieved_chunks_metadata=k_meta,
                    has_terminal=has_terminal,
                    n_submitted_responses=n_submitted,
                )
            )
    return records, n_orphans, HeadlineTotals(**totals)


def _min_submitted(settings: AnnotationSettings, workspace_name: str | None, calibration: bool) -> int:
    """Per-purpose min_submitted threshold for retrieval. Defaults conservatively to 1 if unresolved."""
    if workspace_name is None:
        return 1
    resolved = settings.resolved_task(workspace_name, Task.RETRIEVAL)
    if calibration:
        return resolved.calibration_min_submitted or 1
    return resolved.production_min_submitted


def compute_panel_status(client: rg.Argilla, settings: AnnotationSettings) -> StatusReport:
    """Compute live per-panel status across prod + cal retrieval datasets.

    Pure read; safe to invoke against live datasets without side effects.
    """
    records, n_orphans_skipped, headline = _collect_records(client, settings)
    workspace_name, _ = resolve_task_purposes(settings, Task.RETRIEVAL)

    groups: dict[str, list[_ChunkRecord]] = {}
    for rec in records:
        groups.setdefault(rec.record_uuid, []).append(rec)

    panels: dict[str, PanelStatus] = {}
    n_complete = 0
    n_distribution_satisfied = 0
    n_integrity_warnings = 0
    for uuid, group in groups.items():
        chunk_ids_seen = {r.chunk_id for r in group}
        chunk_ids_terminal = {r.chunk_id for r in group if r.has_terminal}
        k_records = len(chunk_ids_seen)
        k_values = {r.n_retrieved_chunks_metadata for r in group if r.n_retrieved_chunks_metadata > 0}
        if len(k_values) > 1:
            logger.warning(
                "record_uuid=%s: inconsistent n_retrieved_chunks across records: %s - using max",
                uuid,
                sorted(k_values),
            )
        k_metadata = max(k_values, default=0)
        n_terminal = len(chunk_ids_terminal)
        panel_complete = k_records > 0 and n_terminal == k_records
        distribution_satisfied = all(
            rec.n_submitted_responses >= _min_submitted(settings, workspace_name, rec.calibration) for rec in group
        )
        integrity_ok = k_metadata == 0 or k_metadata == k_records
        if not integrity_ok:
            logger.warning(
                "record_uuid=%s: integrity warning - %d records but n_retrieved_chunks metadata=%d",
                uuid,
                k_records,
                k_metadata,
            )
            n_integrity_warnings += 1
        if panel_complete:
            n_complete += 1
        if distribution_satisfied:
            n_distribution_satisfied += 1
        panels[uuid] = PanelStatus(
            record_uuid=uuid,
            k_records=k_records,
            k_metadata=k_metadata,
            n_terminal=n_terminal,
            panel_complete=panel_complete,
            distribution_satisfied=distribution_satisfied,
            integrity_ok=integrity_ok,
        )

    if n_orphans_skipped:
        logger.warning(
            "panel status: %d retrieval record(s) skipped (empty record_uuid metadata)",
            n_orphans_skipped,
        )

    return StatusReport(
        panels=panels,
        headline=headline,
        n_panels=len(panels),
        n_complete=n_complete,
        n_distribution_satisfied=n_distribution_satisfied,
        n_integrity_warnings=n_integrity_warnings,
        n_orphans_skipped=n_orphans_skipped,
    )


@dataclass(frozen=True)
class TagResult:
    """Counts from one ``tag_incomplete_chunks`` pass."""

    n_tagged: int  # chunks newly stamped with needs_completion
    n_cleared: int  # chunks where the stale tag was removed
    n_already_tagged: int  # already had the tag and still need it (no-op)


def tag_incomplete_chunks(client: rg.Argilla, settings: AnnotationSettings) -> TagResult:
    """Stamp / clear ``needs_completion`` advisory tags on retrieval chunk-records.

    Tag predicate: panel is INCOMPLETE and this chunk is UNRESOLVED (no
    terminal response). Cleared on resolved chunks and on chunks whose panel
    has since completed. Idempotent: every run re-derives the set.

    Declares the property on each retrieval dataset on first encounter
    (``visible_for_annotators=True`` so annotators can filter the UI to
    only these records).
    """
    records, _, _ = _collect_records(client, settings)
    datasets_seen: set[str] = set()
    for rec in records:
        if rec.dataset.name in datasets_seen:
            continue
        ensure_metadata_property(
            rec.dataset,
            rg.TermsMetadataProperty(NEEDS_COMPLETION_KEY, visible_for_annotators=True),
        )
        datasets_seen.add(rec.dataset.name)

    by_uuid: dict[str, list[_ChunkRecord]] = {}
    for rec in records:
        by_uuid.setdefault(rec.record_uuid, []).append(rec)

    n_tagged = 0
    n_cleared = 0
    n_already_tagged = 0
    for uuid, group in by_uuid.items():
        chunk_ids_terminal = {r.chunk_id for r in group if r.has_terminal}
        k_records = len({r.chunk_id for r in group})
        panel_complete = k_records > 0 and len(chunk_ids_terminal) == k_records
        for rec in group:
            already_has_tag = rec.record.metadata.get(NEEDS_COMPLETION_KEY) == NEEDS_COMPLETION_VALUE
            should_have_tag = (not panel_complete) and (not rec.has_terminal)
            if should_have_tag and already_has_tag:
                n_already_tagged += 1
                continue
            if should_have_tag and not already_has_tag:
                upsert_record_metadata(rec.dataset, rec.record, {NEEDS_COMPLETION_KEY: NEEDS_COMPLETION_VALUE})
                n_tagged += 1
                continue
            if not should_have_tag and already_has_tag:
                upsert_record_metadata(rec.dataset, rec.record, {}, remove_keys=[NEEDS_COMPLETION_KEY])
                n_cleared += 1

    logger.info(
        "tag_incomplete_chunks: tagged=%d cleared=%d already_tagged=%d (uuid=%d)",
        n_tagged,
        n_cleared,
        n_already_tagged,
        len(by_uuid),
    )
    return TagResult(n_tagged=n_tagged, n_cleared=n_cleared, n_already_tagged=n_already_tagged)
