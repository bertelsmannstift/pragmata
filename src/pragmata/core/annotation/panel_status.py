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
from pragmata.core.annotation.export_fetcher import TERMINAL_STATUSES, resolve_task_purposes
from pragmata.core.annotation.metadata_ops import build_metadata_upsert, ensure_metadata_property
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)

NEEDS_COMPLETION_KEY = "needs_completion"
NEEDS_COMPLETION_VALUE = "true"


def _has_needs_completion_tag(record: rg.Record) -> bool:
    """Defensive equality check for the needs_completion tag.

    Argilla TermsMetadataProperty may round-trip as the bare value, a
    1-element list, or a normalised-case string. Treat anything string-equal
    to NEEDS_COMPLETION_VALUE (after str-coercion + lowercase) as tagged, so
    the idempotency check survives SDK encoding differences.
    """
    raw = record.metadata.get(NEEDS_COMPLETION_KEY)
    if raw is None:
        return False
    if isinstance(raw, (list, tuple)):
        return any(str(v).strip().lower() == NEEDS_COMPLETION_VALUE for v in raw)
    return str(raw).strip().lower() == NEEDS_COMPLETION_VALUE


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


@dataclass(frozen=True)
class _CollectedRecords:
    """Internal: output of one walk across prod + cal retrieval datasets."""

    records: list[_ChunkRecord]
    n_orphans: int
    headline: HeadlineTotals
    workspace_name: str | None


def _collect_records(client: rg.Argilla, settings: AnnotationSettings) -> _CollectedRecords:
    """Single walk across prod + cal retrieval datasets."""
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
            # Single pass over responses: terminal-presence + submitted-count.
            has_terminal = False
            n_submitted = 0
            for r in record.responses or []:
                if r.status == "submitted":
                    n_submitted += 1
                    has_terminal = True
                elif r.status in TERMINAL_STATUSES:
                    has_terminal = True
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
    return _CollectedRecords(
        records=records,
        n_orphans=n_orphans,
        headline=HeadlineTotals(**totals),
        workspace_name=workspace_name,
    )


def _resolve_min_submitted(settings: AnnotationSettings, workspace_name: str) -> dict[bool, int]:
    """Per-purpose min_submitted thresholds for retrieval, computed once per report.

    Returns ``{False: prod_min, True: cal_min}``. Cal defaults to prod when
    the topology declares no calibration min (the workspace just doesn't run
    calibration for retrieval).
    """
    resolved = settings.resolved_task(workspace_name, Task.RETRIEVAL)
    prod = resolved.production_min_submitted
    return {False: prod, True: resolved.calibration_min_submitted or prod}


@dataclass(frozen=True)
class _PanelFacts:
    """Derived facts for one panel: shared by status and tag-incomplete passes.

    Computed once per (record_uuid, group) so the two consumers cannot drift
    on the panel_complete predicate or the K-source semantics.
    """

    record_uuid: str
    group: list[_ChunkRecord]
    chunk_ids_terminal: set[str]
    chunk_ids_seen: set[str]
    k_records: int  # distinct chunk_ids in this panel (live K)
    k_metadata: int  # n_retrieved_chunks metadata (max if records disagree; 0 if absent)
    panel_complete: bool  # k_records > 0 AND every chunk has a terminal response


def _group_by_uuid(records: list[_ChunkRecord]) -> dict[str, list[_ChunkRecord]]:
    groups: dict[str, list[_ChunkRecord]] = {}
    for rec in records:
        groups.setdefault(rec.record_uuid, []).append(rec)
    return groups


def _panel_facts(uuid: str, group: list[_ChunkRecord]) -> _PanelFacts:
    """Aggregate one panel's facts. Logs inconsistent-K-across-records as a warning."""
    chunk_ids_seen = {r.chunk_id for r in group}
    chunk_ids_terminal = {r.chunk_id for r in group if r.has_terminal}
    k_values = {r.n_retrieved_chunks_metadata for r in group if r.n_retrieved_chunks_metadata > 0}
    if len(k_values) > 1:
        logger.warning(
            "record_uuid=%s: inconsistent n_retrieved_chunks across records: %s - using max",
            uuid,
            sorted(k_values),
        )
    k_metadata = max(k_values, default=0)
    k_records = len(chunk_ids_seen)
    return _PanelFacts(
        record_uuid=uuid,
        group=group,
        chunk_ids_terminal=chunk_ids_terminal,
        chunk_ids_seen=chunk_ids_seen,
        k_records=k_records,
        k_metadata=k_metadata,
        panel_complete=k_records > 0 and len(chunk_ids_terminal) == k_records,
    )


def _build_report(collected: _CollectedRecords, settings: AnnotationSettings) -> StatusReport:
    """Build StatusReport from already-collected records. Pure aggregation."""
    panels: dict[str, PanelStatus] = {}
    n_complete = 0
    n_distribution_satisfied = 0
    n_integrity_warnings = 0
    n_panels_unknown_k = 0
    # Hoisted: settings.resolved_task() returns the same dict for every
    # record in the report, so look it up once instead of per-record.
    # workspace_name is None only when no records were collected, in which
    # case the inner loop never runs - the empty dict is safe.
    thresholds: dict[bool, int] = (
        _resolve_min_submitted(settings, collected.workspace_name) if collected.workspace_name else {}
    )
    for uuid, group in _group_by_uuid(collected.records).items():
        facts = _panel_facts(uuid, group)
        # Distribution: sum submitted responses PER chunk_id (so duplicate
        # chunk-records for one chunk_id don't each get checked separately
        # against the threshold). If a chunk appears in both prod and cal
        # (rare), require the higher threshold.
        submitted_by_chunk: dict[str, int] = {}
        threshold_by_chunk: dict[str, int] = {}
        for rec in group:
            submitted_by_chunk[rec.chunk_id] = submitted_by_chunk.get(rec.chunk_id, 0) + rec.n_submitted_responses
            t = thresholds.get(rec.calibration, 1)
            threshold_by_chunk[rec.chunk_id] = max(threshold_by_chunk.get(rec.chunk_id, 0), t)
        distribution_satisfied = all(n >= threshold_by_chunk[cid] for cid, n in submitted_by_chunk.items())
        integrity_ok = facts.k_metadata == 0 or facts.k_metadata == facts.k_records
        if not integrity_ok:
            logger.warning(
                "record_uuid=%s: integrity warning - %d records but n_retrieved_chunks metadata=%d",
                uuid,
                facts.k_records,
                facts.k_metadata,
            )
            n_integrity_warnings += 1
        if facts.k_metadata == 0:
            n_panels_unknown_k += 1
        if facts.panel_complete:
            n_complete += 1
        if distribution_satisfied:
            n_distribution_satisfied += 1
        panels[uuid] = PanelStatus(
            record_uuid=uuid,
            k_records=facts.k_records,
            k_metadata=facts.k_metadata,
            n_terminal=len(facts.chunk_ids_terminal),
            panel_complete=facts.panel_complete,
            distribution_satisfied=distribution_satisfied,
            integrity_ok=integrity_ok,
        )

    if collected.n_orphans:
        logger.warning(
            "panel status: %d retrieval record(s) skipped (empty record_uuid metadata)",
            collected.n_orphans,
        )
    if panels and n_panels_unknown_k == len(panels):
        # Most likely cause: the n_retrieved_chunks backfill has not been run
        # against this dataset yet, so panel_complete reads as False across
        # the board for the WRONG reason. Surface this so operators don't
        # interpret 0% as "annotators haven't started".
        logger.warning(
            "panel status: ALL %d panel(s) have unknown K (n_retrieved_chunks metadata absent) - "
            "run scripts/backfill_n_retrieved_chunks.py to populate it.",
            len(panels),
        )

    return StatusReport(
        panels=panels,
        headline=collected.headline,
        n_panels=len(panels),
        n_complete=n_complete,
        n_distribution_satisfied=n_distribution_satisfied,
        n_integrity_warnings=n_integrity_warnings,
        n_orphans_skipped=collected.n_orphans,
    )


def compute_panel_status(client: rg.Argilla, settings: AnnotationSettings) -> StatusReport:
    """Compute live per-panel status across prod + cal retrieval datasets.

    Pure read; safe to invoke against live datasets without side effects.
    """
    return _build_report(_collect_records(client, settings), settings)


@dataclass(frozen=True)
class TagResult:
    """Counts from one ``tag_incomplete_chunks`` pass."""

    n_tagged: int  # chunks newly stamped with needs_completion
    n_cleared: int  # chunks where the stale tag was removed
    n_already_tagged: int  # already had the tag and still need it (no-op)


def _apply_tags(collected: _CollectedRecords) -> TagResult:
    """Apply needs_completion tags + clears using already-collected records.

    Batches one ``dataset.records.log`` call per dataset (instead of per
    record), so a panel with N incomplete chunks costs one round-trip per
    dataset rather than N.
    """
    # Declare the property idempotently on every dataset we'll touch.
    datasets_by_name: dict[str, rg.Dataset] = {}
    for rec in collected.records:
        datasets_by_name.setdefault(rec.dataset.name, rec.dataset)
    for dataset in datasets_by_name.values():
        ensure_metadata_property(
            dataset,
            rg.TermsMetadataProperty(NEEDS_COMPLETION_KEY, visible_for_annotators=True),
        )

    # Collect upsert payloads per dataset for one batched log() call each.
    batched: dict[str, list[rg.Record]] = {name: [] for name in datasets_by_name}
    n_tagged = 0
    n_cleared = 0
    n_already_tagged = 0
    for uuid, group in _group_by_uuid(collected.records).items():
        facts = _panel_facts(uuid, group)
        for rec in group:
            already_has_tag = _has_needs_completion_tag(rec.record)
            should_have_tag = (not facts.panel_complete) and (not rec.has_terminal)
            if should_have_tag and already_has_tag:
                n_already_tagged += 1
                continue
            if should_have_tag:
                upsert = build_metadata_upsert(rec.record, {NEEDS_COMPLETION_KEY: NEEDS_COMPLETION_VALUE})
                if upsert is not None:
                    batched[rec.dataset.name].append(upsert)
                    n_tagged += 1
            elif already_has_tag:
                upsert = build_metadata_upsert(rec.record, {}, remove_keys=[NEEDS_COMPLETION_KEY])
                if upsert is not None:
                    batched[rec.dataset.name].append(upsert)
                    n_cleared += 1

    for name, payloads in batched.items():
        if payloads:
            datasets_by_name[name].records.log(payloads)

    logger.info(
        "tag_incomplete_chunks: tagged=%d cleared=%d already_tagged=%d (panels=%d, datasets=%d)",
        n_tagged,
        n_cleared,
        n_already_tagged,
        len({rec.record_uuid for rec in collected.records}),
        len(datasets_by_name),
    )
    return TagResult(n_tagged=n_tagged, n_cleared=n_cleared, n_already_tagged=n_already_tagged)


def tag_incomplete_chunks(client: rg.Argilla, settings: AnnotationSettings) -> TagResult:
    """Stamp / clear ``needs_completion`` advisory tags on retrieval chunk-records.

    Tag predicate: panel is INCOMPLETE and this chunk is UNRESOLVED (no
    terminal response). Cleared on resolved chunks and on chunks whose panel
    has since completed. Idempotent: every run re-derives the set.

    Self-contained convenience wrapper around ``_apply_tags``. Callers that
    have already run ``_collect_records`` (e.g. ``report_status`` after
    ``compute_panel_status``) should call ``_apply_tags`` directly with the
    shared collection to avoid a second walk.
    """
    return _apply_tags(_collect_records(client, settings))
