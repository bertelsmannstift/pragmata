"""Live read-only panel status across all retrieval datasets.

For each retrieval panel (one query = ``(workspace, record_uuid)``, K chunks)
reports two distinct notions of "complete":

- ``panel_complete`` (metric-facing, STRICT) = all K chunks have at least
  one SUBMITTED response. Discards are abstentions, not judgements, so they
  don't count toward "ready for eval scoring" - see completeness.py.
- ``overlap_satisfied`` (operational) = every chunk's submitted-response count
  is >= the dataset's annotator overlap target (Argilla ``min_submitted``;
  typically 1 for production, 3 for calibration - the term used throughout
  the daily report's IAA tables). This is the readiness signal for
  Krippendorff's alpha, distinct from merely having a first opinion: a panel
  can be complete but overlap-unsatisfied (e.g. 1-of-3 calibration votes in).

The walk is CONFIG-FREE: it enumerates the live datasets and selects retrieval
ones by name prefix (``retrieval`` / ``retrieval_*``), so it covers every
workspace/domain in one pass rather than a single configured workspace. Panels
are keyed by ``(workspace, record_uuid)``: retrieval is split across one
workspace per domain, so the same ``record_uuid`` can recur across domains (it
also links a query across the grounding/generation workspaces). Keying by the
bare uuid would fuse those distinct panels once the walk is multi-domain.

``min_submitted`` is read from each dataset's live Argilla settings
(``dataset.settings.distribution.min_submitted``), not local config - for a
live status report, what the server enforces is the source of truth.

Headline totals come from ``dataset.progress()`` aggregated across the walked
datasets.

K is computed by COUNTING distinct chunk-records per panel (every chunk became
a record at import; records are never deleted). This is distinct from the
export-time completeness which sources K from the ``n_retrieved_chunks``
metadata; the live K is the ground truth pre-backfill and the metadata is
cross-checked for integrity.

The read path (``compute_panel_status``) is side-effect free. The optional
``--tag-partial-panels`` advisory write (``_apply_tags``, reached via
``report_status``) stamps partial panels' unresolved chunks for annotator UI
filtering, sharing this same single walk - it is the only Argilla mutation
surface here.
"""

import logging
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, replace

import argilla as rg

from pragmata.core.annotation.metadata_ops import build_metadata_upsert, ensure_metadata_property

logger = logging.getLogger(__name__)

# Terminal response statuses = a judgement was recorded (vs pending/draft).
# Mirrors the inline ``{"submitted", "discarded"}`` in export_fetcher; kept
# local so the status read path carries no import-time coupling to export.
_TERMINAL_STATUSES = frozenset({"submitted", "discarded"})

RETRIEVAL_TASK = "retrieval"
_TASK_ORDER = {"retrieval": 0, "grounding": 1, "generation": 2}
# Changing either of these orphans old key/value metadata already written to
# Argilla records - nothing here renames or removes it. Manual cleanup of the
# stale metadata would be needed on the live datasets.
NEEDS_COMPLETION_KEY = "needs_completion"
NEEDS_COMPLETION_VALUE = "true"


def _task_of(dataset_name: str) -> str:
    """Task label for a dataset = its name prefix before the first underscore."""
    return dataset_name.split("_", 1)[0]


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


def _select_datasets(client: rg.Argilla, workspace: str | None, task: str | None) -> Iterator[rg.Dataset]:
    """Config-free dataset selection: iterate the live server, filter by name.

    ``task`` matches a dataset-name prefix (e.g. ``retrieval`` matches
    ``retrieval_production`` / ``retrieval_calibration``); ``workspace`` is an
    exact workspace-name filter. ``None`` means no filter on that axis. No
    ``AnnotationSettings`` needed, so status covers every workspace at once.
    """
    for ds in client.datasets:
        if workspace is not None and ds.workspace.name != workspace:
            continue
        if task is not None and ds.name != task and not ds.name.startswith(f"{task}_"):
            continue
        yield ds


def _dataset_min_submitted(dataset: rg.Dataset) -> int:
    """Read the dataset's Argilla ``min_submitted`` overlap threshold, live.

    Sourced from ``dataset.settings.distribution`` (the SDK never leaves this
    None - it defaults to ``min_submitted=1``), so status needs no local
    topology config to compute distribution-satisfaction.
    """
    return int(getattr(dataset.settings.distribution, "min_submitted", 1))


@dataclass(frozen=True)
class PanelStatus:
    """Live status facts for one retrieval panel (one ``(workspace, record_uuid)``)."""

    workspace: str
    record_uuid: str
    k_records: int  # distinct chunk-records seen (live K)
    k_metadata: int  # n_retrieved_chunks metadata (0 if missing)
    n_terminal: int  # distinct chunks with >=1 terminal response (submitted OR discarded)
    n_submitted: int  # distinct chunks with >=1 submitted response (used by panel_complete)
    panel_complete: bool  # STRICT: k_records > 0 and n_submitted == k_records
    overlap_satisfied: bool  # every chunk meets its dataset's annotator overlap (min_submitted)
    integrity_ok: bool  # k_records == k_metadata (when metadata present)


@dataclass(frozen=True)
class HeadlineTotals:
    """Aggregate counts from ``dataset.progress()`` across walked datasets."""

    total: int
    completed: int
    pending: int


@dataclass(frozen=True)
class ProgressRow:
    """Record-level progress for one grouping (a task, a workspace, or a dataset)."""

    label: str  # display label: task name, workspace name, or "workspace/dataset"
    task: str  # the task this row belongs to (equals label for by-task rows)
    total: int
    completed: int  # records that reached their Argilla min_submitted
    pending: int


@dataclass(frozen=True)
class ProgressReport:
    """All-task record progress from ``dataset.progress()``, grouped three ways."""

    grand: HeadlineTotals
    by_task: list[ProgressRow]
    by_workspace: list[ProgressRow]
    by_dataset: list[ProgressRow]


@dataclass(frozen=True)
class TagResult:
    """Counts from one ``--tag-partial-panels`` pass."""

    n_tagged: int  # chunks newly stamped with needs_completion
    n_cleared: int  # chunks where the stale tag was removed
    n_already_tagged: int  # already had the tag and still need it (no-op)


@dataclass(frozen=True)
class StatusReport:
    """Live per-panel status + headline aggregates.

    ``progress`` (all-task record counts) is attached by ``report_status``; the
    panel fields below are retrieval-only. ``tag_result`` is None unless
    ``--tag-partial-panels`` ran in the same pass.
    """

    panels: dict[tuple[str, str], PanelStatus]
    headline: HeadlineTotals
    n_panels: int
    n_complete: int
    n_overlap_satisfied: int
    n_integrity_warnings: int
    n_orphans_skipped: int
    progress: "ProgressReport | None" = None
    tag_result: "TagResult | None" = None

    def with_progress(self, progress: ProgressReport) -> "StatusReport":
        """Return a copy with the all-task ``progress`` summary attached."""
        return replace(self, progress=progress)

    def with_tag_result(self, tag_result: TagResult) -> "StatusReport":
        """Return a copy with ``tag_result`` set (the dataclass is frozen)."""
        return replace(self, tag_result=tag_result)


@dataclass
class _ChunkRecord:
    """Internal: live record snapshot for status + tag passes."""

    record: rg.Record  # argilla record handle (needed for tag write)
    dataset: rg.Dataset  # owning dataset (needed for tag write)
    workspace: str  # owning workspace name (panel grouping key)
    record_uuid: str
    chunk_id: str
    min_submitted: int  # this dataset's Argilla overlap threshold (live-sourced)
    n_retrieved_chunks_metadata: int
    has_terminal: bool  # >=1 response in {submitted, discarded}
    has_submitted: bool  # >=1 response with status == submitted (subset of has_terminal)
    n_submitted_responses: int


@dataclass(frozen=True)
class _CollectedRecords:
    """Internal: output of one config-free walk across the selected datasets."""

    records: list[_ChunkRecord]
    n_orphans: int
    headline: HeadlineTotals


def _collect_records(
    client: rg.Argilla, *, workspace: str | None = None, task: str | None = RETRIEVAL_TASK
) -> _CollectedRecords:
    """Single config-free walk across the selected retrieval datasets."""
    records: list[_ChunkRecord] = []
    n_orphans = 0
    totals = {"total": 0, "completed": 0, "pending": 0}
    for dataset in _select_datasets(client, workspace, task):
        ws_name = dataset.workspace.name
        min_submitted = _dataset_min_submitted(dataset)
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
            # Single pass over responses: terminal-presence + submitted-count + has-submitted.
            has_terminal = False
            has_submitted = False
            n_submitted = 0
            for r in record.responses or []:
                if r.status == "submitted":
                    n_submitted += 1
                    has_submitted = True
                    has_terminal = True
                elif r.status in _TERMINAL_STATUSES:
                    has_terminal = True
            records.append(
                _ChunkRecord(
                    record=record,
                    dataset=dataset,
                    workspace=ws_name,
                    record_uuid=record_uuid,
                    chunk_id=chunk_id,
                    min_submitted=min_submitted,
                    n_retrieved_chunks_metadata=k_meta,
                    has_terminal=has_terminal,
                    has_submitted=has_submitted,
                    n_submitted_responses=n_submitted,
                )
            )
    return _CollectedRecords(records=records, n_orphans=n_orphans, headline=HeadlineTotals(**totals))


@dataclass(frozen=True)
class _PanelFacts:
    """Derived facts for one panel: shared by status and tag passes.

    Computed once per panel so the two consumers cannot drift on the
    panel_complete predicate or the K-source semantics.
    """

    record_uuid: str
    group: list[_ChunkRecord]
    chunk_ids_terminal: set[str]
    chunk_ids_submitted: set[str]
    chunk_ids_seen: set[str]
    k_records: int  # distinct chunk_ids in this panel (live K)
    k_metadata: int  # n_retrieved_chunks metadata (max if records disagree; 0 if absent)
    panel_complete: bool  # STRICT: k_records > 0 AND every chunk has a SUBMITTED response


def _group_by_panel(records: list[_ChunkRecord]) -> dict[tuple[str, str], list[_ChunkRecord]]:
    """Group by ``(workspace, record_uuid)`` - the panel identity across datasets.

    Keyed by workspace too, not the bare uuid: retrieval spans one workspace
    per domain, so the same ``record_uuid`` recurs across domains (and, more
    broadly, across a query's grounding/generation workspaces). A bare-uuid key
    would fuse those distinct panels once the walk is multi-domain.
    """
    groups: dict[tuple[str, str], list[_ChunkRecord]] = {}
    for rec in records:
        groups.setdefault((rec.workspace, rec.record_uuid), []).append(rec)
    return groups


def _panel_facts(uuid: str, group: list[_ChunkRecord]) -> _PanelFacts:
    """Aggregate one panel's facts. Logs inconsistent-K-across-records as a warning."""
    chunk_ids_seen = {r.chunk_id for r in group}
    chunk_ids_terminal = {r.chunk_id for r in group if r.has_terminal}
    chunk_ids_submitted = {r.chunk_id for r in group if r.has_submitted}
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
        chunk_ids_submitted=chunk_ids_submitted,
        chunk_ids_seen=chunk_ids_seen,
        k_records=k_records,
        k_metadata=k_metadata,
        # STRICT default - see module docstring.
        panel_complete=k_records > 0 and len(chunk_ids_submitted) == k_records,
    )


def _build_report(collected: _CollectedRecords) -> StatusReport:
    """Build StatusReport from already-collected records. Pure aggregation."""
    panels: dict[tuple[str, str], PanelStatus] = {}
    n_complete = 0
    n_overlap_satisfied = 0
    n_integrity_warnings = 0
    n_panels_unknown_k = 0
    for (ws_name, uuid), group in _group_by_panel(collected.records).items():
        facts = _panel_facts(uuid, group)
        # Overlap: sum submitted responses PER chunk_id (so duplicate
        # chunk-records for one chunk_id don't each get checked separately
        # against the threshold). Each chunk-record carries its own dataset's
        # min_submitted; if a chunk spans prod+cal (rare), require the higher.
        submitted_by_chunk: dict[str, int] = {}
        threshold_by_chunk: dict[str, int] = {}
        for rec in group:
            submitted_by_chunk[rec.chunk_id] = submitted_by_chunk.get(rec.chunk_id, 0) + rec.n_submitted_responses
            threshold_by_chunk[rec.chunk_id] = max(threshold_by_chunk.get(rec.chunk_id, 0), rec.min_submitted)
        overlap_satisfied = all(n >= threshold_by_chunk[cid] for cid, n in submitted_by_chunk.items())
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
        if overlap_satisfied:
            n_overlap_satisfied += 1
        panels[(ws_name, uuid)] = PanelStatus(
            workspace=ws_name,
            record_uuid=uuid,
            k_records=facts.k_records,
            k_metadata=facts.k_metadata,
            n_terminal=len(facts.chunk_ids_terminal),
            n_submitted=len(facts.chunk_ids_submitted),
            panel_complete=facts.panel_complete,
            overlap_satisfied=overlap_satisfied,
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
        n_overlap_satisfied=n_overlap_satisfied,
        n_integrity_warnings=n_integrity_warnings,
        n_orphans_skipped=collected.n_orphans,
    )


def compute_panel_status(
    client: rg.Argilla, *, workspace: str | None = None, task: str | None = RETRIEVAL_TASK
) -> StatusReport:
    """Compute live per-panel status across the selected retrieval datasets.

    Config-free: walks every matching dataset (all workspaces by default).
    Pure read; safe to invoke against live datasets without side effects.
    """
    return _build_report(_collect_records(client, workspace=workspace, task=task))


def compute_task_progress(client: rg.Argilla, *, workspace: str | None = None) -> ProgressReport:
    """All-task record progress from ``dataset.progress()``.

    Grouped by task, workspace, and dataset. Config-free and cheap: one
    ``progress()`` call per dataset (no record iteration), covering every
    task/workspace in one pass. Rows are ordered retrieval -> grounding ->
    generation, then by workspace/dataset name.
    """
    grand = {"total": 0, "completed": 0, "pending": 0}
    by_task: dict[str, dict[str, int]] = {}
    by_ws: dict[tuple[str, str], dict[str, int]] = {}
    by_ds: dict[tuple[str, str, str], dict[str, int]] = {}

    def _acc(bucket: dict[str, int], p: dict) -> None:
        for key in bucket:
            bucket[key] += int(p.get(key, 0) or 0)

    def _bucket(store: dict, key: Hashable) -> dict[str, int]:
        return store.setdefault(key, {"total": 0, "completed": 0, "pending": 0})

    for ds in _select_datasets(client, workspace, task=None):
        task = _task_of(ds.name)
        ws_name = ds.workspace.name
        p = dict(ds.progress())
        _acc(grand, p)
        _acc(_bucket(by_task, task), p)
        _acc(_bucket(by_ws, (ws_name, task)), p)
        _acc(_bucket(by_ds, (ws_name, ds.name, task)), p)

    def _row(label: str, task: str, b: dict[str, int]) -> ProgressRow:
        return ProgressRow(label=label, task=task, total=b["total"], completed=b["completed"], pending=b["pending"])

    task_rows = [_row(t, t, b) for t, b in sorted(by_task.items(), key=lambda kv: _TASK_ORDER.get(kv[0], 99))]
    ws_rows = [
        _row(ws, task, b)
        for (ws, task), b in sorted(by_ws.items(), key=lambda kv: (_TASK_ORDER.get(kv[0][1], 99), kv[0][0]))
    ]
    ds_rows = [
        _row(f"{ws}/{name}", task, b)
        for (ws, name, task), b in sorted(
            by_ds.items(), key=lambda kv: (_TASK_ORDER.get(kv[0][2], 99), kv[0][0], kv[0][1])
        )
    ]
    return ProgressReport(grand=HeadlineTotals(**grand), by_task=task_rows, by_workspace=ws_rows, by_dataset=ds_rows)


def _apply_tags(collected: _CollectedRecords) -> TagResult:
    """Stamp / clear the ``needs_completion`` advisory tag on PARTIAL panels.

    Tag predicate: the panel is PARTIAL (at least one chunk has a submitted
    response but NOT all chunks do) AND this chunk is UNRESOLVED (no terminal
    response). The tag is cleared on resolved chunks and on non-partial panels
    (fully-unstarted or complete), so a fully-unstarted panel is never tagged.
    Idempotent: every run re-derives the set.

    Dataset-local and overlap-indifferent: PARTIAL/UNRESOLVED are derived
    from ``has_terminal`` (>=1 response, any status) vs ``k_records`` alone -
    ``min_submitted`` never enters this predicate. So a calibration chunk
    with 1-of-3 submissions already counts as resolved for tagging purposes,
    even though it hasn't hit its overlap target; this tag means "nobody has
    looked at this yet," not "this hasn't reached ``overlap_satisfied``" (see
    the module docstring's PARTIAL/``overlap_satisfied`` distinction).

    Batches one ``dataset.records.log`` per owning dataset. Datasets are keyed
    by ``(workspace, name)`` because the same bare name (``retrieval_production``)
    recurs across domains, so batching by name alone would misroute payloads.
    """
    datasets_by_key: dict[tuple[str, str], rg.Dataset] = {}
    for rec in collected.records:
        datasets_by_key.setdefault((rec.workspace, rec.dataset.name), rec.dataset)
    for dataset in datasets_by_key.values():
        ensure_metadata_property(dataset, rg.TermsMetadataProperty(NEEDS_COMPLETION_KEY, visible_for_annotators=True))

    batched: dict[tuple[str, str], list[rg.Record]] = {}
    n_tagged = n_cleared = n_already = 0
    for (_ws, uuid), group in _group_by_panel(collected.records).items():
        facts = _panel_facts(uuid, group)
        # PARTIAL: some but not all chunks have a submitted response. A panel
        # that is fully-unstarted (0 submitted) or complete is NOT partial.
        panel_partial = 0 < len(facts.chunk_ids_submitted) < facts.k_records
        for rec in group:
            should_have_tag = panel_partial and not rec.has_terminal
            already = _has_needs_completion_tag(rec.record)
            if should_have_tag and already:
                n_already += 1
                continue
            if should_have_tag:
                upsert = build_metadata_upsert(rec.record, {NEEDS_COMPLETION_KEY: NEEDS_COMPLETION_VALUE})
            elif already:
                upsert = build_metadata_upsert(rec.record, {}, remove_keys=[NEEDS_COMPLETION_KEY])
            else:
                continue
            if upsert is None:
                continue
            key = (rec.workspace, rec.dataset.name)
            batched.setdefault(key, []).append(upsert)
            if should_have_tag:
                n_tagged += 1
            else:
                n_cleared += 1

    for key, payloads in batched.items():
        # payloads is never empty: keys are created only on first append.
        datasets_by_key[key].records.log(payloads)

    logger.info(
        "tag_partial_panels: tagged=%d cleared=%d already_tagged=%d (panels=%d, datasets=%d)",
        n_tagged,
        n_cleared,
        n_already,
        len({(r.workspace, r.record_uuid) for r in collected.records}),
        len(datasets_by_key),
    )
    return TagResult(n_tagged=n_tagged, n_cleared=n_cleared, n_already_tagged=n_already)


def tag_partial_panels(
    client: rg.Argilla, *, workspace: str | None = None, task: str | None = RETRIEVAL_TASK
) -> TagResult:
    """Stamp / clear ``needs_completion`` advisory tags on partial retrieval panels.

    Tag predicate: panel is PARTIAL and this chunk is UNRESOLVED (see
    ``_apply_tags``). Config-free; covers every workspace in one pass.

    Self-contained wrapper around ``_apply_tags``. Callers that already ran
    ``_collect_records`` (e.g. ``report_status`` with ``tag_partial_panels=True``)
    should call ``_apply_tags`` directly with the shared collection to avoid a
    second walk.
    """
    return _apply_tags(_collect_records(client, workspace=workspace, task=task))
