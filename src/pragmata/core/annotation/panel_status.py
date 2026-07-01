"""Live read-only panel status across all retrieval datasets.

For each retrieval panel (one query = ``(workspace, record_uuid)``, K chunks)
reports two distinct notions of "complete":

- ``panel_complete`` (metric-facing, STRICT) = all K chunks have at least
  one SUBMITTED response. Discards are abstentions, not judgements, so they
  don't count toward "ready for eval scoring" - see completeness.py.
- ``distribution_satisfied`` (operational) = every chunk's submitted-response
  count is >= the dataset's Argilla ``min_submitted`` threshold (typically 1
  for production, 3 for calibration). A panel can be metric-complete but
  distribution-unsatisfied, or vice versa - don't conflate them.

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

Pure read; safe to invoke against live datasets without side effects. The
optional ``--tag-partial-panels`` advisory write that stamps records for
annotator UI filtering ships in a follow-up PR (separates the read path from
any Argilla mutation surface).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import argilla as rg

from pragmata.core.annotation.export_fetcher import TERMINAL_STATUSES

logger = logging.getLogger(__name__)

RETRIEVAL_TASK = "retrieval"


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
    distribution_satisfied: bool  # every chunk meets its dataset min_submitted
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

    panels: dict[tuple[str, str], PanelStatus]
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
                elif r.status in TERMINAL_STATUSES:
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
    n_distribution_satisfied = 0
    n_integrity_warnings = 0
    n_panels_unknown_k = 0
    for (ws_name, uuid), group in _group_by_panel(collected.records).items():
        facts = _panel_facts(uuid, group)
        # Distribution: sum submitted responses PER chunk_id (so duplicate
        # chunk-records for one chunk_id don't each get checked separately
        # against the threshold). Each chunk-record carries its own dataset's
        # min_submitted; if a chunk spans prod+cal (rare), require the higher.
        submitted_by_chunk: dict[str, int] = {}
        threshold_by_chunk: dict[str, int] = {}
        for rec in group:
            submitted_by_chunk[rec.chunk_id] = submitted_by_chunk.get(rec.chunk_id, 0) + rec.n_submitted_responses
            threshold_by_chunk[rec.chunk_id] = max(threshold_by_chunk.get(rec.chunk_id, 0), rec.min_submitted)
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
        panels[(ws_name, uuid)] = PanelStatus(
            workspace=ws_name,
            record_uuid=uuid,
            k_records=facts.k_records,
            k_metadata=facts.k_metadata,
            n_terminal=len(facts.chunk_ids_terminal),
            n_submitted=len(facts.chunk_ids_submitted),
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


def compute_panel_status(
    client: rg.Argilla, *, workspace: str | None = None, task: str | None = RETRIEVAL_TASK
) -> StatusReport:
    """Compute live per-panel status across the selected retrieval datasets.

    Config-free: walks every matching dataset (all workspaces by default).
    Pure read; safe to invoke against live datasets without side effects.
    """
    return _build_report(_collect_records(client, workspace=workspace, task=task))
