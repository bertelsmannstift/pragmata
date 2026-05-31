"""Per-panel retrieval completeness: how many chunks per query have a submitted response.

A "panel" is the set of retrieval chunk-records sharing one ``record_uuid``
(one query, K chunks).

The exported ``panel_complete`` flag uses the STRICT (submitted-only)
definition: a panel is complete iff every one of its K chunks has at least
one SUBMITTED response. This matches what downstream eval scorers actually
need: pragmata's ``DiscardReason`` enum is refusal-only
(``INVALID_OR_UNREALISTIC`` / ``UNCLEAR`` / ``OUTSIDE_REVIEWER_EXPERTISE``),
so a discarded chunk is an abstention — not a judgement — and treating it
as covered would feed unjudged chunks into NDCG@K / precision@K denominators
as if they carried a 0-relevance label.

Three count columns let consumers derive any policy they need:

- ``n_annotated_chunks`` = distinct chunks with >=1 terminal response (submitted OR discarded)
- ``n_submitted_chunks`` = distinct chunks with >=1 submitted response
- ``n_discarded_chunks`` = distinct chunks with >=1 discarded response

Sets aren't disjoint — a chunk where annotator A submitted and annotator B
discarded is in all three. So ``n_submitted + n_discarded >= n_annotated``
(equality when no mixed chunks). A consumer wanting the PERMISSIVE
("terminal counts as covered") policy derives it in one line::

    panel_complete_permissive = (n_annotated_chunks == n_retrieved_chunks)

Computation is INDEPENDENT of the export's ``include_discarded`` flag (which
gates only row emission to the CSV). When invoked from ``run_export`` the
underlying walk is shared with the export's row-emission fetch so we don't
scroll the retrieval datasets twice; standalone callers use the
``compute_completeness(client, settings)`` convenience that does its own walk.
"""

import logging
from dataclasses import dataclass

import argilla as rg

from pragmata.core.annotation.export_fetcher import RetrievalRecordSnapshot, walk_retrieval_records
from pragmata.core.schemas.annotation_export import CompletenessSummary, KBucketStat
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanelCompleteness:
    """Completeness facts for one retrieval panel (one ``record_uuid``).

    ``n_*_chunks`` count DISTINCT chunk_ids. The three sets overlap when a
    chunk has both submitted and discarded responses from different
    annotators; see module docstring for the derivation formulas.
    """

    record_uuid: str
    k: int  # n_retrieved_chunks metadata (max if records disagree; 0 if absent)
    n_annotated_chunks: int  # chunks with >=1 terminal response (union of submitted + discarded)
    n_submitted_chunks: int  # chunks with >=1 submitted response (used by panel_complete)
    n_discarded_chunks: int  # chunks with >=1 discarded response
    panel_complete: bool  # STRICT: k > 0 and n_submitted_chunks == k
    n_records_seen: int  # distinct chunk-records seen (for integrity check)


@dataclass(frozen=True)
class CompletenessReport:
    """Per-uuid completeness map + the aggregate summary for the export sidecar."""

    by_uuid: dict[str, PanelCompleteness]
    summary: CompletenessSummary


def k_bucket(k: int) -> str:
    """Bucket key for the by_k_bucket cross-tab. Mirrors handover (~15/61/24% split)."""
    if k < 5:
        return "k_lt_5"
    if k == 5:
        return "k_eq_5"
    return "k_gt_5"


_BUCKET_KEYS = ("k_lt_5", "k_eq_5", "k_gt_5")


def compute_completeness_from_records(snapshots: list[RetrievalRecordSnapshot]) -> CompletenessReport:
    """Pure aggregator over pre-walked retrieval snapshots.

    Shared by ``compute_completeness`` (standalone) and ``run_export`` (which
    walks once and feeds both the row-emission and the completeness passes).
    """
    groups: dict[str, dict] = {}
    n_orphans_skipped = 0

    for snap in snapshots:
        record_uuid = snap.record_uuid
        if not record_uuid:
            n_orphans_skipped += 1
            continue
        group = groups.setdefault(
            record_uuid,
            {
                "chunk_ids_seen": set(),
                "chunk_ids_terminal": set(),
                "chunk_ids_submitted": set(),
                "chunk_ids_discarded": set(),
                "k_max": 0,
                "k_min": 0,
                "n_records_with_k": 0,  # mixed-backfill detector
            },
        )
        group["chunk_ids_seen"].add(snap.chunk_id)
        if snap.n_retrieved_chunks_metadata > 0:
            k_meta = snap.n_retrieved_chunks_metadata
            group["k_max"] = max(group["k_max"], k_meta)
            group["k_min"] = k_meta if group["k_min"] == 0 else min(group["k_min"], k_meta)
            group["n_records_with_k"] += 1
        if snap.has_submitted:
            group["chunk_ids_submitted"].add(snap.chunk_id)
        if snap.has_discarded:
            group["chunk_ids_discarded"].add(snap.chunk_id)
        if snap.has_submitted or snap.has_discarded:
            group["chunk_ids_terminal"].add(snap.chunk_id)

    by_uuid: dict[str, PanelCompleteness] = {}
    n_integrity_warnings = 0
    n_panels_unknown_k = 0
    buckets: dict[str, dict[str, int]] = {key: {"n_panels": 0, "n_complete": 0} for key in _BUCKET_KEYS}
    by_k: dict[int, dict[str, int]] = {}
    n_complete = 0
    for uuid, group in groups.items():
        k_max = group["k_max"]
        k_min = group["k_min"]
        if k_min and k_max != k_min:
            logger.warning(
                "record_uuid=%s: inconsistent n_retrieved_chunks across records (min=%d max=%d) - using max",
                uuid,
                k_min,
                k_max,
            )
        k = k_max
        n_annotated = len(group["chunk_ids_terminal"])
        n_submitted = len(group["chunk_ids_submitted"])
        n_discarded = len(group["chunk_ids_discarded"])
        n_seen = len(group["chunk_ids_seen"])
        n_with_k = group["n_records_with_k"]
        # STRICT default: every chunk has a submitted response. Discards are
        # abstentions, not judgements - see module docstring.
        panel_complete = k > 0 and n_submitted == k
        if k > 0 and n_seen != k:
            logger.warning(
                "record_uuid=%s: integrity warning - saw %d distinct chunk-records but n_retrieved_chunks=%d",
                uuid,
                n_seen,
                k,
            )
            n_integrity_warnings += 1
        elif n_with_k and n_with_k < n_seen:
            logger.warning(
                "record_uuid=%s: integrity warning - %d/%d records carry n_retrieved_chunks metadata "
                "(panel partially backfilled)",
                uuid,
                n_with_k,
                n_seen,
            )
            n_integrity_warnings += 1
        if k == 0:
            n_panels_unknown_k += 1
        by_uuid[uuid] = PanelCompleteness(
            record_uuid=uuid,
            k=k,
            n_annotated_chunks=n_annotated,
            n_submitted_chunks=n_submitted,
            n_discarded_chunks=n_discarded,
            panel_complete=panel_complete,
            n_records_seen=n_seen,
        )
        bucket = buckets[k_bucket(k)]
        bucket["n_panels"] += 1
        k_stat = by_k.setdefault(k, {"n_panels": 0, "n_complete": 0})
        k_stat["n_panels"] += 1
        if panel_complete:
            bucket["n_complete"] += 1
            k_stat["n_complete"] += 1
            n_complete += 1

    if n_orphans_skipped:
        logger.warning(
            "retrieval completeness: %d record(s) skipped (empty record_uuid metadata)",
            n_orphans_skipped,
        )
    if by_uuid and n_panels_unknown_k == len(by_uuid):
        logger.warning(
            "retrieval completeness: ALL %d panel(s) have unknown K (n_retrieved_chunks metadata absent) - "
            "run scripts/backfill_n_retrieved_chunks.py to populate it.",
            len(by_uuid),
        )

    n_panels = len(by_uuid)
    fraction = (n_complete / n_panels) if n_panels else 0.0
    summary = CompletenessSummary(
        n_panels=n_panels,
        n_complete=n_complete,
        fraction_complete=fraction,
        by_k_bucket={key: KBucketStat(**value) for key, value in buckets.items()},
        by_k={k: KBucketStat(**value) for k, value in sorted(by_k.items())},
        n_integrity_warnings=n_integrity_warnings,
        n_orphans_skipped=n_orphans_skipped,
    )
    return CompletenessReport(by_uuid=by_uuid, summary=summary)


def compute_completeness(client: rg.Argilla, settings: AnnotationSettings) -> CompletenessReport:
    """Walk retrieval datasets once and compute panel completeness.

    Standalone-caller convenience; ``run_export`` calls ``walk_retrieval_records``
    once itself and feeds the result to both this aggregator and the export's
    row builder.
    """
    return compute_completeness_from_records(walk_retrieval_records(client, settings))
