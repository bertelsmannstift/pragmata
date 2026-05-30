"""Per-panel retrieval completeness: how many chunks per query have a terminal response.

A "panel" is the set of retrieval chunk-records sharing one ``record_uuid``
(one query). A chunk counts as resolved if it has at least one terminal
response (submitted OR discarded) - a deliberately-discarded chunk is
metric-covered, not a hole.

Computation is INDEPENDENT of the export's ``include_discarded`` flag (which
gates only row emission to the CSV). This module issues its own retrieval
fetch and always treats both terminal statuses as covered, so the resulting
``panel_complete`` / ``n_annotated_chunks`` are stable regardless of the
export caller's discard policy.
"""

import logging
from dataclasses import dataclass

import argilla as rg

from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.annotation.export_fetcher import TERMINAL_STATUSES, resolve_task_purposes
from pragmata.core.schemas.annotation_export import CompletenessSummary, KBucketStat
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanelCompleteness:
    """Completeness facts for one retrieval panel (one ``record_uuid``)."""

    record_uuid: str
    k: int
    n_annotated_chunks: int
    panel_complete: bool
    n_records_seen: int


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


def compute_completeness(client: rg.Argilla, settings: AnnotationSettings) -> CompletenessReport:
    """Compute per-record_uuid retrieval panel completeness across prod + cal.

    Iterates retrieval records UNFILTERED (no response.status query filter) so
    the integrity check sees the full record set, then evaluates each
    record's terminal status in Python. Discarded responses count as covered.
    """
    workspace_name, purposes = resolve_task_purposes(settings, Task.RETRIEVAL)

    groups: dict[str, dict] = {}
    n_orphans_skipped = 0

    for calibration in purposes:
        ds_name = dataset_name(Task.RETRIEVAL, calibration=calibration, dataset_id=settings.dataset_id)
        dataset = client.datasets(ds_name, workspace=workspace_name)
        if dataset is None:
            continue
        for record in dataset.records(with_responses=True):
            record_uuid: str = record.metadata.get("record_uuid", "")
            if not record_uuid:
                n_orphans_skipped += 1
                continue
            chunk_id: str = record.metadata.get("chunk_id", "")
            k_meta = int(record.metadata.get("n_retrieved_chunks") or 0)
            group = groups.setdefault(
                record_uuid,
                {
                    "chunk_ids_terminal": set(),
                    "chunk_ids_seen": set(),
                    "k_max": 0,
                    "k_min": 0,
                },
            )
            group["chunk_ids_seen"].add(chunk_id)
            if k_meta > 0:
                group["k_max"] = max(group["k_max"], k_meta)
                group["k_min"] = k_meta if group["k_min"] == 0 else min(group["k_min"], k_meta)
            if any(r.status in TERMINAL_STATUSES for r in (record.responses or [])):
                group["chunk_ids_terminal"].add(chunk_id)

    by_uuid: dict[str, PanelCompleteness] = {}
    n_integrity_warnings = 0
    # Fuse the per-uuid aggregation and the bucket cross-tab into one pass
    # so we don't walk by_uuid twice.
    buckets: dict[str, dict[str, int]] = {key: {"n_panels": 0, "n_complete": 0} for key in _BUCKET_KEYS}
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
        n_seen = len(group["chunk_ids_seen"])
        panel_complete = k > 0 and n_annotated == k
        if k > 0 and n_seen != k:
            logger.warning(
                "record_uuid=%s: integrity warning - saw %d distinct chunk-records but n_retrieved_chunks=%d",
                uuid,
                n_seen,
                k,
            )
            n_integrity_warnings += 1
        by_uuid[uuid] = PanelCompleteness(
            record_uuid=uuid,
            k=k,
            n_annotated_chunks=n_annotated,
            panel_complete=panel_complete,
            n_records_seen=n_seen,
        )
        bucket = buckets[k_bucket(k)]
        bucket["n_panels"] += 1
        if panel_complete:
            bucket["n_complete"] += 1
            n_complete += 1

    if n_orphans_skipped:
        logger.warning(
            "retrieval completeness: %d record(s) skipped (empty record_uuid metadata)",
            n_orphans_skipped,
        )

    n_panels = len(by_uuid)
    fraction = (n_complete / n_panels) if n_panels else 0.0
    summary = CompletenessSummary(
        n_panels=n_panels,
        n_complete=n_complete,
        fraction_complete=fraction,
        by_k_bucket={key: KBucketStat(**value) for key, value in buckets.items()},
        n_integrity_warnings=n_integrity_warnings,
        n_orphans_skipped=n_orphans_skipped,
    )
    return CompletenessReport(by_uuid=by_uuid, summary=summary)
