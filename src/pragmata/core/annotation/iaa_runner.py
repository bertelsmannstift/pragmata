"""Orchestrate IAA computation from exported annotation CSVs."""

import logging
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from pragmata.core.annotation.export_runner import TASK_ANNOTATION_SCHEMA, TASK_CSV_ATTR, TASK_EXPORT_ROW
from pragmata.core.annotation.iaa import (
    bootstrap_alpha,
    cohen_kappa,
    krippendorff_alpha_nominal,
    percentage_agreement,
)
from pragmata.core.csv_io import read_csv
from pragmata.core.paths.annotation_paths import AnnotationExportPaths, IaaPaths
from pragmata.core.schemas.annotation_export import AnnotationBase
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.iaa_report import (
    AnnotatorPair,
    IaaReport,
    LabelAgreement,
    TaskAgreement,
)

logger = logging.getLogger(__name__)

TASK_LABELS: dict[Task, list[str]] = {
    task: [name for name, info in schema.model_fields.items() if info.annotation in (bool, bool | None)]
    for task, schema in TASK_ANNOTATION_SCHEMA.items()
}


def _or_none(value: float) -> float | None:
    """Convert NaN to None for JSON-safe Pydantic fields."""
    return None if math.isnan(value) else value


def _pivot_task(
    rows: list[AnnotationBase], labels: list[str]
) -> tuple[dict[str, np.ndarray], list[str], dict[str, dict[str, dict[str, bool]]]]:
    """Pivot all labels for a task in a single pass over rows.

    Returns:
        Tuple of (label -> (annotators x items) matrix, sorted annotator IDs,
        parsed records structure for pairwise kappa).
    """
    annotators: set[str] = set()
    # record_uuid -> annotator_id -> {label: bool}
    records: dict[str, dict[str, dict[str, bool]]] = {}
    for row in rows:
        rid = row.record_uuid
        aid = row.annotator_id
        annotators.add(aid)
        records.setdefault(rid, {}).setdefault(aid, {})
        for lab in labels:
            records[rid][aid][lab] = getattr(row, lab)

    ann_list = sorted(annotators)
    item_list = sorted(records.keys())
    ann_idx = {a: i for i, a in enumerate(ann_list)}

    matrices: dict[str, np.ndarray] = {}
    for lab in labels:
        data = np.full((len(ann_list), len(item_list)), np.nan)
        for j, rid in enumerate(item_list):
            for aid, vals in records[rid].items():
                v = vals[lab]
                if v is not None:
                    data[ann_idx[aid], j] = float(v)
        matrices[lab] = data

    return matrices, ann_list, records


def _compute_pairwise_kappa(
    records: dict[str, dict[str, dict[str, bool]]], labels: list[str], annotators: list[str]
) -> list[AnnotatorPair]:
    """Compute mean Cohen's kappa across labels for each annotator pair."""
    pairs: list[AnnotatorPair] = []
    for a, b in combinations(annotators, 2):
        shared = sorted(rid for rid, anns in records.items() if a in anns and b in anns)
        if not shared:
            continue
        kappas = []
        for lab in labels:
            arr_a = np.array([records[r][a][lab] for r in shared], dtype=np.int8)
            arr_b = np.array([records[r][b][lab] for r in shared], dtype=np.int8)
            k = cohen_kappa(arr_a, arr_b)
            if not np.isnan(k):
                kappas.append(k)
        if not kappas:
            continue
        mean_kappa = float(np.mean(kappas))
        pairs.append(AnnotatorPair(annotator_a=a, annotator_b=b, kappa=mean_kappa, n_shared_items=len(shared)))

    return pairs


def _filter_rows(
    rows: list[AnnotationBase],
    *,
    exclude_annotators: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[AnnotationBase]:
    """Filter annotation rows by annotator and/or time window."""
    excluded = set(exclude_annotators) if exclude_annotators else set()
    filtered = []
    for row in rows:
        if row.annotator_id in excluded:
            continue
        if after and row.created_at < after:
            continue
        if before and row.created_at > before:
            continue
        filtered.append(row)
    return filtered


def run_iaa(
    export_paths: AnnotationExportPaths,
    iaa_paths: IaaPaths,
    tasks: list[Task],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
    exclude_annotators: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> IaaReport:
    """Run IAA analysis on exported annotation CSVs.

    Args:
        export_paths: Resolved export path bundle (CSVs must exist).
        iaa_paths: Resolved IAA output path bundle.
        tasks: Tasks to analyse.
        n_resamples: Bootstrap iterations for confidence intervals.
        ci: Confidence level for bootstrap CIs.
        seed: Optional RNG seed for reproducible bootstrap.
        exclude_annotators: Annotator IDs to exclude from analysis.
        after: Only include annotations created after this datetime.
        before: Only include annotations created before this datetime.

    Returns:
        Populated IAA report, also written to ``iaa_paths.report``.
    """
    task_results: list[TaskAgreement] = []

    for task in tasks:
        csv_path: Path = getattr(export_paths, TASK_CSV_ATTR[task])
        if not csv_path.exists():
            logger.warning("Skipping %s: CSV not found at %s", task.value, csv_path)
            continue

        rows = _filter_rows(
            read_csv(csv_path, TASK_EXPORT_ROW[task]),
            exclude_annotators=exclude_annotators,
            after=after,
            before=before,
        )
        if not rows:
            logger.warning("Skipping %s: CSV is empty", task.value)
            continue

        labels = TASK_LABELS[task]
        matrices, annotators, records = _pivot_task(rows, labels)

        label_results: list[LabelAgreement] = []
        for label in labels:
            data = matrices[label]

            alpha = krippendorff_alpha_nominal(data)
            ci_lower, ci_upper = bootstrap_alpha(data, n_resamples=n_resamples, ci=ci, seed=seed)
            pct = percentage_agreement(data)
            n_items = int(np.sum(np.sum(~np.isnan(data), axis=0) >= 2))

            label_results.append(
                LabelAgreement(
                    label=label,
                    alpha=_or_none(alpha),
                    ci_lower=_or_none(ci_lower),
                    ci_upper=_or_none(ci_upper),
                    n_items=n_items,
                    n_annotators=len(annotators),
                    pct_agreement=_or_none(pct),
                )
            )

        pairwise = _compute_pairwise_kappa(records, labels, annotators)
        task_results.append(TaskAgreement(task=task, labels=label_results, pairwise_kappa=pairwise))
        logger.info("IAA for %s: %d labels, %d annotator pairs", task.value, len(label_results), len(pairwise))

    report = IaaReport(
        export_id=export_paths.export_dir.name,
        created_at=datetime.now(timezone.utc),
        tasks=task_results,
        n_bootstrap_resamples=n_resamples,
        ci_level=ci,
    )

    iaa_paths.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("IAA report written to %s", iaa_paths.report)

    return report
