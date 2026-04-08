"""Orchestrate IAA computation from exported annotation CSVs."""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from pragmata.core.annotation.iaa import (
    bootstrap_alpha,
    cohen_kappa,
    krippendorff_alpha_nominal,
    percentage_agreement,
)
from pragmata.core.paths.annotation_paths import AnnotationExportPaths, IaaPaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.iaa_report import (
    AnnotatorPair,
    IaaReport,
    LabelAgreement,
    TaskAgreement,
)

logger = logging.getLogger(__name__)

TASK_LABELS: dict[Task, list[str]] = {
    Task.RETRIEVAL: ["topically_relevant", "evidence_sufficient", "misleading"],
    Task.GROUNDING: [
        "support_present",
        "unsupported_claim_present",
        "contradicted_claim_present",
        "source_cited",
        "fabricated_source",
    ],
    Task.GENERATION: ["proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"],
}

_TASK_CSV: dict[Task, str] = {
    Task.RETRIEVAL: "retrieval_annotation_csv",
    Task.GROUNDING: "grounding_annotation_csv",
    Task.GENERATION: "generation_annotation_csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(value: str) -> bool:
    return value.lower() == "true"


def _pivot_label(rows: list[dict[str, str]], label: str) -> tuple[np.ndarray, list[str]]:
    """Pivot long-format rows into a wide (annotators x items) matrix.

    Returns:
        Tuple of (data matrix with NaN for missing, sorted annotator IDs).
    """
    records: dict[str, dict[str, bool]] = {}
    annotators: set[str] = set()
    for row in rows:
        rid = row["record_uuid"]
        aid = row["annotator_id"]
        annotators.add(aid)
        records.setdefault(rid, {})[aid] = _to_bool(row[label])

    ann_list = sorted(annotators)
    item_list = sorted(records.keys())
    ann_idx = {a: i for i, a in enumerate(ann_list)}

    data = np.full((len(ann_list), len(item_list)), np.nan)
    for j, rid in enumerate(item_list):
        for aid, val in records[rid].items():
            data[ann_idx[aid], j] = float(val)

    return data, ann_list


def _compute_pairwise_kappa(
    rows: list[dict[str, str]], labels: list[str], annotators: list[str]
) -> list[AnnotatorPair]:
    """Compute mean Cohen's kappa across labels for each annotator pair."""
    # Build per-record, per-annotator label vectors.
    by_annotator: dict[str, dict[str, dict[str, bool]]] = {}
    for row in rows:
        aid = row["annotator_id"]
        rid = row["record_uuid"]
        by_annotator.setdefault(aid, {})[rid] = {lab: _to_bool(row[lab]) for lab in labels}

    pairs: list[AnnotatorPair] = []
    for a, b in combinations(annotators, 2):
        shared = sorted(set(by_annotator.get(a, {})) & set(by_annotator.get(b, {})))
        if not shared:
            continue
        kappas = []
        for lab in labels:
            arr_a = np.array([by_annotator[a][r][lab] for r in shared], dtype=np.int8)
            arr_b = np.array([by_annotator[b][r][lab] for r in shared], dtype=np.int8)
            k = cohen_kappa(arr_a, arr_b)
            if not np.isnan(k):
                kappas.append(k)
        if not kappas:
            continue
        mean_kappa = float(np.mean(kappas))
        pairs.append(AnnotatorPair(annotator_a=a, annotator_b=b, kappa=mean_kappa, n_shared_items=len(shared)))

    return pairs


def run_iaa(
    export_paths: AnnotationExportPaths,
    iaa_paths: IaaPaths,
    tasks: list[Task],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> IaaReport:
    """Run IAA analysis on exported annotation CSVs.

    Args:
        export_paths: Resolved export path bundle (CSVs must exist).
        iaa_paths: Resolved IAA output path bundle.
        tasks: Tasks to analyse.
        n_resamples: Bootstrap iterations for confidence intervals.
        ci: Confidence level for bootstrap CIs.
        seed: Optional RNG seed for reproducible bootstrap.

    Returns:
        Populated IAA report, also written to ``iaa_paths.report``.
    """
    task_results: list[TaskAgreement] = []

    for task in tasks:
        csv_path: Path = getattr(export_paths, _TASK_CSV[task])
        if not csv_path.exists():
            logger.warning("Skipping %s: CSV not found at %s", task.value, csv_path)
            continue

        rows = _read_csv(csv_path)
        if not rows:
            logger.warning("Skipping %s: CSV is empty", task.value)
            continue

        labels = TASK_LABELS[task]
        label_results: list[LabelAgreement] = []
        all_annotators: list[str] = []

        for label in labels:
            data, annotators = _pivot_label(rows, label)
            if not all_annotators:
                all_annotators = annotators

            alpha = krippendorff_alpha_nominal(data)
            ci_lower, ci_upper = bootstrap_alpha(data, n_resamples=n_resamples, ci=ci, seed=seed)
            pct = percentage_agreement(data)

            # Count items with >= 2 annotations.
            n_items = int(np.sum(np.sum(~np.isnan(data), axis=0) >= 2))

            label_results.append(
                LabelAgreement(
                    label=label,
                    alpha=None if math.isnan(alpha) else alpha,
                    ci_lower=None if math.isnan(ci_lower) else ci_lower,
                    ci_upper=None if math.isnan(ci_upper) else ci_upper,
                    n_items=n_items,
                    n_annotators=len(annotators),
                    pct_agreement=None if math.isnan(pct) else pct,
                )
            )

        pairwise = _compute_pairwise_kappa(rows, labels, all_annotators)
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
