"""Annotation import implementation — validation, record building, and fan-out.

Validates raw dicts against the canonical schema, builds Argilla Record
objects from typed QueryResponsePair inputs, partitions them into calibration
vs production buckets per task, and logs them to the matching Argilla
datasets. The api/ layer resolves settings and delegates here.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling  # no public re-export in argilla v2; resolved to 2.8.0 (see pyproject.toml)

from pragmata.core.annotation.argilla_ops import create_dataset
from pragmata.core.annotation.argilla_task_definitions import (
    WIDGET_FIELD_PLACEHOLDERS,
    build_task_settings,
    dataset_name,
)
from pragmata.core.annotation.locales.registry import CATALOGS
from pragmata.core.schemas.annotation_import import (
    Chunk,
    PartitionManifest,
    PartitionManifestEntry,
    QueryResponsePair,
)
from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordError:
    """Validation failure for a single input record."""

    index: int
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validate_records(): typed pairs and per-index errors."""

    valid: list[QueryResponsePair]
    errors: list[RecordError]


def validate_records(records: list[dict[str, Any]]) -> ValidationResult:
    """Validate raw dicts against the canonical QueryResponsePair schema.

    Pure validation — no Argilla I/O.

    Args:
        records: Raw dictionaries to validate against QueryResponsePair.

    Returns:
        ValidationResult with successfully parsed pairs and per-index errors.
    """
    valid: list[QueryResponsePair] = []
    errors: list[RecordError] = []
    for i, raw in enumerate(records):
        try:
            valid.append(QueryResponsePair.model_validate(raw))
        except Exception as exc:
            errors.append(RecordError(index=i, detail=str(exc)))
    return ValidationResult(valid=valid, errors=errors)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def derive_record_uuid(pair: QueryResponsePair) -> str:
    """SHA-256 digest of canonical content fields — stable across calls for identical pairs."""
    # chunk_ids sorted for order invariance — same chunks in any order produce the same UUID
    chunk_ids = "|".join(sorted(c.chunk_id for c in pair.chunks))
    canonical = f"{pair.query}\x00{pair.answer}\x00{pair.context_set}\x00{chunk_ids}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _chunk_id_digest(chunk_id: str) -> str:
    """Short stable digest of ``chunk_id`` for Argilla record ids.

    Hashing keeps the id charset-safe regardless of how chunk_ids are formatted,
    while staying stable under chunk reordering (unlike a positional index).
    """
    return hashlib.sha256(chunk_id.encode()).hexdigest()[:16]


def build_retrieval_record_for_chunk(
    pair: QueryResponsePair,
    record_uuid: str,
    chunk: Chunk,
) -> rg.Record:
    """One Argilla record for a single (pair, chunk) annotation item.

    The Argilla record id is derived from ``(record_uuid, chunk_id)`` so it
    matches the partition identity: reimporting a pair with chunks in a
    different order keeps each chunk's record id stable and prevents stale
    records being stranded in the wrong calibration/production dataset.
    """
    metadata: dict = {
        "record_uuid": record_uuid,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_rank": chunk.chunk_rank,
    }
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"ret-{record_uuid}-{_chunk_id_digest(chunk.chunk_id)}",
        fields={
            "query": pair.query,
            "chunk": chunk.text,
            "generated_answer": {"text": pair.answer},
            **WIDGET_FIELD_PLACEHOLDERS,
        },
        metadata=metadata,
    )


def build_grounding_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    """Single Argilla record for grounding evaluation."""
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gnd-{record_uuid}",
        fields={
            "answer": pair.answer,
            "context_set": pair.context_set,
            "query": {"text": pair.query},
            **WIDGET_FIELD_PLACEHOLDERS,
        },
        metadata=metadata,
    )


def build_generation_record(pair: QueryResponsePair, record_uuid: str) -> rg.Record:
    """Single Argilla record for generation evaluation."""
    metadata: dict = {"record_uuid": record_uuid}
    if pair.language is not None:
        metadata["language"] = pair.language
    return rg.Record(
        id=f"gen-{record_uuid}",
        fields={
            "query": pair.query,
            "answer": pair.answer,
            "context_set": {"text": pair.context_set},
            **WIDGET_FIELD_PLACEHOLDERS,
        },
        metadata=metadata,
    )


def _invert_workspace_map(settings: AnnotationSettings) -> dict[Task, str]:
    """Invert workspaces topology to task → workspace_base."""
    return {task: ws_base for ws_base, ws in settings.workspaces.items() for task in ws.tasks}


# ---------------------------------------------------------------------------
# Partition logic
# ---------------------------------------------------------------------------


def _calibration_digest(unit_id: str, task: Task, seed: int) -> int:
    """Deterministic per-(task, unit) digest in [0, 2^32).

    Hash input mixes seed, task name, and the annotation unit identifier so
    per-task draws are statistically independent: a unit that lands in
    retrieval-calibration is not constrained to also land in
    grounding-calibration. The unit_id is ``record_uuid`` for grounding /
    generation (one item per record) and ``f"{record_uuid}:{chunk_id}"`` for
    retrieval (one item per chunk).
    """
    return int(hashlib.sha256(f"{seed}\x00{task.value}\x00{unit_id}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Partition manifest IO
# ---------------------------------------------------------------------------


def load_partition_manifest(path: Path, *, dataset_id: str, partition_seed: int) -> PartitionManifest:
    """Load an existing manifest from disk or create an empty one.

    Raises if the on-disk manifest's dataset_id does not match the caller's,
    which would indicate a base_dir or scope mispoint that could silently
    corrupt assignments across scopes.
    """
    if path.exists():
        manifest = PartitionManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.dataset_id != dataset_id:
            raise ValueError(
                f"Partition manifest at {path} has dataset_id={manifest.dataset_id!r} "
                f"but the current scope is {dataset_id!r}. Refusing to load to avoid "
                "cross-scope assignment corruption."
            )
        return manifest
    now = datetime.now(timezone.utc)
    return PartitionManifest(
        dataset_id=dataset_id,
        created_at=now,
        updated_at=now,
        partition_seed=partition_seed,
        assignments={},
    )


def write_partition_manifest(path: Path, manifest: PartitionManifest) -> None:
    """Write manifest atomically (write-tmp-then-rename).

    The parent directory must already exist; path resolution and directory
    creation belong in core/paths/ (see AnnotationImportPaths.ensure_dirs).
    """
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class PartitionResult:
    """Outcome of ``assign_partitions()``.

    Bundles the per-record manifest entries with the per-task fraction
    resolved this run and the rid → pair map, so downstream callers don't
    redo the workspace walk or recompute UUIDs.

    Attributes:
        assignments: record_uuid → PartitionManifestEntry, covering every
            input pair (existing + newly assigned).
        pairs_by_rid: record_uuid → QueryResponsePair for every input pair.
        calibration_fraction: Per-task resolved fraction in force this run.
            ``0.0`` for tasks absent from the workspaces topology.
    """

    assignments: dict[str, PartitionManifestEntry]
    pairs_by_rid: dict[str, QueryResponsePair]
    calibration_fraction: dict[Task, float]


def assign_partitions(
    pairs: list[QueryResponsePair],
    *,
    manifest: PartitionManifest,
    settings: AnnotationSettings,
    import_id: str,
) -> PartitionResult:
    """Resolve per-(task, annotation-item) calibration assignments.

    The annotation item differs by task: for grounding and generation, one item
    per ``record_uuid``; for retrieval, one item per ``(record_uuid, chunk_id)``.
    Per-task fraction is resolved via ``settings.resolved_task`` so workspace /
    task overrides for ``calibration_fraction`` are honoured.

    Existing assignments are locked for re-import stability and never rewritten.
    A new record is bucketed across all active tasks by per-(task, unit) digest
    ``hash(seed || task || unit_id)`` against ``fraction * 2^32``. An existing
    record is *backfilled* for any task the topology has since gained (e.g.
    retrieval-only at first import, grounding added later) using the same
    deterministic digest, so late-activated tasks partition pre-existing records
    identically to fresh ones rather than silently defaulting to production.

    Args:
        pairs: Validated pairs to partition.
        manifest: Mutated in place to record new and backfilled assignments.
        settings: Resolves per-task fraction via ``resolved_task``.
        import_id: Stamped on new entries for provenance.

    Returns:
        A ``PartitionResult`` bundling the assignments, the rid → pair map, and
        the per-task fraction resolved this run.
    """
    now = datetime.now(timezone.utc)
    seed = manifest.partition_seed
    workspace_for_task = _invert_workspace_map(settings)

    pairs_by_rid: dict[str, QueryResponsePair] = {derive_record_uuid(pair): pair for pair in pairs}
    per_task_fraction: dict[Task, float] = {}
    threshold_for_task: dict[Task, int | None] = {}
    for task in Task:
        ws_base = workspace_for_task.get(task)
        if ws_base is None:
            per_task_fraction[task] = 0.0
            threshold_for_task[task] = None
            continue
        fraction = settings.resolved_task(ws_base, task).calibration_fraction
        per_task_fraction[task] = fraction
        threshold_for_task[task] = int(fraction * (2**32)) if 0.0 < fraction < 1.0 else None

    def is_cal(task: Task, unit_id: str) -> bool:
        fraction = per_task_fraction[task]
        threshold = threshold_for_task[task]
        return fraction >= 1.0 or (threshold is not None and _calibration_digest(unit_id, task, seed) < threshold)

    active_tasks = {task for task in Task if workspace_for_task.get(task) is not None}
    assignments: dict[str, PartitionManifestEntry] = {}
    changed = False
    for rid, pair in pairs_by_rid.items():
        existing = manifest.assignments.get(rid)
        if existing is None:
            grnd_gen: dict[Task, bool] = {}
            retrieval: dict[str, bool] = {}
            for task in active_tasks:
                for unit_id, chunk_id in _enumerate_units(rid, pair, task):
                    _write_unit(grnd_gen, retrieval, chunk_id, task, is_cal=is_cal(task, unit_id))
            entry = PartitionManifestEntry(
                grounding_generation_calibration=grnd_gen,
                retrieval_chunk_calibration=retrieval,
                import_id=import_id,
                calibration_fraction_at_import=dict(per_task_fraction),
                assigned_at=now,
            )
            manifest.assignments[rid] = entry
            assignments[rid] = entry
            changed = True
            continue

        entry = _backfill_entry(existing, rid, pair, active_tasks, per_task_fraction, is_cal)
        if entry is not existing:
            manifest.assignments[rid] = entry
            changed = True
        assignments[rid] = entry

    if changed:
        manifest.updated_at = now
    return PartitionResult(
        assignments=assignments,
        pairs_by_rid=pairs_by_rid,
        calibration_fraction=per_task_fraction,
    )


def _backfill_entry(
    entry: PartitionManifestEntry,
    rid: str,
    pair: QueryResponsePair,
    active_tasks: set[Task],
    per_task_fraction: dict[Task, float],
    is_cal: Callable[[Task, str], bool],
) -> PartitionManifestEntry:
    """Assign active tasks/chunks absent from an existing entry; never rewrite.

    Returns ``entry`` unchanged when nothing is missing, else a new entry with
    the newly-active units assigned via the same deterministic digest a fresh
    import uses. Provenance (``import_id``, ``assigned_at``) is preserved from the
    original assignment; ``calibration_fraction_at_import`` records the resolved
    fraction for each backfilled task.
    """
    grnd_gen = dict(entry.grounding_generation_calibration)
    retrieval = dict(entry.retrieval_chunk_calibration)
    fraction_at_import = dict(entry.calibration_fraction_at_import)
    added = False
    for task in active_tasks:
        for unit_id, chunk_id in _enumerate_units(rid, pair, task):
            if task == Task.RETRIEVAL:
                assert chunk_id is not None  # retrieval always has chunk_id
                if chunk_id in retrieval:
                    continue
                retrieval[chunk_id] = is_cal(task, unit_id)
            elif task not in grnd_gen:
                grnd_gen[task] = is_cal(task, unit_id)
            else:
                continue
            fraction_at_import[task] = per_task_fraction[task]
            added = True
    if not added:
        return entry
    return entry.model_copy(
        update={
            "grounding_generation_calibration": grnd_gen,
            "retrieval_chunk_calibration": retrieval,
            "calibration_fraction_at_import": fraction_at_import,
        }
    )


def _enumerate_units(rid: str, pair: QueryResponsePair, task: Task) -> list[tuple[str, str | None]]:
    """Annotation units for this (pair, task) as ``(unit_id, chunk_id_or_None)``."""
    if task == Task.RETRIEVAL:
        return [(f"{rid}:{chunk.chunk_id}", chunk.chunk_id) for chunk in pair.chunks]
    return [(rid, None)]


def _write_unit(
    grnd_gen: dict[Task, bool],
    retrieval: dict[str, bool],
    chunk_id: str | None,
    task: Task,
    *,
    is_cal: bool,
) -> None:
    """Route a per-task assignment into the right per-record bucket."""
    if task == Task.RETRIEVAL:
        assert chunk_id is not None  # retrieval always has chunk_id
        retrieval[chunk_id] = is_cal
    else:
        grnd_gen[task] = is_cal


@dataclass(frozen=True)
class TaskUnitCounts:
    """Per-task tallies across a collection of manifest entries.

    Annotation units are chunks for retrieval and records for grounding /
    generation. ``calibration`` counts only items routed to calibration;
    ``total`` counts all items (calibration + production).
    """

    calibration: dict[Task, int]
    total: dict[Task, int]


def count_units_per_task(entries: Iterable[PartitionManifestEntry]) -> TaskUnitCounts:
    """Single pass over ``entries`` returning per-task calibration and total counts."""
    calibration: dict[Task, int] = {task: 0 for task in Task}
    total: dict[Task, int] = {task: 0 for task in Task}
    for entry in entries:
        for task in (Task.GROUNDING, Task.GENERATION):
            if task in entry.grounding_generation_calibration:
                total[task] += 1
                if entry.grounding_generation_calibration[task]:
                    calibration[task] += 1
        for is_cal in entry.retrieval_chunk_calibration.values():
            total[Task.RETRIEVAL] += 1
            if is_cal:
                calibration[Task.RETRIEVAL] += 1
    return TaskUnitCounts(calibration=calibration, total=total)


@dataclass(frozen=True)
class PartitionSummary:
    """Per-task partition reporting derived from a set of manifest entries.

    Each dict is keyed by ``Task``. Retrieval counts chunks; grounding and
    generation count records.

    Attributes:
        calibration_count: Annotation items routed to calibration.
        total_count: Total annotation items (calibration + production).
        production_count: Annotation items routed to production.
        realised_fraction: Actual share routed to calibration this run
            (calibration / total). Zero when no items were assigned.
        configured_fraction: Configured ``calibration_fraction`` resolved per
            task. May differ from ``realised_fraction`` on re-imports because
            prior assignments are locked by the manifest.
    """

    calibration_count: dict[Task, int]
    total_count: dict[Task, int]
    production_count: dict[Task, int]
    realised_fraction: dict[Task, float]
    configured_fraction: dict[Task, float]


def summarize_partitions(
    entries: Iterable[PartitionManifestEntry],
    configured_fraction: dict[Task, float],
) -> PartitionSummary:
    """Compute per-task partition reporting from manifest entries.

    Counts calibration vs total annotation items per task in a single pass
    (retrieval counts chunks; grounding/generation count records) and derives
    the realised calibration fraction. ``configured_fraction`` is the per-task
    fraction already resolved by ``assign_partitions`` (see
    ``PartitionResult.calibration_fraction``); it is carried through unchanged
    rather than re-resolved from settings, keeping a single source of truth.
    """
    counts = count_units_per_task(entries)
    calibration_count = counts.calibration
    total_count = counts.total
    production_count = {task: total_count[task] - calibration_count[task] for task in Task}
    realised_fraction = {
        task: (calibration_count[task] / total_count[task]) if total_count[task] else 0.0 for task in Task
    }
    return PartitionSummary(
        calibration_count=calibration_count,
        total_count=total_count,
        production_count=production_count,
        realised_fraction=realised_fraction,
        configured_fraction=configured_fraction,
    )


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def _build_batches(
    pairs_by_rid: dict[str, QueryResponsePair],
    assignments: dict[str, PartitionManifestEntry],
) -> dict[tuple[Task, bool], list[rg.Record]]:
    """Build Argilla records keyed by (task, calibration) bucket.

    Per-item routing: grounding and generation read their flag from
    ``entry.grounding_generation_calibration``; each retrieval chunk reads its
    flag from ``entry.retrieval_chunk_calibration[chunk_id]`` independently so
    different chunks of one record can land in different buckets.
    """
    batches: dict[tuple[Task, bool], list[rg.Record]] = {
        (task, is_cal): [] for task in Task for is_cal in (False, True)
    }
    for record_uuid, pair in pairs_by_rid.items():
        entry = assignments[record_uuid]
        grnd_cal = entry.grounding_generation_calibration.get(Task.GROUNDING, False)
        gen_cal = entry.grounding_generation_calibration.get(Task.GENERATION, False)
        batches[(Task.GROUNDING, grnd_cal)].append(build_grounding_record(pair, record_uuid))
        batches[(Task.GENERATION, gen_cal)].append(build_generation_record(pair, record_uuid))
        for chunk in pair.chunks:
            chunk_cal = entry.retrieval_chunk_calibration.get(chunk.chunk_id, False)
            batches[(Task.RETRIEVAL, chunk_cal)].append(build_retrieval_record_for_chunk(pair, record_uuid, chunk))
    return batches


def _detect_dataset_locale(dataset: rg.Dataset, task: Task) -> Locale | None:
    """Infer an existing Argilla dataset's creation locale from its label displays.

    Probes the first ``LabelQuestion``'s value→display map (via the private
    ``_model.settings.options`` — public ``.labels`` is lossy, returns keys
    only) against each per-locale catalog. Returns ``None`` if no locale
    matches — e.g. when the dataset was provisioned outside pragmata.
    """
    question = next(
        (q for q in dataset.settings.questions if isinstance(q, rg.LabelQuestion)),
        None,
    )
    if question is None:
        return None
    try:
        options = question._model.settings.options
    except AttributeError:
        return None
    existing = {opt["value"]: opt["text"] for opt in options}
    for loc, catalog in CATALOGS.items():
        rendered = {v: catalog.get((task, "label", f"{question.name}.{v}")) for v in existing}
        if all(rendered.values()) and rendered == existing:
            return loc
    return None


def _ensure_dataset(
    client: rg.Argilla,
    *,
    task: Task,
    calibration: bool,
    min_submitted: int,
    ws_base: str,
    dataset_id: str,
    task_settings_map: dict[Task, rg.Settings],
    locale: Locale,
) -> rg.Dataset:
    """Resolve or create the Argilla dataset for a (task, purpose) pair.

    When an existing dataset is found, its creation locale is probed against
    the configured ``locale``. A mismatch is logged as a warning and the
    import proceeds — label *values* are locale-invariant so data integrity
    is preserved (only display text differs).
    """
    ds_name = dataset_name(task, calibration=calibration, dataset_id=dataset_id)
    workspace = client.workspaces(ws_base)
    if workspace is None:
        raise RuntimeError(f"Workspace {ws_base!r} not found. Run setup() first.")
    base_settings = task_settings_map[task]
    task_cfg = rg.Settings(
        fields=base_settings.fields,
        questions=base_settings.questions,
        metadata=base_settings.metadata,
        guidelines=base_settings.guidelines,
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )
    dataset, ds_created = create_dataset(client, ds_name, ws_base, task_cfg)
    if ds_created:
        logger.info("Auto-created dataset %r in workspace %r", ds_name, ws_base)
    else:
        existing_locale = _detect_dataset_locale(dataset, task)
        if existing_locale is not None and existing_locale != locale:
            logger.warning(
                "Locale mismatch on dataset %r (workspace %r): existing dataset was created "
                "with locale=%r but this import uses locale=%r. Appending records (label "
                "values are locale-invariant; only display text differs).",
                ds_name,
                ws_base,
                existing_locale,
                locale,
            )
    return dataset


def fan_out_records(
    client: rg.Argilla,
    settings: AnnotationSettings,
    *,
    partition: PartitionResult,
) -> dict[str, int]:
    """Build and log Argilla records to per-purpose datasets.

    Datasets are created on-the-fly if they don't exist (idempotent).
    Workspaces must already exist (call setup() first).

    Args:
        client: Argilla client.
        settings: Annotation settings (topology, dataset_id).
        partition: Output of ``assign_partitions``. Carries both the per-record
            manifest entries and the pair-by-rid map needed for fan-out.

    Returns:
        Mapping of dataset name to record count for that dataset. Per-task
        calibration vs production totals are computable from
        ``partition.assignments`` and stay in the api layer.
    """
    task_to_ws = _invert_workspace_map(settings)
    batches = _build_batches(partition.pairs_by_rid, partition.assignments)

    dataset_counts: dict[str, int] = {}
    # Memoise build_task_settings per locale — its output depends only on (settings, locale).
    settings_by_locale: dict[Locale, dict[Task, rg.Settings]] = {}

    for (task, calibration), rg_records in batches.items():
        if not rg_records:
            continue
        ws_base = task_to_ws.get(task)
        if ws_base is None:
            logger.warning("Task %r not in workspaces topology - skipping", task)
            continue
        resolved = settings.resolved_task(ws_base, task)
        if calibration:
            if resolved.calibration_min_submitted is None:
                # assign_partitions only assigns calibration when topology supports
                # it; surfacing as an error rather than silently routing to production.
                raise RuntimeError(
                    f"Task {task.value} has calibration records assigned but topology disables calibration"
                )
            min_submitted = resolved.calibration_min_submitted
        else:
            min_submitted = resolved.production_min_submitted
        locale = resolved.locale
        if locale not in settings_by_locale:
            settings_by_locale[locale] = build_task_settings(settings, locale)
        task_settings_map = settings_by_locale[locale]
        dataset = _ensure_dataset(
            client,
            task=task,
            calibration=calibration,
            min_submitted=min_submitted,
            ws_base=ws_base,
            dataset_id=settings.dataset_id,
            task_settings_map=task_settings_map,
            locale=locale,
        )
        dataset.records.log(rg_records, on_error=RecordErrorHandling.WARN)
        ds_name = dataset.name
        dataset_counts[ds_name] = len(rg_records)
        logger.info("Logged %d records to dataset %r", len(rg_records), ds_name)

    return dataset_counts
