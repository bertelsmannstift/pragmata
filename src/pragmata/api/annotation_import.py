"""Annotation import API - thin orchestration over core/ implementation."""

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.loaders import RecordInput, resolve_records
from pragmata.core.annotation.locales.registry import register_catalog_dir
from pragmata.core.annotation.record_builder import (
    RecordError,
    assign_partitions,
    count_calibration_per_task,
    fan_out_records,
    load_partition_manifest,
    validate_records,
    write_partition_manifest,
)
from pragmata.core.paths.annotation_paths import (
    resolve_annotation_paths,
    resolve_import_paths,
)
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_import import PartitionManifestEntry
from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportResult:
    """Outcome of import_records(): per-task counts, dataset counts, and validation errors.

    Per-task dicts are keyed by ``Task`` because partition is per-(task,
    annotation-item): retrieval counts chunks; grounding and generation
    count records. Values fall through to the resolved per-(workspace, task)
    settings when present.

    Attributes:
        total_records: Number of raw dicts received as input.
        dataset_counts: Records submitted per dataset name.
        calibration_count: Per-task count of annotation items routed to the
            calibration dataset.
        production_count: Per-task count of annotation items routed to the
            production dataset.
        calibration_fraction: Per-task configured fraction. May differ from
            ``realised_calibration_fraction`` on re-imports because prior
            assignments are locked by the manifest, and under a binding cap.
        realised_calibration_fraction: Per-task actual share of items routed
            to calibration this run (calibration / (calibration + production)).
            Zero when no items were assigned for that task.
        calibration_max_records: Per-task configured absolute cap. ``None`` =
            uncapped for that task.
        errors: Per-record validation failures (index + detail).
    """

    total_records: int
    dataset_counts: dict[str, int]
    calibration_count: dict[Task, int] = field(default_factory=dict)
    production_count: dict[Task, int] = field(default_factory=dict)
    calibration_fraction: dict[Task, float] = field(default_factory=dict)
    realised_calibration_fraction: dict[Task, float] = field(default_factory=dict)
    calibration_max_records: dict[Task, int | None] = field(default_factory=dict)
    errors: list[RecordError] = field(default_factory=list)


def import_records(
    records: RecordInput,
    *,
    api_url: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    format: str = "auto",
    dataset_id: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    calibration_fraction: float | Unset = UNSET,
    calibration_max_records: int | None | Unset = UNSET,
    calibration_min_submitted: int | None | Unset = UNSET,
    calibration_partition_seed: int | Unset = UNSET,
    locale: Locale | Unset = UNSET,
    locale_catalog_dir: str | Path | None | Unset = UNSET,
) -> ImportResult:
    """Validate and fan out records to per-purpose Argilla annotation datasets.

    Accepts raw dicts, file paths (JSON, JSONL, CSV), HuggingFace Datasets,
    or pandas DataFrames. File format is detected by extension or overridden
    via the format kwarg. All inputs are resolved to list[dict] before
    validation against the canonical import schema.

    Valid records are partitioned into calibration vs production buckets via
    a deterministic record_uuid hash. Calibration records go to
    ``task_<task>_calibration`` (overlap = ``calibration_min_submitted``),
    production records go to ``task_<task>_production`` (overlap =
    ``production_min_submitted``). The partition is locked across re-imports
    via a manifest sidecar at
    ``annotation/imports/{dataset_id}/partition.meta.json``.

    Validation failures are collected in ImportResult.errors - invalid
    records are skipped, not raised.

    Datasets are auto-created if they don't exist. Workspaces must already
    exist (call setup() first). Record IDs are derived from content hashes
    for idempotent upsert.

    Credential resolution:
    - ``api_url``: kwarg > ``ARGILLA_API_URL`` env > config (``argilla.api_url``)
    - ``api_key``: kwarg > ``ARGILLA_API_KEY`` env (secrets never live in config)
    - ``locale``: kwarg > ``--config`` file (``annotation.locale``) > default
      EN. Cascades to per-workspace/per-task overrides defined in the YAML
      config.

    Args:
        records: Input data — list[dict], file path (str/Path), HF Dataset,
            or pandas DataFrame.
        api_url: Argilla server URL.
        api_key: Argilla API key.
        format: File format override — 'auto' (default), 'json', 'jsonl',
            or 'csv'. Only used for str/Path inputs.
        dataset_id: Suffix appended to dataset names for run scoping.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.
        calibration_fraction: Deployment-level fraction of annotation items
            routed to the calibration dataset (inherited by workspaces/tasks
            unless overridden in YAML config). Falls through to YAML config
            and the built-in default (0.1) when omitted. Set to 0.0 for
            production-only batches.
        calibration_max_records: Deployment-level absolute cap on calibration
            annotation items per task. ``None`` is uncapped (just the
            fractional knob). Inherited by workspaces/tasks unless overridden
            in YAML config. Cap unit is the annotation item: chunks for
            retrieval, records for grounding / generation.
        calibration_min_submitted: Deployment-level overlap requirement for
            the calibration dataset. ``None`` disables calibration entirely
            (must be paired with ``calibration_fraction=0.0``). Inherits to
            workspaces/tasks unless they override it.
        calibration_partition_seed: Deterministic seed used to assign new
            records to calibration vs production buckets. Existing
            assignments are locked by the partition manifest; this only
            affects records not yet seen.
        locale: Deployment-level UI locale for Argilla dataset
            titles/questions/guidelines used when auto-creating datasets.
            Cascades to workspaces/tasks unless they carve out their own
            value in YAML.
        locale_catalog_dir: Directory of user-provided locale YAML files
            layered over the bundled catalogs (user wins on stem collision).
            Must exist if set. Falls back to YAML config.

    Returns:
        ImportResult with totals, per-dataset counts, partition counts, and
        validation errors.
    """
    raw = resolve_records(records, format=format)
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env={"argilla": {"api_url": os.environ.get("ARGILLA_API_URL")}} if os.environ.get("ARGILLA_API_URL") else None,
        overrides={
            "argilla": {"api_url": api_url},
            "dataset_id": dataset_id,
            "base_dir": base_dir,
            "calibration_fraction": calibration_fraction,
            "calibration_max_records": calibration_max_records,
            "calibration_min_submitted": calibration_min_submitted,
            "calibration_partition_seed": calibration_partition_seed,
            "locale": locale,
            "locale_catalog_dir": locale_catalog_dir,
        },
    )
    if settings.locale_catalog_dir is not None:
        register_catalog_dir(settings.locale_catalog_dir)

    api_key = api_key if isinstance(api_key, str) else resolve_api_key("argilla")
    client = resolve_argilla_client(settings.argilla.api_url, api_key)
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    paths = resolve_annotation_paths(workspace=workspace).ensure_dirs()
    import_paths = resolve_import_paths(workspace=workspace, dataset_id=settings.dataset_id).ensure_dirs()

    import_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    with error_log(paths.tool_root):
        validation = validate_records(raw)
        if validation.errors:
            logger.warning("Validation failed for %d of %d records", len(validation.errors), len(raw))

        manifest = load_partition_manifest(
            import_paths.partition_manifest,
            dataset_id=settings.dataset_id,
            partition_seed=settings.calibration_partition_seed,
        )
        if manifest.partition_seed != settings.calibration_partition_seed:
            logger.warning(
                "calibration_partition_seed=%d differs from manifest's stored seed=%d; "
                "using stored seed (existing assignments preserved, new records use stored seed)",
                settings.calibration_partition_seed,
                manifest.partition_seed,
            )

        assignments = assign_partitions(
            validation.valid,
            manifest=manifest,
            settings=settings,
            import_id=import_id,
        )

        calibration_count = count_calibration_per_task(assignments.values())
        total_count = _total_units_per_task(assignments.values())
        production_count = {task: total_count[task] - calibration_count[task] for task in Task}
        realised_fraction = {
            task: (calibration_count[task] / total_count[task]) if total_count[task] else 0.0 for task in Task
        }
        configured_fraction, configured_cap = _resolve_per_task_settings(settings)

        # Manifest is written only after fan-out succeeds. On failure, the
        # in-memory assignments are dropped; deterministic hashing ensures the
        # next attempt re-derives the same buckets without persisting state for
        # records that were never logged.
        dataset_counts = fan_out_records(client, validation.valid, settings, assignments=assignments)
        write_partition_manifest(import_paths.partition_manifest, manifest)
    logger.info(
        "Import complete: %d records across %d datasets. Per-task calibration: %s",
        len(raw),
        len(dataset_counts),
        ", ".join(
            f"{task.value}={calibration_count[task]} "
            f"(realised={realised_fraction[task]:.3f}, "
            f"configured={configured_fraction[task]:.3f}"
            f"{', cap=' + str(configured_cap[task]) if configured_cap[task] is not None else ''})"
            for task in Task
        ),
    )
    return ImportResult(
        total_records=len(raw),
        dataset_counts=dataset_counts,
        calibration_count=calibration_count,
        production_count=production_count,
        calibration_fraction=configured_fraction,
        realised_calibration_fraction=realised_fraction,
        calibration_max_records=configured_cap,
        errors=validation.errors,
    )


def _total_units_per_task(entries: Iterable[PartitionManifestEntry]) -> dict[Task, int]:
    """Per-task total annotation-unit count across entries.

    For retrieval, sums chunks; for grounding/generation, counts records that
    carry a flag for that task (in practice every record carries both, but
    guard with ``.get`` for forward-compat).
    """
    totals: dict[Task, int] = {task: 0 for task in Task}
    for entry in entries:
        for task in (Task.GROUNDING, Task.GENERATION):
            if task in entry.grounding_generation_calibration:
                totals[task] += 1
        totals[Task.RETRIEVAL] += len(entry.retrieval_chunk_calibration)
    return totals


def _resolve_per_task_settings(
    settings: AnnotationSettings,
) -> tuple[dict[Task, float], dict[Task, int | None]]:
    """Resolve per-task ``calibration_fraction`` and ``calibration_max_records``.

    Picks the first workspace that owns each task. Returns zero / None for any
    task missing from the workspaces topology, surfacing as zero realised
    fraction rather than erroring (matches pre-refactor behaviour for absent
    tasks).
    """
    fraction: dict[Task, float] = {}
    cap: dict[Task, int | None] = {}
    for ws_name, ws in settings.workspaces.items():
        for task in ws.tasks:
            if task in fraction:
                continue
            resolved = settings.resolved_task(ws_name, task)
            fraction[task] = resolved.calibration_fraction
            cap[task] = resolved.calibration_max_records
    for task in Task:
        fraction.setdefault(task, 0.0)
        cap.setdefault(task, None)
    return fraction, cap
