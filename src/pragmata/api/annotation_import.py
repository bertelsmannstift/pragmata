"""Annotation import API - thin orchestration over core/ implementation."""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.loaders import RecordInput, resolve_records
from pragmata.core.annotation.record_builder import (
    RecordError,
    assign_partitions,
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
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportResult:
    """Outcome of import_records(): counts per dataset and validation errors.

    Attributes:
        total_records: Number of raw dicts received as input.
        dataset_counts: Records submitted per dataset name.
        calibration_count: Records routed to the calibration dataset.
        production_count: Records routed to the production dataset.
        calibration_fraction: Effective fraction in force this run.
        errors: Per-record validation failures (index + detail).
    """

    total_records: int
    dataset_counts: dict[str, int]
    calibration_count: int = 0
    production_count: int = 0
    calibration_fraction: float = 0.0
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
    calibration_fraction: float = 0.1,
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
        calibration_fraction: Fraction of records routed to the calibration
            dataset for this import. Default 0.1 (industry standard for IAA
            pilots). Set to 0.0 for production-only batches; use a non-zero
            value when starting a calibration phase or re-calibrating.

    Returns:
        ImportResult with totals, per-dataset counts, partition counts, and
        validation errors.

    Raises:
        ValueError: ``calibration_fraction`` is outside [0.0, 1.0], or is
            > 0 when topology declares no calibration dataset for any task
            receiving records.
    """
    if not 0.0 <= calibration_fraction <= 1.0:
        raise ValueError(f"calibration_fraction must be in [0.0, 1.0]; got {calibration_fraction}")

    raw = resolve_records(records, format=format)
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, str | Path) else None,
        env={"argilla": {"api_url": os.environ.get("ARGILLA_API_URL")}} if os.environ.get("ARGILLA_API_URL") else None,
        overrides={
            "argilla": {"api_url": api_url},
            "dataset_id": dataset_id,
            "base_dir": base_dir,
        },
    )

    if calibration_fraction > 0.0:
        missing = [
            task.value
            for task_overlaps in settings.workspace_dataset_map.values()
            for task, overlap in task_overlaps.items()
            if overlap.calibration_min_submitted is None
        ]
        if missing:
            raise ValueError(
                f"calibration_fraction={calibration_fraction} > 0 but topology has no "
                f"calibration dataset for tasks: {sorted(set(missing))}. Either set "
                f"calibration_fraction=0.0 or enable calibration in workspace_dataset_map."
            )

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
            fraction=calibration_fraction,
            import_id=import_id,
        )
        write_partition_manifest(import_paths.partition_manifest, manifest)

        calibration_count = sum(1 for is_cal in assignments.values() if is_cal)
        production_count = len(assignments) - calibration_count

        dataset_counts = fan_out_records(client, validation.valid, settings, assignments=assignments)
    logger.info(
        "Import complete: %d records across %d datasets (calibration=%d, production=%d, fraction=%.3f)",
        len(raw),
        len(dataset_counts),
        calibration_count,
        production_count,
        calibration_fraction,
    )
    return ImportResult(
        total_records=len(raw),
        dataset_counts=dataset_counts,
        calibration_count=calibration_count,
        production_count=production_count,
        calibration_fraction=calibration_fraction,
        errors=validation.errors,
    )
