"""Annotation import API — thin orchestration over core/ implementation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import argilla as rg

from pragmata.api._error_log import error_log
from pragmata.core.annotation.loaders import RecordInput, resolve_records
from pragmata.core.annotation.record_builder import (
    RecordError,
    fan_out_records,
    validate_records,
)
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportResult:
    """Outcome of import_records(): counts per dataset and validation errors.

    Attributes:
        total_records: Number of raw dicts received as input.
        dataset_counts: Records submitted per dataset name.
        errors: Per-record validation failures (index + detail).
    """

    total_records: int
    dataset_counts: dict[str, int]
    errors: list[RecordError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_records(
    client: rg.Argilla,
    records: RecordInput,
    *,
    format: str = "auto",
    workspace_prefix: str | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> ImportResult:
    """Validate and fan out records to the three Argilla annotation datasets.

    Accepts raw dicts, file paths (JSON, JSONL, CSV), HuggingFace Datasets,
    or pandas DataFrames. File format is detected by extension or overridden
    via the format kwarg. All inputs are resolved to list[dict] before
    validation against the canonical import schema.

    Valid records produce entries in retrieval (one per chunk), grounding
    (one per pair), and generation (one per pair) datasets. Validation
    failures are collected in ImportResult.errors — invalid records are
    skipped, not raised.

    Record IDs are derived from content hashes for idempotent upsert.
    Datasets must already exist (call setup() first).

    Args:
        client: Connected Argilla client instance.
        records: Input data — list[dict], file path (str/Path), HF Dataset,
            or pandas DataFrame.
        format: File format override — 'auto' (default), 'json', 'jsonl',
            or 'csv'. Only used for str/Path inputs.
        workspace_prefix: Prefix used when the environment was created.
        base_dir: Workspace base directory. Defaults to cwd.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ImportResult with totals, per-dataset counts, and validation errors.
    """
    raw = resolve_records(records, format=format)
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix, "base_dir": base_dir},
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    with error_log(workspace.tool_root("annotation")):
        validation = validate_records(raw)
        if validation.errors:
            logger.warning("Validation failed for %d of %d records", len(validation.errors), len(raw))
        dataset_counts = fan_out_records(client, validation.valid, settings)
    logger.info("Import complete: %d records across %d datasets", len(raw), len(dataset_counts))
    return ImportResult(
        total_records=len(raw),
        dataset_counts=dataset_counts,
        errors=validation.errors,
    )
