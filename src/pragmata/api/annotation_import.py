"""Annotation import API — thin orchestration over core/ implementation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import argilla as rg

from pragmata.core.annotation.record_builder import (
    RecordError,
    fan_out_records,
    validate_records,
)
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, load_config_file

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
    records: list[dict],
    *,
    workspace_prefix: str | object = UNSET,
    config_path: str | Path | object = UNSET,
) -> ImportResult:
    """Validate and fan out records to the three Argilla annotation datasets.

    Raw dicts are validated against the canonical import schema. Valid
    records produce entries in retrieval (one per chunk), grounding
    (one per pair), and generation (one per pair) datasets. Validation
    failures are collected in ImportResult.errors — invalid records are
    skipped, not raised.

    Record IDs are derived from content hashes for idempotent upsert.
    Datasets must already exist (call setup() first).

    Args:
        client: Connected Argilla client instance.
        records: Raw dicts conforming to the canonical import schema.
        workspace_prefix: Prefix used when the environment was created.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        ImportResult with totals, per-dataset counts, and validation errors.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(cast("str | Path", config_path)) if config_path is not UNSET else None,
        overrides={"workspace_prefix": workspace_prefix},
    )
    validation = validate_records(records)
    dataset_counts = fan_out_records(client, validation.valid, settings)
    return ImportResult(
        total_records=len(records),
        dataset_counts=dataset_counts,
        errors=validation.errors,
    )
