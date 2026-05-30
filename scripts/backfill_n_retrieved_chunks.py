r"""One-off migration: backfill n_retrieved_chunks metadata on existing retrieval records.

Records imported before the n_retrieved_chunks field existed do not carry K.
The new export-time completeness pass (panel_complete / n_annotated_chunks)
reads K from this metadata field, so for in-flight batches we need to
populate it on the live records.

JOIN
----
The source jsonl files (``~/pragmata-workspace/publikationsbot_output/*.jsonl``)
have NO ``record_uuid`` field and extra top-level keys (``query_id, domain,
topic, ...``) that ``QueryResponsePair(extra="forbid")`` rejects. The join
to live records goes through ``derive_record_uuid``: per source line, project
the canonical fields ``{query, answer, chunks, context_set, language}``,
build a ``QueryResponsePair``, hash it via ``derive_record_uuid`` - that
hash is the ``record_uuid`` the records were imported under, and K is
``len(pair.chunks)``.

SAFETY
------
- ``--dry-run`` is the DEFAULT. Pass ``--apply`` to actually write.
- Each write goes through ``upsert_record_metadata``, which fetches the
  record's current metadata and sends the FULL merged dict (Argilla v2.8.0
  metadata is REPLACE-not-merge).
- Property declaration uses ``ensure_metadata_property`` (idempotent
  additive create).
- VALIDATE ON A CLONED DATASET BEFORE RUNNING AGAINST LIVE.

USAGE
-----
    # dry-run (default): count what would change, no writes
    python scripts/backfill_n_retrieved_chunks.py \\
        ~/pragmata-workspace/publikationsbot_output/*.jsonl

    # apply for real (after dry-run + clone validation)
    python scripts/backfill_n_retrieved_chunks.py --apply \\
        ~/pragmata-workspace/publikationsbot_output/*.jsonl

Credentials resolve from ARGILLA_API_URL / ARGILLA_API_KEY env vars (or
the YAML config via ``--config``), same as the ``annotation`` CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import argilla as rg

from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.annotation.client import resolve_argilla_client
from pragmata.core.annotation.export_fetcher import resolve_task_purposes
from pragmata.core.annotation.metadata_ops import ensure_metadata_property, upsert_record_metadata
from pragmata.core.annotation.record_builder import derive_record_uuid
from pragmata.core.schemas.annotation_import import QueryResponsePair
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, load_config_file, resolve_api_key

logger = logging.getLogger("backfill_n_retrieved_chunks")

# Fields the import path uses to build a QueryResponsePair. Other top-level
# keys in the source jsonl (query_id, domain, topic, etc.) are dropped here
# because QueryResponsePair declares extra="forbid".
_CANONICAL_FIELDS = ("query", "answer", "chunks", "context_set", "language")


@dataclass(frozen=True)
class BackfillStats:
    """Counts from one backfill pass over a single retrieval dataset."""

    n_updated: int  # records whose metadata WOULD be (dry-run) or WAS (apply) written
    n_already_correct: int  # n_retrieved_chunks already matches K - no-op
    n_skipped_no_join: int  # record_uuid not in the K map (source line not found)
    n_skipped_orphan: int  # record has no record_uuid metadata


def load_k_map(jsonl_paths: list[Path]) -> dict[str, int]:
    """Read source jsonl(s) and build ``{record_uuid: K}``.

    Skips lines that can't form a valid QueryResponsePair (missing canonical
    fields, empty chunks, validation failure) so error / no-retrieval files
    in the same directory are harmless to include.
    """
    k_map: dict[str, int] = {}
    for path in jsonl_paths:
        if not path.exists():
            logger.warning("source file missing: %s", path)
            continue
        with path.open() as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d: invalid json (%s)", path, line_no, exc)
                    continue
                projected = {k: data[k] for k in _CANONICAL_FIELDS if k in data}
                if "chunks" not in projected or not projected["chunks"]:
                    continue
                try:
                    pair = QueryResponsePair.model_validate(projected)
                except Exception as exc:
                    logger.debug("%s:%d: not a valid pair (%s)", path, line_no, exc)
                    continue
                uuid = derive_record_uuid(pair)
                k = len(pair.chunks)
                if uuid in k_map and k_map[uuid] != k:
                    logger.warning("%s:%d: uuid=%s has conflicting K (%d vs %d)", path, line_no, uuid, k_map[uuid], k)
                k_map[uuid] = k
    return k_map


def backfill_dataset(
    dataset: rg.Dataset,
    k_map: dict[str, int],
    *,
    dry_run: bool,
) -> BackfillStats:
    """Backfill n_retrieved_chunks on one Argilla retrieval dataset."""
    if not dry_run:
        ensure_metadata_property(
            dataset,
            rg.IntegerMetadataProperty("n_retrieved_chunks", min=1, visible_for_annotators=False),
        )
    n_updated = 0
    n_already_correct = 0
    n_skipped_no_join = 0
    n_skipped_orphan = 0
    for record in dataset.records(with_responses=False):
        record_uuid = record.metadata.get("record_uuid", "")
        if not record_uuid:
            n_skipped_orphan += 1
            continue
        k = k_map.get(record_uuid)
        if k is None:
            n_skipped_no_join += 1
            continue
        current = record.metadata.get("n_retrieved_chunks")
        if current == k:
            n_already_correct += 1
            continue
        if dry_run:
            n_updated += 1
            continue
        upsert_record_metadata(dataset, record, {"n_retrieved_chunks": k})
        n_updated += 1
    return BackfillStats(
        n_updated=n_updated,
        n_already_correct=n_already_correct,
        n_skipped_no_join=n_skipped_no_join,
        n_skipped_orphan=n_skipped_orphan,
    )


def _resolve_settings(*, config: str | None, dataset_id: str | None, base_dir: str | None) -> AnnotationSettings:
    return AnnotationSettings.resolve(
        config=load_config_file(config) if config else None,
        env={"argilla": {"api_url": os.environ.get("ARGILLA_API_URL")}} if os.environ.get("ARGILLA_API_URL") else None,
        overrides={
            "argilla": {"api_url": UNSET},
            "dataset_id": UNSET if dataset_id is None else dataset_id,
            "base_dir": UNSET if base_dir is None else base_dir,
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl_paths", nargs="+", type=Path, help="Source jsonl files (publikationsbot output).")
    parser.add_argument("--apply", action="store_true", help="Actually write. Default is dry-run.")
    parser.add_argument("--config", default=None, help="Path to YAML config for annotation settings.")
    parser.add_argument("--dataset-id", default=None, help="Suffix scoping Argilla dataset names.")
    parser.add_argument("--base-dir", default=None, help="Workspace base directory.")
    parser.add_argument("--api-key", default=None, help="Argilla API key (or set ARGILLA_API_KEY).")
    parser.add_argument("--api-url", default=None, help="Argilla server URL (or set ARGILLA_API_URL).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dry_run = not args.apply
    if dry_run:
        logger.info("DRY-RUN mode (no writes). Pass --apply to actually backfill.")
    else:
        logger.warning("APPLY mode: writes will be made to live Argilla records.")

    k_map = load_k_map(args.jsonl_paths)
    logger.info("Loaded K map for %d record_uuid(s) from %d source file(s).", len(k_map), len(args.jsonl_paths))

    settings = _resolve_settings(config=args.config, dataset_id=args.dataset_id, base_dir=args.base_dir)
    api_url = args.api_url or settings.argilla.api_url
    api_key = args.api_key or resolve_api_key("argilla")
    client = resolve_argilla_client(api_url, api_key)

    workspace_name, purposes = resolve_task_purposes(settings, Task.RETRIEVAL)
    totals = BackfillStats(0, 0, 0, 0)
    for calibration in purposes:
        ds_name = dataset_name(Task.RETRIEVAL, calibration=calibration, dataset_id=settings.dataset_id)
        dataset = client.datasets(ds_name, workspace=workspace_name)
        if dataset is None:
            logger.info("Dataset %s not found - skipping.", ds_name)
            continue
        logger.info("Processing %s ...", ds_name)
        stats = backfill_dataset(dataset, k_map, dry_run=dry_run)
        logger.info(
            "%s: updated=%d already_correct=%d no_join=%d orphan=%d",
            ds_name,
            stats.n_updated,
            stats.n_already_correct,
            stats.n_skipped_no_join,
            stats.n_skipped_orphan,
        )
        totals = BackfillStats(
            n_updated=totals.n_updated + stats.n_updated,
            n_already_correct=totals.n_already_correct + stats.n_already_correct,
            n_skipped_no_join=totals.n_skipped_no_join + stats.n_skipped_no_join,
            n_skipped_orphan=totals.n_skipped_orphan + stats.n_skipped_orphan,
        )

    action = "would update" if dry_run else "updated"
    logger.info(
        "TOTAL: %s=%d already_correct=%d no_join=%d orphan=%d",
        action,
        totals.n_updated,
        totals.n_already_correct,
        totals.n_skipped_no_join,
        totals.n_skipped_orphan,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
