"""Find (and optionally tag) the records still needed to complete a bundle.

A *bundle* is the set of Argilla records sharing a ``record_uuid`` - one
underlying query-response. Only *partially*-complete bundles are reported: at
least one record annotated but not the whole bundle, so its UNRESOLVED records
(no submitted or discarded response yet) are what's left to finish it. Fully
unstarted bundles - and single-record generation/grounding bundles, which can
never be partial - are excluded.

Generic across tasks: a retrieval bundle fans out to K chunk-records while a
generation/grounding bundle is a single record, but the same predicate applies.
No config needed - datasets are selected by workspace/task name.

Read-only by default. With ``tag=True`` it stamps the shared ``needs_completion``
advisory tag on the unresolved records (and clears stale tags), so annotators
can filter straight to them in the Argilla UI - same predicate and write path
as ``status --tag-incomplete``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pragmata.core.annotation.metadata_ops import build_metadata_upsert, ensure_metadata_property
from pragmata.core.annotation.panel_status import (
    NEEDS_COMPLETION_KEY,
    NEEDS_COMPLETION_VALUE,
    _has_needs_completion_tag,
)

if TYPE_CHECKING:
    import argilla as rg

_TERMINAL_STATUSES = frozenset({"submitted", "discarded"})
_TASK_SUFFIXES = ("retrieval", "grounding", "generation")


def workspace_domain(workspace: str) -> str:
    """Domain label for a workspace: its name without the trailing ``_<task>``."""
    for task in _TASK_SUFFIXES:
        if workspace.endswith(f"_{task}"):
            return workspace[: -(len(task) + 1)]
    return workspace


@dataclass(frozen=True)
class IncompleteBundle:
    """One bundle (``record_uuid``) that is not yet complete."""

    workspace: str
    dataset: str
    record_uuid: str
    n_records: int  # records in the bundle (live)
    n_submitted: int  # records with >=1 submitted response
    missing_record_ids: list[str]  # unresolved records that still need annotating
    missing_chunk_ids: list[str]  # their chunk_id metadata ("" for non-retrieval tasks)


@dataclass(frozen=True)
class IncompleteReport:
    """Incomplete bundles in scope, plus tag-write counts when ``tagged``."""

    bundles: list[IncompleteBundle]
    tagged: bool = False
    n_tagged: int = 0
    n_cleared: int = 0
    n_already_tagged: int = 0

    @property
    def n_bundles(self) -> int:
        return len(self.bundles)

    @property
    def n_records(self) -> int:
        return sum(len(b.missing_record_ids) for b in self.bundles)

    @property
    def n_domains(self) -> int:
        return len({workspace_domain(b.workspace) for b in self.bundles})

    @property
    def tasks(self) -> list[str]:
        return sorted({b.dataset.split("_", 1)[0] for b in self.bundles})


def _n_submitted(record: rg.Record) -> int:
    return sum(1 for r in (record.responses or []) if r.status == "submitted")


def _has_terminal(record: rg.Record) -> bool:
    return any(r.status in _TERMINAL_STATUSES for r in (record.responses or []))


def _select_datasets(client: rg.Argilla, workspace: str | None, task: str | None) -> Iterator[rg.Dataset]:
    for ds in client.datasets:
        if workspace is not None and ds.workspace.name != workspace:
            continue
        if task is not None and ds.name != task and not ds.name.startswith(f"{task}_"):
            continue
        yield ds


def find_incomplete(
    client: rg.Argilla,
    *,
    workspace: str | None = None,
    task: str | None = None,
    tag: bool = False,
) -> IncompleteReport:
    """List incomplete bundles in scope; with ``tag=True`` also write the tags.

    ``workspace``/``task`` are exact name filters (``task`` matches the dataset
    name prefix, e.g. ``retrieval``); ``None`` means no filter on that axis.
    """
    import argilla as rg  # runtime import (module keeps argilla TYPE_CHECKING-only)

    bundles: list[IncompleteBundle] = []
    n_tagged = n_cleared = n_already = 0

    for ds in _select_datasets(client, workspace, task):
        groups: dict[str, list[rg.Record]] = {}
        for rec in ds.records(with_responses=True):
            uuid = rec.metadata.get("record_uuid", "")
            if uuid:
                groups.setdefault(uuid, []).append(rec)

        if tag and groups:
            ensure_metadata_property(
                ds, rg.TermsMetadataProperty(NEEDS_COMPLETION_KEY, visible_for_annotators=True)
            )

        pending: list[rg.Record] = []
        for uuid, group in groups.items():
            n_started = sum(1 for rec in group if _n_submitted(rec) >= 1)
            # PARTIAL: at least one record annotated but not the whole bundle.
            # Single-record bundles (generation/grounding) can never be partial.
            is_partial = 0 < n_started < len(group)
            unresolved = [rec for rec in group if not _has_terminal(rec)]
            if is_partial and unresolved:
                bundles.append(
                    IncompleteBundle(
                        workspace=ds.workspace.name,
                        dataset=ds.name,
                        record_uuid=uuid,
                        n_records=len(group),
                        n_submitted=n_started,
                        missing_record_ids=[str(rec.id) for rec in unresolved],
                        missing_chunk_ids=[rec.metadata.get("chunk_id", "") for rec in unresolved],
                    )
                )
            if not tag:
                continue
            # Tag the unresolved records of partial bundles; idempotently clear
            # tags that no longer apply (same write path as --tag-incomplete).
            for rec in group:
                should_tag = is_partial and (not _has_terminal(rec))
                has_tag = _has_needs_completion_tag(rec)
                if should_tag and has_tag:
                    n_already += 1
                elif should_tag:
                    upsert = build_metadata_upsert(rec, {NEEDS_COMPLETION_KEY: NEEDS_COMPLETION_VALUE})
                    if upsert is not None:
                        pending.append(upsert)
                        n_tagged += 1
                elif has_tag:
                    upsert = build_metadata_upsert(rec, {}, remove_keys=[NEEDS_COMPLETION_KEY])
                    if upsert is not None:
                        pending.append(upsert)
                        n_cleared += 1

        if tag and pending:
            ds.records.log(pending)

    bundles.sort(key=lambda b: (b.workspace, b.dataset, b.record_uuid))
    return IncompleteReport(
        bundles=bundles,
        tagged=tag,
        n_tagged=n_tagged,
        n_cleared=n_cleared,
        n_already_tagged=n_already,
    )
