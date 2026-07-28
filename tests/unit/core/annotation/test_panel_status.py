"""Unit tests for live panel status (read-only, config-free)."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from pragmata.core.annotation.panel_status import (
    compute_panel_status,
    compute_task_progress,
    tag_partial_panels,
)

WS = "dom_retrieval"


def _record(
    *,
    record_id: str = "rec-x",
    record_uuid: str = "u1",
    chunk_id: str = "c1",
    n_retrieved_chunks: int | None = None,
    response_statuses: list[str] | None = None,
    needs_completion: str | None = None,
) -> MagicMock:
    metadata: dict[str, object] = {"record_uuid": record_uuid, "chunk_id": chunk_id}
    if n_retrieved_chunks is not None:
        metadata["n_retrieved_chunks"] = n_retrieved_chunks
    if needs_completion is not None:
        metadata["needs_completion"] = needs_completion
    rec = MagicMock()
    rec.id = record_id
    rec.metadata = metadata
    responses = []
    for s in response_statuses or []:
        r = MagicMock()
        r.status = s
        responses.append(r)
    rec.responses = responses
    return rec


def _dataset(
    name: str,
    records: list[MagicMock],
    *,
    workspace: str = WS,
    min_submitted: int = 1,
    progress: dict | None = None,
) -> MagicMock:
    """A config-free dataset handle: name + workspace + records + live min_submitted."""
    ds = MagicMock()
    ds.name = name
    ds.workspace.name = workspace
    ds.records.side_effect = lambda *a, **kw: iter(records)
    ds.progress.return_value = progress or {"total": 0, "completed": 0, "pending": 0}
    ds.settings.distribution.min_submitted = min_submitted
    return ds


def _client(datasets: list[MagicMock]) -> MagicMock:
    """Config-free client: ``client.datasets`` is the iterable of dataset handles."""
    client = MagicMock()
    client.datasets = list(datasets)
    return client


# ---------------------------------------------------------------------------
# compute_panel_status
# ---------------------------------------------------------------------------


class TestComputePanelStatus:
    def test_headline_aggregated_from_progress(self) -> None:
        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(3)]
        report = compute_panel_status(
            _client([_dataset("retrieval_production", records, progress={"total": 10, "completed": 4, "pending": 6})])
        )
        assert report.headline.total == 10
        assert report.headline.completed == 4
        assert report.headline.pending == 6

    def test_panel_complete_when_every_chunk_has_terminal(self) -> None:
        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        report = compute_panel_status(_client([_dataset("retrieval_production", records)]))
        panel = report.panels[(WS, "u1")]
        assert panel.panel_complete is True
        assert panel.k_records == 5
        assert panel.n_terminal == 5

    def test_label_vs_overlap_diverges_in_calibration(self) -> None:
        """One submitted response per cal chunk → panel_complete but NOT overlap_satisfied (cal needs 3)."""
        cal_records = [
            _record(record_uuid="u-cal", chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(3)
        ]
        report = compute_panel_status(
            _client(
                [
                    _dataset("retrieval_production", []),
                    _dataset("retrieval_calibration", cal_records, min_submitted=3),
                ]
            )
        )
        panel = report.panels[(WS, "u-cal")]
        assert panel.panel_complete is True
        assert panel.overlap_satisfied is False

    def test_overlap_satisfied_when_three_submitted_in_calibration(self) -> None:
        cal_records = [
            _record(record_uuid="u-cal", chunk_id=f"c{i}", response_statuses=["submitted", "submitted", "submitted"])
            for i in range(3)
        ]
        report = compute_panel_status(_client([_dataset("retrieval_calibration", cal_records, min_submitted=3)]))
        panel = report.panels[(WS, "u-cal")]
        assert panel.panel_complete is True
        assert panel.overlap_satisfied is True

    def test_min_submitted_read_from_live_dataset_settings(self) -> None:
        """overlap_satisfied uses each dataset's live min_submitted, not config."""
        # 2 submitted on the only chunk; dataset says it needs 3 → not satisfied.
        records = [_record(record_uuid="u1", chunk_id="c1", response_statuses=["submitted", "submitted"])]
        report = compute_panel_status(_client([_dataset("retrieval_calibration", records, min_submitted=3)]))
        assert report.panels[(WS, "u1")].overlap_satisfied is False

    def test_integrity_warning_when_records_mismatch_metadata_k(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [
            _record(chunk_id="c1", n_retrieved_chunks=5, response_statuses=["submitted"]),
            _record(chunk_id="c2", n_retrieved_chunks=5, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.panel_status"):
            report = compute_panel_status(_client([_dataset("retrieval_production", records)]))
        panel = report.panels[(WS, "u1")]
        assert panel.integrity_ok is False
        assert report.n_integrity_warnings == 1
        assert any("integrity warning" in r.message for r in caplog.records)

    def test_integrity_ok_when_metadata_absent(self) -> None:
        """No metadata K → integrity_ok=True (skips the check). Pre-backfill state."""
        records = [_record(chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"])]
        report = compute_panel_status(_client([_dataset("retrieval_production", records)]))
        assert report.panels[(WS, "u1")].integrity_ok is True

    def test_orphan_record_excluded(self) -> None:
        records = [
            _record(record_uuid="", chunk_id="orphan"),
            _record(record_uuid="u1", chunk_id="c1", response_statuses=["submitted"]),
        ]
        report = compute_panel_status(_client([_dataset("retrieval_production", records)]))
        assert (WS, "") not in report.panels
        assert report.n_orphans_skipped == 1

    def test_walks_prod_and_cal(self) -> None:
        prod = [_record(record_uuid="u-prod", chunk_id="c1", response_statuses=["submitted"])]
        cal = [_record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted"])]
        report = compute_panel_status(
            _client([_dataset("retrieval_production", prod), _dataset("retrieval_calibration", cal, min_submitted=3)])
        )
        assert set(report.panels) == {(WS, "u-prod"), (WS, "u-cal")}

    def test_same_uuid_in_two_workspaces_not_fused(self) -> None:
        """Multi-domain walk: identical record_uuid in two workspaces stays two panels."""
        d1 = _dataset(
            "retrieval_production",
            [_record(record_uuid="u1", chunk_id="c1", response_statuses=["submitted"])],
            workspace="dom1_retrieval",
        )
        d2 = _dataset(
            "retrieval_production",
            [_record(record_uuid="u1", chunk_id="c1", response_statuses=[])],
            workspace="dom2_retrieval",
        )
        report = compute_panel_status(_client([d1, d2]))
        assert set(report.panels) == {("dom1_retrieval", "u1"), ("dom2_retrieval", "u1")}
        assert report.panels[("dom1_retrieval", "u1")].panel_complete is True
        assert report.panels[("dom2_retrieval", "u1")].panel_complete is False

    def test_non_retrieval_datasets_are_skipped(self) -> None:
        """Default task='retrieval' selects by name prefix; other tasks are ignored."""
        report = compute_panel_status(
            _client(
                [
                    _dataset("generation_production", [_record(record_uuid="g1", chunk_id="c1")]),
                    _dataset(
                        "retrieval_production",
                        [_record(record_uuid="u1", chunk_id="c1", response_statuses=["submitted"])],
                    ),
                ]
            )
        )
        assert set(report.panels) == {(WS, "u1")}


class TestComputePanelStatusEdgeCases:
    def test_overlap_aggregates_submissions_across_duplicate_chunk_records(self) -> None:
        """If a single chunk_id has two records (rare anomaly), submissions sum across them."""
        rec_a = _record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted", "submitted"])
        rec_b = _record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted", "submitted"])
        report = compute_panel_status(_client([_dataset("retrieval_calibration", [rec_a, rec_b], min_submitted=3)]))
        panel = report.panels[(WS, "u-cal")]
        assert panel.overlap_satisfied is True

    def test_warns_when_all_panels_have_unknown_k(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pre-backfill: all records missing n_retrieved_chunks → distinct warning."""
        records = [
            _record(record_uuid="u1", chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"]),
            _record(record_uuid="u2", chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.panel_status"):
            compute_panel_status(_client([_dataset("retrieval_production", records)]))
        assert any("unknown K" in r.message for r in caplog.records)


class TestComputeTaskProgress:
    def test_groups_by_task_workspace_dataset(self) -> None:
        client = _client(
            [
                _dataset(
                    "retrieval_production",
                    [],
                    workspace="A_retrieval",
                    progress={"total": 100, "completed": 10, "pending": 90},
                ),
                _dataset(
                    "grounding_production",
                    [],
                    workspace="A_grounding",
                    progress={"total": 20, "completed": 5, "pending": 15},
                ),
                _dataset(
                    "generation_production",
                    [],
                    workspace="A_generation",
                    progress={"total": 30, "completed": 9, "pending": 21},
                ),
            ]
        )
        pr = compute_task_progress(client)
        # grand total across all tasks
        assert (pr.grand.total, pr.grand.completed, pr.grand.pending) == (150, 24, 126)
        # by-task ordered retrieval -> grounding -> generation
        assert [r.task for r in pr.by_task] == ["retrieval", "grounding", "generation"]
        assert pr.by_task[0].total == 100
        # by-workspace and by-dataset carry the same numbers, labelled differently
        assert {r.label for r in pr.by_workspace} == {"A_retrieval", "A_grounding", "A_generation"}
        assert any(r.label == "A_retrieval/retrieval_production" for r in pr.by_dataset)

    def test_all_tasks_present_even_with_zero_progress(self) -> None:
        client = _client(
            [
                _dataset(
                    "grounding_production",
                    [],
                    workspace="B_grounding",
                    progress={"total": 5, "completed": 0, "pending": 5},
                )
            ]
        )
        pr = compute_task_progress(client)
        assert [r.task for r in pr.by_task] == ["grounding"]
        assert pr.grand.completed == 0


class TestTagPartialPanels:
    def test_tags_unresolved_chunks_of_partial_panel(self) -> None:
        # K=3: one chunk submitted, two unstarted → partial → tag the two.
        recs = [
            _record(chunk_id="c1", response_statuses=["submitted"], record_id="r1"),
            _record(chunk_id="c2", response_statuses=[], record_id="r2"),
            _record(chunk_id="c3", response_statuses=[], record_id="r3"),
        ]
        ds = _dataset("retrieval_production", recs)
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([ds]))
        assert (result.n_tagged, result.n_cleared, result.n_already_tagged) == (2, 0, 0)
        (logged,) = ds.records.log.call_args.args
        assert {r.id for r in logged} == {"r2", "r3"}

    def test_fully_unstarted_panel_not_tagged(self) -> None:
        recs = [_record(chunk_id=f"c{i}", response_statuses=[], record_id=f"r{i}") for i in range(3)]
        ds = _dataset("retrieval_production", recs)
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([ds]))
        assert result.n_tagged == 0
        ds.records.log.assert_not_called()

    def test_complete_panel_not_tagged(self) -> None:
        recs = [_record(chunk_id=f"c{i}", response_statuses=["submitted"], record_id=f"r{i}") for i in range(3)]
        ds = _dataset("retrieval_production", recs)
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([ds]))
        assert result.n_tagged == 0

    def test_clears_stale_tag_on_non_partial_panel(self) -> None:
        # Fully-unstarted panel whose chunk carries a stale needs_completion tag
        # (e.g. from the old broad predicate) → cleared, never re-tagged.
        recs = [
            _record(chunk_id="c1", response_statuses=[], record_id="r1", needs_completion="true"),
            _record(chunk_id="c2", response_statuses=[], record_id="r2"),
        ]
        ds = _dataset("retrieval_production", recs)
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([ds]))
        assert (result.n_tagged, result.n_cleared) == (0, 1)
        (logged,) = ds.records.log.call_args.args
        assert {r.id for r in logged} == {"r1"}
        assert "needs_completion" not in logged[0].metadata

    def test_already_tagged_is_idempotent(self) -> None:
        recs = [
            _record(chunk_id="c1", response_statuses=["submitted"], record_id="r1"),
            _record(chunk_id="c2", response_statuses=[], record_id="r2", needs_completion="true"),
        ]
        ds = _dataset("retrieval_production", recs)
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([ds]))
        assert (result.n_tagged, result.n_cleared, result.n_already_tagged) == (0, 0, 1)
        ds.records.log.assert_not_called()

    def test_split_panel_across_prod_and_cal(self) -> None:
        # Per-item calibration: the only annotated chunk is in cal; prod chunks
        # unstarted. Panel is partial ACROSS datasets → tag the prod chunks.
        prod = _dataset(
            "retrieval_production",
            [
                _record(chunk_id="c2", response_statuses=[], record_id="p2"),
                _record(chunk_id="c3", response_statuses=[], record_id="p3"),
            ],
        )
        cal = _dataset(
            "retrieval_calibration",
            [_record(chunk_id="c1", response_statuses=["submitted"], record_id="cal1")],
            min_submitted=3,
        )
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([prod, cal]))
        assert result.n_tagged == 2
        (logged,) = prod.records.log.call_args.args
        assert {r.id for r in logged} == {"p2", "p3"}
        cal.records.log.assert_not_called()

    def test_writes_routed_per_workspace_when_names_collide(self) -> None:
        d1 = _dataset(
            "retrieval_production",
            [
                _record(record_uuid="uA", chunk_id="c1", response_statuses=["submitted"], record_id="a1"),
                _record(record_uuid="uA", chunk_id="c2", response_statuses=[], record_id="a2"),
            ],
            workspace="dom1_retrieval",
        )
        d2 = _dataset(
            "retrieval_production",
            [
                _record(record_uuid="uB", chunk_id="c1", response_statuses=["submitted"], record_id="b1"),
                _record(record_uuid="uB", chunk_id="c2", response_statuses=[], record_id="b2"),
            ],
            workspace="dom2_retrieval",
        )
        with patch("argilla.TermsMetadataProperty"):
            result = tag_partial_panels(_client([d1, d2]))
        assert result.n_tagged == 2
        (l1,) = d1.records.log.call_args.args
        (l2,) = d2.records.log.call_args.args
        assert {r.id for r in l1} == {"a2"}
        assert {r.id for r in l2} == {"b2"}
