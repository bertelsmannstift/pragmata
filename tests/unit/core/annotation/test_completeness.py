"""Unit tests for retrieval panel completeness."""

import logging
from unittest.mock import MagicMock

import pytest

from pragmata.core.annotation.completeness import (
    PanelCompleteness,
    compute_completeness,
    k_bucket,
)


def _make_record(
    *,
    record_uuid: str = "uuid-a",
    chunk_id: str = "c1",
    n_retrieved_chunks: int | None = 5,
    response_statuses: list[str] | None = None,
) -> MagicMock:
    """A mock Argilla record. ``response_statuses=None`` means a record with no responses."""
    metadata: dict[str, object] = {"record_uuid": record_uuid, "chunk_id": chunk_id}
    if n_retrieved_chunks is not None:
        metadata["n_retrieved_chunks"] = n_retrieved_chunks
    rec = MagicMock()
    rec.metadata = metadata
    responses = []
    for status in response_statuses or []:
        r = MagicMock()
        r.status = status
        responses.append(r)
    rec.responses = responses
    return rec


def _settings(*, with_calibration: bool = False) -> MagicMock:
    """AnnotationSettings stub for compute_completeness."""
    settings = MagicMock()
    settings.dataset_id = ""
    ws_settings = MagicMock()
    ws_settings.tasks = ["retrieval"]
    settings.workspaces = {"ws1": ws_settings}
    resolved = MagicMock()
    resolved.calibration_min_submitted = 3 if with_calibration else None
    settings.resolved_task.return_value = resolved
    return settings


def _client(records_by_dataset: dict[str, list[MagicMock]]) -> MagicMock:
    """Mock client.datasets(name) returns a dataset whose .records(...) yields the given records.

    Each call to dataset.records(...) returns a FRESH iterator (single dataset
    can be fetched multiple times in one run).
    """
    client = MagicMock()

    def _datasets(name: str, workspace: str | None = None):
        records = records_by_dataset.get(name)
        if records is None:
            return None
        ds = MagicMock()
        ds.records.side_effect = lambda *a, **kw: iter(records)
        return ds

    client.datasets.side_effect = _datasets
    return client


# ---------------------------------------------------------------------------
# k_bucket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [(0, "k_lt_5"), (1, "k_lt_5"), (4, "k_lt_5"), (5, "k_eq_5"), (6, "k_gt_5"), (20, "k_gt_5")],
)
def test_k_bucket(k: int, expected: str) -> None:
    assert k_bucket(k) == expected


# ---------------------------------------------------------------------------
# compute_completeness — core semantics
# ---------------------------------------------------------------------------


class TestComputeCompleteness:
    def test_panel_complete_when_all_chunks_have_submitted_response(self) -> None:
        records = [_make_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        panel = report.by_uuid["uuid-a"]
        assert panel == PanelCompleteness(
            record_uuid="uuid-a",
            k=5,
            n_annotated_chunks=5,
            n_discarded_chunks=0,
            panel_complete=True,
            n_records_seen=5,
        )
        assert report.summary.n_panels == 1
        assert report.summary.n_complete == 1
        assert report.summary.fraction_complete == 1.0

    def test_discarded_counts_as_covered(self) -> None:
        """A chunk with only a discarded response is metric-covered, not a hole.

        Computed independent of include_discarded: the export may omit the
        discarded row entirely, but the panel is still complete.
        """
        records = [
            _make_record(chunk_id="c1", response_statuses=["submitted"]),
            _make_record(chunk_id="c2", response_statuses=["submitted"]),
            _make_record(chunk_id="c3", response_statuses=["discarded"]),
            _make_record(chunk_id="c4", response_statuses=["submitted"]),
            _make_record(chunk_id="c5", response_statuses=["discarded"]),
        ]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert report.by_uuid["uuid-a"].panel_complete is True
        assert report.by_uuid["uuid-a"].n_annotated_chunks == 5

    def test_partial_panel_not_complete(self) -> None:
        records = [
            _make_record(chunk_id="c1", response_statuses=["submitted"]),
            _make_record(chunk_id="c2", response_statuses=["submitted"]),
            _make_record(chunk_id="c3", response_statuses=[]),  # no responses yet
            _make_record(chunk_id="c4", response_statuses=[]),
            _make_record(chunk_id="c5", response_statuses=[]),
        ]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        panel = report.by_uuid["uuid-a"]
        assert panel.panel_complete is False
        assert panel.n_annotated_chunks == 2
        assert panel.k == 5

    def test_draft_only_response_does_not_count(self) -> None:
        """Only submitted/discarded count as terminal; draft does not."""
        records = [_make_record(chunk_id=f"c{i}", response_statuses=["draft"]) for i in range(5)]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert report.by_uuid["uuid-a"].n_annotated_chunks == 0
        assert report.by_uuid["uuid-a"].panel_complete is False

    def test_distinct_by_chunk_id(self) -> None:
        """Duplicate records for the same chunk_id count as one chunk."""
        records = [
            _make_record(chunk_id="c1", response_statuses=["submitted"]),
            _make_record(chunk_id="c1", response_statuses=["submitted"]),  # duplicate
            _make_record(chunk_id="c2", response_statuses=["submitted"]),
        ]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert report.by_uuid["uuid-a"].n_annotated_chunks == 2
        assert report.by_uuid["uuid-a"].n_records_seen == 2

    def test_orphan_record_skipped_and_counted(self) -> None:
        """Records with empty record_uuid are excluded and counted in n_orphans_skipped."""
        records = [
            _make_record(record_uuid="", chunk_id="orphan"),
            _make_record(record_uuid="uuid-a", chunk_id="c1", response_statuses=["submitted"]),
        ]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert "" not in report.by_uuid
        assert "uuid-a" in report.by_uuid
        assert report.summary.n_orphans_skipped == 1

    def test_integrity_warning_when_records_seen_not_equal_k(self, caplog: pytest.LogCaptureFixture) -> None:
        """Saw fewer (or more) chunk-records than K → integrity warning logged + counter."""
        records = [
            _make_record(chunk_id="c1", n_retrieved_chunks=5, response_statuses=["submitted"]),
            _make_record(chunk_id="c2", n_retrieved_chunks=5, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.completeness"):
            report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert report.summary.n_integrity_warnings == 1
        assert any("integrity warning" in r.message for r in caplog.records)

    def test_inconsistent_k_across_records_uses_max_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [
            _make_record(chunk_id="c1", n_retrieved_chunks=5, response_statuses=["submitted"]),
            _make_record(chunk_id="c2", n_retrieved_chunks=7, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.completeness"):
            report = compute_completeness(_client({"retrieval_production": records}), _settings())
        assert report.by_uuid["uuid-a"].k == 7
        assert any("inconsistent n_retrieved_chunks" in r.message for r in caplog.records)

    def test_k_zero_panel_is_not_complete(self) -> None:
        """Missing/zero n_retrieved_chunks metadata never produces a complete panel."""
        records = [_make_record(chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"])]
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        panel = report.by_uuid["uuid-a"]
        assert panel.k == 0
        assert panel.panel_complete is False

    def test_calibration_dataset_walked_when_enabled(self) -> None:
        """When calibration is declared, both prod and cal datasets are fetched."""
        prod_records = [
            _make_record(record_uuid="u-prod", chunk_id="c1", response_statuses=["submitted"], n_retrieved_chunks=1)
        ]
        cal_records = [
            _make_record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted"], n_retrieved_chunks=1)
        ]
        client = _client({"retrieval_production": prod_records, "retrieval_calibration": cal_records})
        report = compute_completeness(client, _settings(with_calibration=True))
        assert set(report.by_uuid.keys()) == {"u-prod", "u-cal"}
        assert report.summary.n_panels == 2
        assert report.summary.n_complete == 2

    def test_calibration_skipped_when_topology_disables(self) -> None:
        """Without calibration_min_submitted, the cal dataset is not fetched."""
        prod_records = [
            _make_record(record_uuid="u-prod", chunk_id="c1", response_statuses=["submitted"], n_retrieved_chunks=1)
        ]
        cal_records = [_make_record(record_uuid="u-cal")]
        client = _client({"retrieval_production": prod_records, "retrieval_calibration": cal_records})
        report = compute_completeness(client, _settings(with_calibration=False))
        assert "u-cal" not in report.by_uuid
        assert "u-prod" in report.by_uuid

    def test_missing_dataset_is_skipped_silently(self) -> None:
        """A None dataset (e.g. cal not yet created) doesn't break the walk."""
        client = _client({})  # no datasets
        report = compute_completeness(client, _settings())
        assert report.by_uuid == {}
        assert report.summary.n_panels == 0
        assert report.summary.fraction_complete == 0.0

    def test_by_k_bucket_cross_tab(self) -> None:
        """Panels are bucketed by K (<5, =5, >5) with per-bucket complete counts."""

        def panel(uuid: str, k: int, n_terminal: int) -> list[MagicMock]:
            return [
                _make_record(
                    record_uuid=uuid,
                    chunk_id=f"{uuid}-c{i}",
                    n_retrieved_chunks=k,
                    response_statuses=["submitted"] if i < n_terminal else [],
                )
                for i in range(k)
            ]

        records = (
            panel("small-complete", k=3, n_terminal=3)
            + panel("small-partial", k=3, n_terminal=1)
            + panel("five-complete", k=5, n_terminal=5)
            + panel("big-partial", k=7, n_terminal=4)
        )
        report = compute_completeness(_client({"retrieval_production": records}), _settings())
        buckets = report.summary.by_k_bucket
        assert buckets["k_lt_5"].n_panels == 2
        assert buckets["k_lt_5"].n_complete == 1
        assert buckets["k_eq_5"].n_panels == 1
        assert buckets["k_eq_5"].n_complete == 1
        assert buckets["k_gt_5"].n_panels == 1
        assert buckets["k_gt_5"].n_complete == 0
        assert report.summary.n_panels == 4
        assert report.summary.n_complete == 2
        assert report.summary.fraction_complete == pytest.approx(0.5)
