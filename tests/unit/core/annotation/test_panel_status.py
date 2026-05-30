"""Unit tests for live panel status + --tag-incomplete write."""

import logging
from unittest.mock import MagicMock

import pytest

from pragmata.core.annotation.panel_status import (
    NEEDS_COMPLETION_KEY,
    NEEDS_COMPLETION_VALUE,
    compute_panel_status,
    tag_incomplete_chunks,
)


def _record(
    *,
    record_id: str = "rec-x",
    record_uuid: str = "u1",
    chunk_id: str = "c1",
    n_retrieved_chunks: int | None = None,
    response_statuses: list[str] | None = None,
    needs_completion_tag: bool = False,
) -> MagicMock:
    metadata: dict[str, object] = {"record_uuid": record_uuid, "chunk_id": chunk_id}
    if n_retrieved_chunks is not None:
        metadata["n_retrieved_chunks"] = n_retrieved_chunks
    if needs_completion_tag:
        metadata[NEEDS_COMPLETION_KEY] = NEEDS_COMPLETION_VALUE
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


def _settings(
    *,
    with_calibration: bool = False,
    production_min_submitted: int = 1,
    calibration_min_submitted: int | None = 3,
) -> MagicMock:
    settings = MagicMock()
    settings.dataset_id = ""
    ws_settings = MagicMock()
    ws_settings.tasks = ["retrieval"]
    settings.workspaces = {"ws1": ws_settings}
    resolved = MagicMock()
    resolved.production_min_submitted = production_min_submitted
    resolved.calibration_min_submitted = calibration_min_submitted if with_calibration else None
    settings.resolved_task.return_value = resolved
    return settings


def _client(records_by_dataset: dict[str, list[MagicMock]], progress: dict | None = None) -> MagicMock:
    """Build a mock client where each dataset name maps to its records (re-iterable per call)."""
    client = MagicMock()
    p = progress or {"total": 0, "completed": 0, "pending": 0}

    def _datasets(name: str, workspace: str | None = None):
        records = records_by_dataset.get(name)
        if records is None:
            return None
        ds = MagicMock()
        ds.name = name
        ds.records.side_effect = lambda *a, **kw: iter(records)
        ds.progress.return_value = p
        # Settings.metadata gettable as absent (so ensure_metadata_property adds the tag prop)
        ds.settings.metadata.__getitem__ = MagicMock(return_value=None)
        return ds

    client.datasets.side_effect = _datasets
    return client


# ---------------------------------------------------------------------------
# compute_panel_status
# ---------------------------------------------------------------------------


class TestComputePanelStatus:
    def test_headline_aggregated_from_progress(self) -> None:
        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(3)]
        report = compute_panel_status(
            _client({"retrieval_production": records}, progress={"total": 10, "completed": 4, "pending": 6}),
            _settings(),
        )
        assert report.headline.total == 10
        assert report.headline.completed == 4
        assert report.headline.pending == 6

    def test_panel_complete_when_every_chunk_has_terminal(self) -> None:
        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        report = compute_panel_status(_client({"retrieval_production": records}), _settings())
        panel = report.panels["u1"]
        assert panel.panel_complete is True
        assert panel.k_records == 5
        assert panel.n_terminal == 5

    def test_label_vs_distribution_diverges_in_calibration(self) -> None:
        """One submitted response per cal chunk → panel_complete but NOT distribution_satisfied (cal needs 3)."""
        cal_records = [
            _record(record_uuid="u-cal", chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(3)
        ]
        report = compute_panel_status(
            _client({"retrieval_production": [], "retrieval_calibration": cal_records}),
            _settings(with_calibration=True, calibration_min_submitted=3),
        )
        panel = report.panels["u-cal"]
        assert panel.panel_complete is True
        assert panel.distribution_satisfied is False

    def test_distribution_satisfied_when_three_submitted_in_calibration(self) -> None:
        cal_records = [
            _record(record_uuid="u-cal", chunk_id=f"c{i}", response_statuses=["submitted", "submitted", "submitted"])
            for i in range(3)
        ]
        report = compute_panel_status(
            _client({"retrieval_production": [], "retrieval_calibration": cal_records}),
            _settings(with_calibration=True, calibration_min_submitted=3),
        )
        panel = report.panels["u-cal"]
        assert panel.panel_complete is True
        assert panel.distribution_satisfied is True

    def test_integrity_warning_when_records_mismatch_metadata_k(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [
            _record(chunk_id="c1", n_retrieved_chunks=5, response_statuses=["submitted"]),
            _record(chunk_id="c2", n_retrieved_chunks=5, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.panel_status"):
            report = compute_panel_status(_client({"retrieval_production": records}), _settings())
        panel = report.panels["u1"]
        assert panel.integrity_ok is False
        assert report.n_integrity_warnings == 1
        assert any("integrity warning" in r.message for r in caplog.records)

    def test_integrity_ok_when_metadata_absent(self) -> None:
        """No metadata K → integrity_ok=True (skips the check). Pre-backfill state."""
        records = [_record(chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"])]
        report = compute_panel_status(_client({"retrieval_production": records}), _settings())
        assert report.panels["u1"].integrity_ok is True

    def test_orphan_record_excluded(self) -> None:
        records = [
            _record(record_uuid="", chunk_id="orphan"),
            _record(record_uuid="u1", chunk_id="c1", response_statuses=["submitted"]),
        ]
        report = compute_panel_status(_client({"retrieval_production": records}), _settings())
        assert "" not in report.panels
        assert report.n_orphans_skipped == 1

    def test_walks_prod_then_cal_when_enabled(self) -> None:
        prod = [_record(record_uuid="u-prod", chunk_id="c1", response_statuses=["submitted"])]
        cal = [_record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted"])]
        report = compute_panel_status(
            _client({"retrieval_production": prod, "retrieval_calibration": cal}),
            _settings(with_calibration=True),
        )
        assert set(report.panels) == {"u-prod", "u-cal"}


# ---------------------------------------------------------------------------
# tag_incomplete_chunks
# ---------------------------------------------------------------------------


class TestTagIncompleteChunks:
    def test_tags_only_unresolved_chunks_in_incomplete_panels(self) -> None:
        """Incomplete panel with one resolved + two unresolved chunks → tag only the unresolved ones."""
        records = [
            _record(record_id="r1", chunk_id="c1", response_statuses=["submitted"]),  # resolved
            _record(record_id="r2", chunk_id="c2", response_statuses=[]),  # unresolved
            _record(record_id="r3", chunk_id="c3", response_statuses=[]),  # unresolved
            _record(record_id="r4", chunk_id="c4", n_retrieved_chunks=4, response_statuses=["submitted"]),
        ]
        # K=4 (from metadata or records-seen=4), n_terminal=2 → incomplete panel.
        client = _client({"retrieval_production": records})

        result = tag_incomplete_chunks(client, _settings())

        # r1 + r4: resolved → not tagged.
        # r2 + r3: panel incomplete + chunk unresolved → tagged.
        assert result.n_tagged == 2

    def test_does_not_tag_discarded_chunks(self) -> None:
        """Discarded counts as resolved/terminal; never tag it as needs_completion."""
        records = [
            _record(record_id="r1", chunk_id="c1", response_statuses=["discarded"]),  # resolved (discarded)
            _record(record_id="r2", chunk_id="c2", response_statuses=[]),  # unresolved
            _record(record_id="r3", chunk_id="c3", response_statuses=[]),  # unresolved
        ]
        client = _client({"retrieval_production": records})

        result = tag_incomplete_chunks(client, _settings())

        assert result.n_tagged == 2  # only r2 + r3

    def test_does_not_tag_complete_panel(self) -> None:
        records = [_record(record_id=f"r{i}", chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        client = _client({"retrieval_production": records})

        result = tag_incomplete_chunks(client, _settings())

        assert result.n_tagged == 0

    def test_clears_stale_tag_on_now_resolved_chunk(self) -> None:
        """Chunk previously tagged but now resolved → tag cleared."""
        records = [
            _record(
                record_id="r1",
                chunk_id="c1",
                response_statuses=["submitted"],
                needs_completion_tag=True,
            ),
            _record(record_id="r2", chunk_id="c2", response_statuses=[]),
        ]
        client = _client({"retrieval_production": records})

        result = tag_incomplete_chunks(client, _settings())

        assert result.n_cleared == 1  # r1 was tagged, now resolved
        assert result.n_tagged == 1  # r2 stays incomplete-unresolved

    def test_idempotent_already_tagged_no_rewrite(self) -> None:
        records = [
            _record(record_id="r1", chunk_id="c1", response_statuses=[], needs_completion_tag=True),
            _record(record_id="r2", chunk_id="c2", response_statuses=[]),
        ]
        client = _client({"retrieval_production": records})

        result = tag_incomplete_chunks(client, _settings())

        # r1 already tagged AND still needs the tag → counted as already_tagged (no write).
        assert result.n_already_tagged == 1
        # r2: needs tagging.
        assert result.n_tagged == 1

    def test_idempotent_against_list_shaped_tag_value(self) -> None:
        """Argilla may round-trip TermsMetadataProperty as ['true'] (list shape).

        The defensive equality treats list-shaped values as already-tagged so
        we don't re-write the same tag every run.
        """
        rec = MagicMock()
        rec.id = "r1"
        rec.metadata = {
            "record_uuid": "u1",
            "chunk_id": "c1",
            "needs_completion": ["true"],
        }
        r1 = MagicMock()
        r1.status = "submitted"
        rec.responses = []  # unresolved, so the tag should still be wanted
        client = _client({"retrieval_production": [rec, _record(record_id="r2", chunk_id="c2", response_statuses=[])]})

        result = tag_incomplete_chunks(client, _settings())

        assert result.n_already_tagged == 1  # rec was tagged (list-shape), still needs it
        assert result.n_tagged == 1  # r2 needs the tag

    def test_distribution_aggregates_submissions_across_duplicate_chunk_records(self) -> None:
        """If a single chunk_id has two records (rare anomaly), submissions sum across them.

        Without this, two records-for-one-chunk would each get checked
        independently against min_submitted, producing a spurious False.
        """
        # Two records under the same panel uuid AND same chunk_id, each with
        # 2 submitted responses. Cal threshold is 3. Total submissions on
        # chunk c1 = 4 >= 3, so distribution must be satisfied.
        rec_a = _record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted", "submitted"])
        rec_b = _record(record_uuid="u-cal", chunk_id="c1", response_statuses=["submitted", "submitted"])
        report = compute_panel_status(
            _client({"retrieval_production": [], "retrieval_calibration": [rec_a, rec_b]}),
            _settings(with_calibration=True, calibration_min_submitted=3),
        )
        panel = report.panels["u-cal"]
        assert panel.distribution_satisfied is True

    def test_warns_when_all_panels_have_unknown_k(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pre-backfill: all records missing n_retrieved_chunks → distinct warning,
        so operators don't read 0% complete as 'annotators haven't started'."""
        records = [
            _record(record_uuid="u1", chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"]),
            _record(record_uuid="u2", chunk_id="c1", n_retrieved_chunks=None, response_statuses=["submitted"]),
        ]
        with caplog.at_level(logging.WARNING, logger="pragmata.core.annotation.panel_status"):
            compute_panel_status(_client({"retrieval_production": records}), _settings())
        assert any("unknown K" in r.message for r in caplog.records)
