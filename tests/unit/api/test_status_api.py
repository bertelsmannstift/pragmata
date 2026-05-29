"""Tests for the annotation status API — Argilla client is fully mocked."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    empty_dataset = MagicMock()
    empty_dataset.records.side_effect = lambda *a, **kw: iter([])
    empty_dataset.progress.return_value = {"total": 0, "completed": 0, "pending": 0}
    empty_dataset.settings.metadata.__getitem__ = MagicMock(return_value=None)
    client.datasets.return_value = empty_dataset

    import pragmata.api.annotation_status as status_module

    monkeypatch.setattr(status_module, "resolve_argilla_client", lambda api_url, api_key: client)
    monkeypatch.delenv("ARGILLA_API_URL", raising=False)
    monkeypatch.setenv("ARGILLA_API_KEY", "test-key")
    return client


def _record(
    *,
    record_id: str = "rec-x",
    record_uuid: str = "u1",
    chunk_id: str = "c1",
    n_retrieved_chunks: int = 5,
    response_statuses: list[str] | None = None,
) -> MagicMock:
    metadata = {
        "record_uuid": record_uuid,
        "chunk_id": chunk_id,
        "n_retrieved_chunks": n_retrieved_chunks,
    }
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


def _set_production_dataset(client: MagicMock, dataset: MagicMock) -> None:
    """Wire client.datasets so production names resolve to ``dataset`` and calibration names to None."""

    def _datasets(name: str, workspace: str | None = None):
        if "_calibration" in name:
            return None
        return dataset

    client.datasets.side_effect = _datasets


class TestReportStatus:
    def test_empty_dataset_reports_zero(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        report, tag_result = report_status(base_dir=tmp_path)
        assert report.n_panels == 0
        assert report.n_complete == 0
        assert tag_result is None

    def test_panel_facts_returned(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        dataset = MagicMock()
        dataset.records.side_effect = lambda *a, **kw: iter(records)
        dataset.progress.return_value = {"total": 5, "completed": 5, "pending": 0}
        dataset.settings.metadata.__getitem__ = MagicMock(return_value=None)
        _set_production_dataset(mock_client, dataset)

        report, _ = report_status(base_dir=tmp_path)
        assert report.n_panels == 1
        assert report.n_complete == 1
        assert report.panels["u1"].panel_complete is True
        assert report.headline.total == 5

    def test_tag_incomplete_runs_when_requested(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        # Incomplete panel: K=2, only 1 chunk submitted.
        records = [
            _record(record_id="r1", chunk_id="c1", n_retrieved_chunks=2, response_statuses=["submitted"]),
            _record(record_id="r2", chunk_id="c2", n_retrieved_chunks=2, response_statuses=[]),
        ]
        dataset = MagicMock()
        dataset.name = "retrieval_production"
        dataset.records.side_effect = lambda *a, **kw: iter(records)
        dataset.progress.return_value = {"total": 2, "completed": 1, "pending": 1}
        dataset.settings.metadata.__getitem__ = MagicMock(return_value=None)
        _set_production_dataset(mock_client, dataset)

        report, tag_result = report_status(base_dir=tmp_path, tag_incomplete=True)
        assert report.panels["u1"].panel_complete is False
        assert tag_result is not None
        assert tag_result.n_tagged == 1  # only r2 (unresolved)

    def test_tag_incomplete_omitted_by_default(self, tmp_path: Path, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        _, tag_result = report_status(base_dir=tmp_path)
        assert tag_result is None
