"""Tests for the annotation status API — Argilla client is fully mocked."""

from unittest.mock import MagicMock

import pytest

WS = "dom_retrieval"


@pytest.fixture()
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    client.datasets = []  # config-free: an iterable of dataset handles

    import pragmata.api.annotation_status as status_module

    monkeypatch.setattr(status_module, "resolve_argilla_client", lambda api_url, api_key: client)
    monkeypatch.delenv("ARGILLA_API_URL", raising=False)
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


def _dataset(
    name: str, records: list[MagicMock], *, workspace: str = WS, min_submitted: int = 1, progress: dict | None = None
) -> MagicMock:
    ds = MagicMock()
    ds.name = name
    ds.workspace.name = workspace
    ds.records.side_effect = lambda *a, **kw: iter(records)
    ds.progress.return_value = progress or {"total": 0, "completed": 0, "pending": 0}
    ds.settings.distribution.min_submitted = min_submitted
    return ds


class TestReportStatus:
    def test_empty_server_reports_zero(self, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        report = report_status(api_key="test-key")
        assert report.n_panels == 0
        assert report.n_complete == 0

    def test_panel_facts_returned(self, mock_client: MagicMock) -> None:
        from pragmata.api.annotation_status import report_status

        records = [_record(chunk_id=f"c{i}", response_statuses=["submitted"]) for i in range(5)]
        mock_client.datasets = [
            _dataset("retrieval_production", records, progress={"total": 5, "completed": 5, "pending": 0})
        ]

        report = report_status(api_key="test-key")
        assert report.n_panels == 1
        assert report.n_complete == 1
        assert report.panels[(WS, "u1")].panel_complete is True
        assert report.headline.total == 5
