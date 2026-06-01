"""Unit tests for partial-bundle (incomplete) detection."""

from unittest.mock import MagicMock

from pragmata.core.annotation.incomplete import find_incomplete, workspace_domain


def _response(status: str) -> MagicMock:
    r = MagicMock()
    r.status = status
    return r


def _record(uuid: str, chunk_id: str, statuses: list[str], rec_id: str) -> MagicMock:
    rec = MagicMock()
    rec.id = rec_id
    rec.metadata = {"record_uuid": uuid, "chunk_id": chunk_id}
    rec.responses = [_response(s) for s in statuses]
    return rec


def _dataset(workspace: str, name: str, records: list[MagicMock]) -> MagicMock:
    ds = MagicMock()
    ds.name = name
    ds.workspace.name = workspace
    ds.records.return_value = records
    return ds


def _partial_pair(uuid: str) -> list[MagicMock]:
    # one record annotated, one not -> a partial bundle
    return [_record(uuid, "c1", ["submitted"], f"{uuid}1"), _record(uuid, "c2", [], f"{uuid}2")]


def test_lists_only_partial_bundles_and_their_unresolved_records() -> None:
    client = MagicMock()
    client.datasets = [
        _dataset(
            "Dom_retrieval",
            "retrieval_production",
            [
                # bundle A - partial: 1 of 3 done, 2 to finish
                _record("A", "c1", ["submitted"], "a1"),
                _record("A", "c2", [], "a2"),
                _record("A", "c3", [], "a3"),
                # bundle B - complete (all submitted): excluded
                _record("B", "c1", ["submitted"], "b1"),
                # bundle C - fully unstarted: excluded
                _record("C", "c1", [], "c1r"),
                _record("C", "c2", [], "c2r"),
            ],
        )
    ]
    report = find_incomplete(client)
    assert report.n_bundles == 1
    bundle = report.bundles[0]
    assert bundle.record_uuid == "A"
    assert bundle.n_records == 3
    assert bundle.n_submitted == 1
    assert sorted(bundle.missing_record_ids) == ["a2", "a3"]
    assert report.n_records == 2
    assert report.n_domains == 1
    assert report.tagged is False


def test_single_record_tasks_are_never_partial() -> None:
    client = MagicMock()
    client.datasets = [
        _dataset(
            "Dom_generation",
            "generation_production",
            [
                _record("X", "", ["submitted"], "x1"),  # complete
                _record("Y", "", [], "y1"),  # unstarted
            ],
        )
    ]
    assert find_incomplete(client).bundles == []


def test_filters_by_workspace_and_task() -> None:
    client = MagicMock()
    client.datasets = [
        _dataset("Dom1_retrieval", "retrieval_production", _partial_pair("A")),
        _dataset("Dom2_retrieval", "retrieval_calibration", _partial_pair("B")),
        _dataset("Dom1_generation", "generation_production", [_record("Z", "", [], "z1")]),
    ]
    assert {b.dataset for b in find_incomplete(client, task="retrieval").bundles} == {
        "retrieval_production",
        "retrieval_calibration",
    }
    assert find_incomplete(client, task="generation").bundles == []
    assert {b.workspace for b in find_incomplete(client, workspace="Dom1_retrieval").bundles} == {"Dom1_retrieval"}


def test_resolved_records_not_listed_to_finish() -> None:
    # partial (1 submitted, 1 discarded) but nothing UNRESOLVED left to do.
    client = MagicMock()
    client.datasets = [
        _dataset(
            "Dom_retrieval",
            "retrieval_production",
            [_record("A", "c1", ["submitted"], "r1"), _record("A", "c2", ["discarded"], "r2")],
        )
    ]
    assert find_incomplete(client).bundles == []


def test_workspace_domain_strips_task_suffix() -> None:
    assert workspace_domain("Demokratie-und-Zusammenhalt_retrieval") == "Demokratie-und-Zusammenhalt"
    assert workspace_domain("Foo_generation") == "Foo"
    assert workspace_domain("plain-workspace") == "plain-workspace"


def test_tag_path_runs_and_tags_unresolved_records() -> None:
    # Regression: the --tag path must not NameError on rg, and tags only the
    # unresolved record(s) of a partial bundle. TermsMetadataProperty needs a
    # live client to construct, so patch it out for this pure-logic test.
    from unittest.mock import patch

    client = MagicMock()
    client.datasets = [_dataset("Dom_retrieval", "retrieval_production", _partial_pair("A"))]
    with patch("argilla.TermsMetadataProperty"):
        report = find_incomplete(client, tag=True)
    assert report.tagged is True
    assert report.n_tagged == 1
    assert report.tasks == ["retrieval"]
