"""Integration tests for annotation import against a live Argilla server.

Run with: pytest tests/annotation/test_import_integration.py -m integration -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

import argilla as rg
import pytest

from chatboteval.api.annotation_import import ImportResult, import_records
from chatboteval.api.annotation_setup import setup_datasets, teardown
from chatboteval.core.schemas.annotation_import import Chunk, QueryResponsePair
from chatboteval.core.settings.annotation_settings import AnnotationSettings

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_PREFIX = "testimport"

_SETTINGS = AnnotationSettings(workspace_prefix=_PREFIX)


def _make_pair(n_chunks: int = 2, *, language: str | None = "de") -> QueryResponsePair:
    return QueryResponsePair(
        query=f"What is chunk group {n_chunks}?",
        answer=f"Answer for {n_chunks} chunks.",
        chunks=[
            Chunk(chunk_id=f"c{i}", doc_id="d1", chunk_rank=i + 1, text=f"Chunk text {i}.") for i in range(n_chunks)
        ],
        context_set="ctx-001",
        language=language,
    )


@pytest.fixture(scope="module")
def client() -> rg.Argilla:
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


@pytest.fixture(autouse=True, scope="module")
def clean_environment(client: rg.Argilla):
    """Tear down and re-setup prefixed environment before/after all tests."""
    teardown(client, settings=_SETTINGS, include_users=False)
    setup_datasets(client, settings=_SETTINGS)
    yield
    teardown(client, settings=_SETTINGS, include_users=False)


@pytest.fixture()
def sample_pairs() -> list[QueryResponsePair]:
    """Two pairs: first has 3 chunks, second has 2 chunks. Total retrieval: 5."""
    return [_make_pair(3), _make_pair(2)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_import_result_type(client: rg.Argilla, sample_pairs: list[QueryResponsePair]) -> None:
    result = import_records(client, sample_pairs, settings=_SETTINGS)
    assert isinstance(result, ImportResult)


@pytest.mark.integration
def test_record_counts_per_dataset(client: rg.Argilla, sample_pairs: list[QueryResponsePair]) -> None:
    """Retrieval: sum(chunks), grounding: N, generation: N."""
    n_pairs = len(sample_pairs)
    n_retrieval = sum(len(p.chunks) for p in sample_pairs)  # 3 + 2 = 5

    result = import_records(client, sample_pairs, settings=_SETTINGS)

    assert result.total_records == n_pairs
    assert result.imported_records == n_pairs

    ret_ds_name = f"{_PREFIX}_task1_retrieval"
    gnd_ds_name = f"{_PREFIX}_task2_grounding"
    gen_ds_name = f"{_PREFIX}_task3_generation"

    assert result.dataset_counts[ret_ds_name] == n_retrieval
    assert result.dataset_counts[gnd_ds_name] == n_pairs
    assert result.dataset_counts[gen_ds_name] == n_pairs


@pytest.mark.integration
def test_records_exist_in_argilla(client: rg.Argilla, sample_pairs: list[QueryResponsePair]) -> None:
    """After import, all three datasets contain records."""
    import_records(client, sample_pairs, settings=_SETTINGS)

    ret_ds = client.datasets(f"{_PREFIX}_task1_retrieval", workspace=f"{_PREFIX}_retrieval")
    gnd_ds = client.datasets(f"{_PREFIX}_task2_grounding", workspace=f"{_PREFIX}_grounding")
    gen_ds = client.datasets(f"{_PREFIX}_task3_generation", workspace=f"{_PREFIX}_generation")

    assert ret_ds is not None
    assert gnd_ds is not None
    assert gen_ds is not None

    # Argilla records are accessible
    assert len(list(ret_ds.records)) > 0
    assert len(list(gnd_ds.records)) > 0
    assert len(list(gen_ds.records)) > 0


@pytest.mark.integration
def test_record_uuid_linkage(client: rg.Argilla, sample_pairs: list[QueryResponsePair]) -> None:
    """record_uuid metadata appears in all three datasets and intersects."""
    import_records(client, sample_pairs, settings=_SETTINGS)

    def _uuids(ds_name: str, ws_name: str) -> set[str]:
        ds = client.datasets(ds_name, workspace=ws_name)
        return {r.metadata["record_uuid"] for r in ds.records if r.metadata.get("record_uuid")}

    ret_uuids = _uuids(f"{_PREFIX}_task1_retrieval", f"{_PREFIX}_retrieval")
    gnd_uuids = _uuids(f"{_PREFIX}_task2_grounding", f"{_PREFIX}_grounding")
    gen_uuids = _uuids(f"{_PREFIX}_task3_generation", f"{_PREFIX}_generation")

    # All three datasets share the same UUIDs
    assert ret_uuids == gnd_uuids == gen_uuids
    assert len(ret_uuids) == len(sample_pairs)


@pytest.mark.integration
def test_idempotent_reimport(client: rg.Argilla, sample_pairs: list[QueryResponsePair]) -> None:
    """Calling import_records twice with same data produces same record count."""
    import_records(client, sample_pairs, settings=_SETTINGS)
    import_records(client, sample_pairs, settings=_SETTINGS)

    ret_ds = client.datasets(f"{_PREFIX}_task1_retrieval", workspace=f"{_PREFIX}_retrieval")
    gnd_ds = client.datasets(f"{_PREFIX}_task2_grounding", workspace=f"{_PREFIX}_grounding")
    gen_ds = client.datasets(f"{_PREFIX}_task3_generation", workspace=f"{_PREFIX}_generation")

    n_retrieval = sum(len(p.chunks) for p in sample_pairs)
    n_pairs = len(sample_pairs)

    assert len(list(ret_ds.records)) == n_retrieval
    assert len(list(gnd_ds.records)) == n_pairs
    assert len(list(gen_ds.records)) == n_pairs


@pytest.mark.integration
def test_validation_errors_do_not_reach_argilla(client: rg.Argilla) -> None:
    """Invalid raw dicts never trigger Argilla writes."""
    from chatboteval.api.annotation_import import validate_records

    invalid_raws = [{"query": "no answer or chunks"}]
    result = validate_records(invalid_raws)
    assert len(result.valid) == 0
    assert len(result.errors) == 1

    # Import the empty valid list — no writes should occur
    import_result = import_records(client, result.valid, settings=_SETTINGS)
    assert import_result.total_records == 0
    assert import_result.dataset_counts == {}
