"""Integration tests for annotation import against a live Argilla server.

Run with: pytest tests/integration/test_annotation_import.py -m "integration and annotation" -v
Requires: make setup (Argilla stack running on localhost:6900)
"""

import argilla as rg
import pytest

from pragmata.api.annotation_import import ImportResult, import_records
from pragmata.core.annotation.setup import setup_workspaces, teardown_resources
from pragmata.core.settings.annotation_settings import AnnotationSettings
from tests.integration._argilla_cleanup import purge_workspace_datasets

pytestmark = [pytest.mark.integration, pytest.mark.annotation]

_API_URL = "http://localhost:6900"
_API_KEY = "argilla.apikey"
_DATASET_ID = "testimport"
_CREDS: dict[str, str] = {"api_url": _API_URL, "api_key": _API_KEY}

_SETTINGS = AnnotationSettings(dataset_id=_DATASET_ID)


def _make_raw(n_chunks: int = 2, *, language: str | None = "de") -> dict:
    return {
        "query": f"What is chunk group {n_chunks}?",
        "answer": f"Answer for {n_chunks} chunks.",
        "chunks": [
            {"chunk_id": f"c{i}", "doc_id": "d1", "chunk_rank": i + 1, "text": f"Chunk text {i}."}
            for i in range(n_chunks)
        ],
        "context_set": "ctx-001",
        "language": language,
    }


@pytest.fixture(scope="module")
def client(annotation_stack_status) -> rg.Argilla:
    if not annotation_stack_status.ready:
        pytest.skip(annotation_stack_status.skip_reason or "annotation stack not ready")
    return rg.Argilla(api_url=_API_URL, api_key=_API_KEY)


@pytest.fixture(autouse=True, scope="module")
def clean_environment(client: rg.Argilla):
    """Tear down and re-setup environment before/after all tests.

    Orphan datasets from earlier runs are purged first — Argilla blocks
    workspace deletion while any dataset is linked.
    """
    teardown_resources(client, _SETTINGS)
    for ws_base in AnnotationSettings().workspace_dataset_map:
        purge_workspace_datasets(client, ws_base)
    teardown_resources(client, AnnotationSettings())
    setup_workspaces(client, _SETTINGS)
    yield
    teardown_resources(client, _SETTINGS)
    for ws_base in AnnotationSettings().workspace_dataset_map:
        purge_workspace_datasets(client, ws_base)
    teardown_resources(client, AnnotationSettings())


@pytest.fixture()
def sample_records() -> list[dict]:
    """Two records: first has 3 chunks, second has 2 chunks. Total retrieval: 5."""
    return [_make_raw(3), _make_raw(2)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_result_type(client: rg.Argilla, sample_records: list[dict]) -> None:
    result = import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)
    assert isinstance(result, ImportResult)


def test_record_counts_per_dataset(client: rg.Argilla, sample_records: list[dict]) -> None:
    """Retrieval: sum(chunks), grounding: N, generation: N."""
    n_records = len(sample_records)
    n_retrieval = sum(len(r["chunks"]) for r in sample_records)  # 3 + 2 = 5

    result = import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)

    assert result.total_records == n_records
    assert result.errors == []

    ret_ds_name = f"retrieval_{_DATASET_ID}"
    gnd_ds_name = f"grounding_{_DATASET_ID}"
    gen_ds_name = f"generation_{_DATASET_ID}"

    assert result.dataset_counts[ret_ds_name] == n_retrieval
    assert result.dataset_counts[gnd_ds_name] == n_records
    assert result.dataset_counts[gen_ds_name] == n_records


def test_records_exist_in_argilla(client: rg.Argilla, sample_records: list[dict]) -> None:
    """After import, all three datasets contain records."""
    import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)

    ret_ds = client.datasets(f"retrieval_{_DATASET_ID}", workspace="retrieval")
    gnd_ds = client.datasets(f"grounding_{_DATASET_ID}", workspace="grounding")
    gen_ds = client.datasets(f"generation_{_DATASET_ID}", workspace="generation")

    assert ret_ds is not None
    assert gnd_ds is not None
    assert gen_ds is not None

    # Argilla records are accessible
    assert len(list(ret_ds.records)) > 0
    assert len(list(gnd_ds.records)) > 0
    assert len(list(gen_ds.records)) > 0


def test_record_uuid_linkage(client: rg.Argilla, sample_records: list[dict]) -> None:
    """record_uuid metadata appears in all three datasets and intersects."""
    import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)

    def _uuids(ds_name: str, ws_name: str) -> set[str]:
        ds = client.datasets(ds_name, workspace=ws_name)
        return {r.metadata["record_uuid"] for r in ds.records if r.metadata.get("record_uuid")}

    ret_uuids = _uuids(f"retrieval_{_DATASET_ID}", "retrieval")
    gnd_uuids = _uuids(f"grounding_{_DATASET_ID}", "grounding")
    gen_uuids = _uuids(f"generation_{_DATASET_ID}", "generation")

    # All three datasets share the same UUIDs
    assert ret_uuids == gnd_uuids == gen_uuids
    assert len(ret_uuids) == len(sample_records)


def test_idempotent_reimport(client: rg.Argilla, sample_records: list[dict]) -> None:
    """Calling import_records twice with same data produces same record count.

    Idempotency relies on deterministic Record.id values derived from content hashes
    (derive_record_uuid). Argilla upserts on Record.id, so identical IDs on the second
    import overwrite existing records rather than creating duplicates.
    """
    import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)
    import_records(sample_records, dataset_id=_DATASET_ID, **_CREDS)

    ret_ds = client.datasets(f"retrieval_{_DATASET_ID}", workspace="retrieval")
    gnd_ds = client.datasets(f"grounding_{_DATASET_ID}", workspace="grounding")
    gen_ds = client.datasets(f"generation_{_DATASET_ID}", workspace="generation")

    n_retrieval = sum(len(r["chunks"]) for r in sample_records)
    n_records = len(sample_records)

    assert len(list(ret_ds.records)) == n_retrieval
    assert len(list(gnd_ds.records)) == n_records
    assert len(list(gen_ds.records)) == n_records


def test_invalid_records_skipped_with_errors(client: rg.Argilla) -> None:
    """Invalid dicts are reported as errors, not sent to Argilla."""
    raw = [{"query": "no answer or chunks"}, _make_raw(1)]

    result = import_records(raw, dataset_id=_DATASET_ID, **_CREDS)

    assert result.total_records == 2
    assert len(result.errors) == 1
    assert result.errors[0].index == 0
    # The one valid record was still imported
    assert sum(result.dataset_counts.values()) > 0


def test_dataset_auto_creation(client: rg.Argilla) -> None:
    """Import auto-creates datasets when they don't exist."""
    auto_id = "autotest"
    auto_settings = AnnotationSettings(dataset_id=auto_id)

    # Clean up any prior run
    teardown_resources(client, auto_settings)

    # Import without prior setup_datasets — datasets should be auto-created
    result = import_records([_make_raw(1)], dataset_id=auto_id, **_CREDS)

    assert result.total_records == 1
    assert result.errors == []

    # Datasets exist
    assert client.datasets(f"retrieval_{auto_id}", workspace="retrieval") is not None
    assert client.datasets(f"grounding_{auto_id}", workspace="grounding") is not None
    assert client.datasets(f"generation_{auto_id}", workspace="generation") is not None

    # Clean up
    teardown_resources(client, auto_settings)
