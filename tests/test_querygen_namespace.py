"""Tests for query generation public namespace."""

from pragmata import querygen


def test_querygen_namespace_importable() -> None:
    """Querygen namespace is importable."""
    assert querygen is not None


def test_querygen_namespace_exposes_expected_api() -> None:
    """Querygen namespace exposes the expected public API."""
    assert hasattr(querygen, "gen_queries")
    assert hasattr(querygen, "QueryGenRunResult")


def test_querygen_namespace_re_exports_api_objects() -> None:
    """Querygen namespace re-exports API-layer objects."""
    from pragmata.api.querygen import QueryGenRunResult, gen_queries

    assert querygen.gen_queries is gen_queries
    assert querygen.QueryGenRunResult is QueryGenRunResult
