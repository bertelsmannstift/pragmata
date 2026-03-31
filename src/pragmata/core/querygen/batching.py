"""Batching helpers for synthetic query generation."""

from collections.abc import Iterator

from pragmata.core.schemas.querygen_plan import QueryBlueprint


def build_candidate_ids(n_queries: int) -> list[str]:
    """Build deterministic run-local candidate IDs."""
    digits = max(3, len(str(n_queries)))
    return [f"c{index:0{digits}d}" for index in range(1, n_queries + 1)]


def iter_batch_sizes(n_queries: int, batch_size: int) -> Iterator[int]:
    """Yield repeated batch sizes for a requested total query count."""
    remaining = n_queries
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        yield current_batch_size
        remaining -= current_batch_size


def chunk_blueprints(
    blueprints: list[QueryBlueprint],
    chunk_size: int,
) -> Iterator[list[QueryBlueprint]]:
    """Chunk ordered stage-1 blueprints for downstream realization."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    for start in range(0, len(blueprints), chunk_size):
        yield blueprints[start : start + chunk_size]