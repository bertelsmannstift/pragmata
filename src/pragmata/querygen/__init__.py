"""Public synthetic query generation namespace."""

from pragmata.api.querygen import QueryGenRunResult as QueryGenRunResult
from pragmata.api.querygen import gen_queries as gen_queries

__all__ = ["gen_queries", "QueryGenRunResult"]
