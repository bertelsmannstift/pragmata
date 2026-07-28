"""Unit tests for grouping validated score frames into per-query values."""

import math

import numpy as np
import pandas as pd
import pytest

from pragmata.core.eval import grouping
from pragmata.core.eval.grouping import GENERATION_METRIC_LABELS, GROUNDING_METRIC_LABELS
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import LABEL_COLUMNS_BY_TASK


def _retrieval_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestRetrievalPerQueryValues:
    def test_two_queries(self):
        # rec-1: three chunks; rec-2: two chunks. Label patterns are chosen so
        # each retrieval metric yields a UNIQUE per-query array - a metric sourced
        # from the wrong label column (e.g. sufficiency_rate <-> misleading) would
        # produce a different array and fail.
        frame = _retrieval_frame(
            [
                {
                    "record_uuid": "rec-1",
                    "chunk_rank": 1,
                    "topically_relevant": 1,
                    "evidence_sufficient": 1,
                    "misleading": 0,
                },
                {
                    "record_uuid": "rec-1",
                    "chunk_rank": 2,
                    "topically_relevant": 1,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
                {
                    "record_uuid": "rec-1",
                    "chunk_rank": 3,
                    "topically_relevant": 0,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
                {
                    "record_uuid": "rec-2",
                    "chunk_rank": 1,
                    "topically_relevant": 0,
                    "evidence_sufficient": 0,
                    "misleading": 1,
                },
                {
                    "record_uuid": "rec-2",
                    "chunk_rank": 2,
                    "topically_relevant": 1,
                    "evidence_sufficient": 0,
                    "misleading": 1,
                },
            ]
        )

        values = grouping.retrieval_per_query_values(frame)

        assert values["topical_precision_at_k"] == pytest.approx([2 / 3, 0.5])
        assert values["sufficiency_hit_at_k"] == [1.0, 0.0]
        assert values["sufficiency_rate_at_k"] == pytest.approx([1 / 3, 0.0])
        assert values["misleading_context_rate_at_k"] == pytest.approx([0.0, 1.0])
        assert values["mean_reciprocal_rank_at_k"] == pytest.approx([1.0, 0.5])
        # rec-2's only relevant chunk sits at rank 2, so NDCG is penalised to
        # DCG/IDCG = (1/log2(3)) / (1/log2(2)) = 1/log2(3).
        assert values["ndcg_at_k"] == pytest.approx([1.0, 1 / math.log2(3)])

    def test_chunks_are_ordered_by_rank(self):
        # Rows supplied out of rank order; rank-1 chunk is the relevant one.
        # Correct ordering -> first relevant at rank 1 -> RR 1.0. Ignoring rank -> 0.5.
        frame = _retrieval_frame(
            [
                {
                    "record_uuid": "rec-1",
                    "chunk_rank": 2,
                    "topically_relevant": 0,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
                {
                    "record_uuid": "rec-1",
                    "chunk_rank": 1,
                    "topically_relevant": 1,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
            ]
        )

        values = grouping.retrieval_per_query_values(frame)

        assert values["mean_reciprocal_rank_at_k"] == pytest.approx([1.0])

    def test_query_order_is_first_appearance(self):
        frame = _retrieval_frame(
            [
                {
                    "record_uuid": "z",
                    "chunk_rank": 1,
                    "topically_relevant": 1,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
                {
                    "record_uuid": "a",
                    "chunk_rank": 1,
                    "topically_relevant": 0,
                    "evidence_sufficient": 0,
                    "misleading": 0,
                },
            ]
        )

        values = grouping.retrieval_per_query_values(frame)

        # 'z' appears first, so its topical precision (1.0) leads.
        assert values["topical_precision_at_k"] == pytest.approx([1.0, 0.0])


class TestGroundingPerQueryValues:
    def test_wiring_each_field_to_its_label(self):
        # Three rows so every mapped column has a distinct pattern: a swap
        # between any two fields would change the routed values and fail.
        frame = pd.DataFrame(
            {
                "record_uuid": ["r1", "r2", "r3"],
                "support_present": [1, 1, 1],
                "unsupported_claim_present": [1, 1, 0],
                "contradicted_claim_present": [1, 0, 0],
                "source_cited": [0, 1, 1],
                "fabricated_source": [0, 0, 0],
            }
        )

        values = grouping.grounding_per_query_values(frame)

        assert values["grounding_presence_rate"] == [1.0, 1.0, 1.0]
        assert values["unsupported_claim_rate"] == [1.0, 1.0, 0.0]
        assert values["contradiction_rate"] == [1.0, 0.0, 0.0]
        assert values["citation_presence_rate"] == [0.0, 1.0, 1.0]
        assert "conditional_fabrication_rate" not in values


class TestGenerationPerQueryValues:
    def test_wiring_each_field_to_its_label(self):
        # Three rows: five distinct column patterns so no two fields collide
        # (only four patterns exist with two rows), catching any rename/swap.
        frame = pd.DataFrame(
            {
                "record_uuid": ["r1", "r2", "r3"],
                "proper_action": [1, 1, 1],
                "response_on_topic": [1, 1, 0],
                "helpful": [1, 0, 0],
                "incomplete": [0, 1, 1],
                "unsafe_content": [0, 0, 1],
            }
        )

        values = grouping.generation_per_query_values(frame)

        assert values["proper_action_rate"] == [1.0, 1.0, 1.0]
        assert values["on_topic_rate"] == [1.0, 1.0, 0.0]
        assert values["helpfulness_rate"] == [1.0, 0.0, 0.0]
        assert values["incompleteness_rate"] == [0.0, 1.0, 1.0]
        assert values["unsafe_content_rate"] == [0.0, 0.0, 1.0]


class TestConditionalFabricationUnits:
    def test_cited_subset(self):
        frame = pd.DataFrame(
            {
                "record_uuid": ["r1", "r2", "r3"],
                "source_cited": [1, 0, 1],
                "fabricated_source": [1, 1, 0],
            }
        )

        units = grouping.conditional_fabrication_units(frame)

        assert np.array_equal(units, np.array([1, 0]))

    def test_no_cited_is_empty(self):
        frame = pd.DataFrame(
            {
                "record_uuid": ["r1"],
                "source_cited": [0],
                "fabricated_source": [1],
            }
        )

        assert grouping.conditional_fabrication_units(frame).shape[0] == 0


class TestMetricLabelMapsStayAligned:
    """Guard against silent drift if a label column is renamed in eval_input."""

    def test_grounding_map_targets_real_labels(self):
        # Four unconditional metrics; fabricated_source is used only via the
        # conditional fabrication rate, so it is the one grounding label absent here.
        assert set(GROUNDING_METRIC_LABELS.values()) | {"fabricated_source"} == set(
            LABEL_COLUMNS_BY_TASK[Task.GROUNDING]
        )

    def test_generation_map_targets_real_labels(self):
        assert set(GENERATION_METRIC_LABELS.values()) == set(LABEL_COLUMNS_BY_TASK[Task.GENERATION])
