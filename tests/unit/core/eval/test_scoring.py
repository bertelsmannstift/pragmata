"""Tests for the pure eval scoring layer (``build_score_report``)."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from pragmata.core.eval.scoring import build_score_report
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import (
    GenerationScoreReport,
    GroundingScoreReport,
    MetricScore,
    RetrievalScoreReport,
    ScoreInputSource,
)

CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
SOURCE = ScoreInputSource(kind="direct_path", ref="in.csv", resolved_path="in.csv")


def _retrieval_frame() -> pd.DataFrame:
    # q1: two chunks (relevant+sufficient, relevant); q2: one irrelevant chunk.
    return pd.DataFrame(
        [
            {
                "record_uuid": "r1",
                "chunk_id": "c1",
                "chunk_rank": 1,
                "topically_relevant": 1,
                "evidence_sufficient": 1,
                "misleading": 0,
            },
            {
                "record_uuid": "r1",
                "chunk_id": "c2",
                "chunk_rank": 2,
                "topically_relevant": 1,
                "evidence_sufficient": 0,
                "misleading": 0,
            },
            {
                "record_uuid": "r2",
                "chunk_id": "c3",
                "chunk_rank": 1,
                "topically_relevant": 0,
                "evidence_sufficient": 0,
                "misleading": 1,
            },
        ]
    )


def _grounding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_uuid": "r1",
                "support_present": 1,
                "unsupported_claim_present": 0,
                "contradicted_claim_present": 0,
                "source_cited": 1,
                "fabricated_source": 1,
            },
            {
                "record_uuid": "r2",
                "support_present": 0,
                "unsupported_claim_present": 1,
                "contradicted_claim_present": 0,
                "source_cited": 0,
                "fabricated_source": 0,
            },
        ]
    )


def _generation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_uuid": "r1",
                "proper_action": 1,
                "response_on_topic": 1,
                "helpful": 1,
                "incomplete": 0,
                "unsafe_content": 0,
            },
            {
                "record_uuid": "r2",
                "proper_action": 0,
                "response_on_topic": 1,
                "helpful": 0,
                "incomplete": 1,
                "unsafe_content": 0,
            },
        ]
    )


def _build(
    frame: pd.DataFrame, task: Task, **kwargs
) -> RetrievalScoreReport | GroundingScoreReport | GenerationScoreReport:
    params = {"ci": 0.95, "n_resamples": 200, "seed": 42, "source": SOURCE, "created_at": CREATED_AT}
    params.update(kwargs)
    return build_score_report(frame, task=task, **params)


class TestRetrieval:
    def test_shape_and_methods(self):
        report = _build(_retrieval_frame(), Task.RETRIEVAL)

        assert isinstance(report, RetrievalScoreReport)
        assert report.task == Task.RETRIEVAL
        assert report.n_examples == 2  # two queries
        assert report.top_k == 2  # inferred max chunk_rank
        assert report.ci_level == 0.95
        assert report.source == SOURCE
        assert report.source.kind == "direct_path"
        assert report.created_at == CREATED_AT
        # topical precision: mean over queries of per-query mean = mean(1.0, 0.0) = 0.5
        assert report.topical_precision_at_k.point == pytest.approx(0.5)

        assert report.sufficiency_hit_at_k.method == "wilson"
        for field in (
            report.topical_precision_at_k,
            report.sufficiency_rate_at_k,
            report.misleading_context_rate_at_k,
            report.mean_reciprocal_rank_at_k,
            report.ndcg_at_k,
        ):
            assert field.method == "bootstrap"

    def test_wilson_point_within_interval_bootstrap_ordered(self):
        report = _build(_retrieval_frame(), Task.RETRIEVAL, seed=7)

        hit = report.sufficiency_hit_at_k
        assert hit.ci_lower <= hit.point <= hit.ci_upper
        for m in (report.topical_precision_at_k, report.ndcg_at_k):
            assert 0.0 <= m.ci_lower <= m.ci_upper <= 1.0

    def test_seed_reproducible(self):
        a = _build(_retrieval_frame(), Task.RETRIEVAL, seed=99, n_resamples=300)
        b = _build(_retrieval_frame(), Task.RETRIEVAL, seed=99, n_resamples=300)
        assert a.ndcg_at_k == b.ndcg_at_k
        assert a.topical_precision_at_k == b.topical_precision_at_k


class TestGrounding:
    def test_conditional_present(self):
        report = _build(_grounding_frame(), Task.GROUNDING)

        assert isinstance(report, GroundingScoreReport)
        assert report.grounding_presence_rate.method == "wilson"
        assert report.grounding_presence_rate.point == pytest.approx(0.5)
        # one cited query (r1) whose source is fabricated -> rate 1.0 over n=1
        assert report.conditional_fabrication_rate is not None
        assert report.conditional_fabrication_rate.point == pytest.approx(1.0)
        assert report.conditional_fabrication_rate.n == 1
        assert report.conditional_fabrication_rate.method == "wilson"

    def test_conditional_none_when_no_citations(self):
        frame = _grounding_frame()
        frame["source_cited"] = 0
        report = _build(frame, Task.GROUNDING)
        assert report.conditional_fabrication_rate is None


class TestGeneration:
    def test_all_wilson(self):
        report = _build(_generation_frame(), Task.GENERATION)

        assert isinstance(report, GenerationScoreReport)
        for m in (
            report.proper_action_rate,
            report.on_topic_rate,
            report.helpfulness_rate,
            report.incompleteness_rate,
            report.unsafe_content_rate,
        ):
            assert isinstance(m, MetricScore)
            assert m.method == "wilson"
        assert report.on_topic_rate.point == pytest.approx(1.0)  # both on-topic
