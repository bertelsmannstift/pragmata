"""Tests for eval dataframe transforms."""

import logging
from collections.abc import Callable

import pandas as pd
import pytest

from pragmata.core.eval.transforms import build_tlmtc_frame
from pragmata.core.schemas.annotation_task import Task

FrameFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def retrieval_train_frame() -> FrameFactory:
    """Return a factory for retrieval training dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "chunk": ["first chunk", "second chunk"],
            "chunk_id": ["chunk-1", "chunk-2"],
            "record_uuid": ["record-1", "record-2"],
            "topically_relevant": [1, 0],
            "evidence_sufficient": [1, 0],
            "misleading": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def generation_train_frame() -> FrameFactory:
    """Return a factory for generation training dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "answer": ["first answer", "second answer"],
            "record_uuid": ["record-1", "record-2"],
            "proper_action": [1, 0],
            "response_on_topic": [1, 0],
            "helpful": [1, 0],
            "incomplete": [0, 1],
            "unsafe_content": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


def test_build_tlmtc_train_frame_maps_retrieval_labels_and_split_group(
    retrieval_train_frame: FrameFactory,
) -> None:
    frame = retrieval_train_frame(doc_id=["doc-1", "doc-2"])

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["first query", "second query"],
            "text_pair": ["first chunk", "second chunk"],
            "chunk_id": ["chunk-1", "chunk-2"],
            "split_group": ["record-1", "record-2"],
            "label_topically_relevant": [1, 0],
            "label_evidence_sufficient": [1, 0],
            "label_misleading": [0, 1],
            "doc_id": ["doc-1", "doc-2"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


@pytest.mark.parametrize(
    ("task", "frame", "expected"),
    [
        pytest.param(
            Task.GROUNDING,
            pd.DataFrame(
                {
                    "answer": ["first answer", "second answer"],
                    "context_set": ["first context", "second context"],
                    "record_uuid": ["record-1", "record-2"],
                    "support_present": [1, 0],
                    "unsupported_claim_present": [0, 1],
                    "contradicted_claim_present": [0, 1],
                    "source_cited": [1, 0],
                    "fabricated_source": [0, 1],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first answer", "second answer"],
                    "text_pair": ["first context", "second context"],
                    "record_uuid": ["record-1", "record-2"],
                    "label_support_present": [1, 0],
                    "label_unsupported_claim_present": [0, 1],
                    "label_contradicted_claim_present": [0, 1],
                    "label_source_cited": [1, 0],
                    "label_fabricated_source": [0, 1],
                }
            ),
            id="grounding",
        ),
        pytest.param(
            Task.GENERATION,
            pd.DataFrame(
                {
                    "query": ["first query", "second query"],
                    "answer": ["first answer", "second answer"],
                    "record_uuid": ["record-1", "record-2"],
                    "proper_action": [1, 0],
                    "response_on_topic": [1, 0],
                    "helpful": [1, 0],
                    "incomplete": [0, 1],
                    "unsafe_content": [0, 1],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first query", "second query"],
                    "text_pair": ["first answer", "second answer"],
                    "record_uuid": ["record-1", "record-2"],
                    "label_proper_action": [1, 0],
                    "label_response_on_topic": [1, 0],
                    "label_helpful": [1, 0],
                    "label_incomplete": [0, 1],
                    "label_unsafe_content": [0, 1],
                }
            ),
            id="generation",
        ),
    ],
)
def test_build_tlmtc_train_frame_maps_non_retrieval_task_columns(
    task: Task,
    frame: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    transformed = build_tlmtc_frame(frame, task=task, mode="train")

    pd.testing.assert_frame_equal(transformed, expected)


@pytest.mark.parametrize(
    ("task", "frame", "expected"),
    [
        pytest.param(
            Task.RETRIEVAL,
            pd.DataFrame({"query": ["first query"], "chunk": ["first chunk"], "record_uuid": ["record-1"]}),
            pd.DataFrame({"text": ["first query"], "text_pair": ["first chunk"], "record_uuid": ["record-1"]}),
            id="retrieval",
        ),
        pytest.param(
            Task.GROUNDING,
            pd.DataFrame({"answer": ["first answer"], "context_set": ["first context"], "record_uuid": ["record-1"]}),
            pd.DataFrame({"text": ["first answer"], "text_pair": ["first context"], "record_uuid": ["record-1"]}),
            id="grounding",
        ),
        pytest.param(
            Task.GENERATION,
            pd.DataFrame({"query": ["first query"], "answer": ["first answer"], "record_uuid": ["record-1"]}),
            pd.DataFrame({"text": ["first query"], "text_pair": ["first answer"], "record_uuid": ["record-1"]}),
            id="generation",
        ),
    ],
)
def test_build_tlmtc_predict_frame_renames_task_columns(
    task: Task,
    frame: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    transformed = build_tlmtc_frame(frame, task=task, mode="predict")

    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_resets_index_and_preserves_extra_column_order() -> None:
    frame = pd.DataFrame(
        {
            "record_uuid": ["record-1", "record-2"],
            "chunk": ["first chunk", "second chunk"],
            "query": ["first query", "second query"],
            "doc_id": ["doc-1", "doc-2"],
        },
        index=[10, 20],
    )

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="predict")

    expected = pd.DataFrame(
        {
            "record_uuid": ["record-1", "record-2"],
            "text_pair": ["first chunk", "second chunk"],
            "text": ["first query", "second query"],
            "doc_id": ["doc-1", "doc-2"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_preserves_retrieval_train_without_record_uuid() -> None:
    frame = pd.DataFrame(
        {
            "query": ["first query"],
            "chunk": ["first chunk"],
            "topically_relevant": [1],
            "evidence_sufficient": [1],
            "misleading": [0],
        }
    )

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["first query"],
            "text_pair": ["first chunk"],
            "label_topically_relevant": [1],
            "label_evidence_sufficient": [1],
            "label_misleading": [0],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_consolidates_training_duplicates_by_earliest_consensus_match(
    generation_train_frame: FrameFactory,
) -> None:
    frame = generation_train_frame(
        query=["query", "query", "query"],
        answer=["answer", "answer", "answer"],
        record_uuid=["record-1", "record-1", "record-1"],
        proper_action=[0, 1, 1],
        response_on_topic=[1, 1, 1],
        helpful=[0, 1, 1],
        incomplete=[1, 0, 0],
        unsafe_content=[0, 0, 0],
        annotator_id=["annotator-a", "annotator-b", "annotator-c"],
    )

    transformed = build_tlmtc_frame(frame, task=Task.GENERATION, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query"],
            "text_pair": ["answer"],
            "record_uuid": ["record-1"],
            "label_proper_action": [1],
            "label_response_on_topic": [1],
            "label_helpful": [1],
            "label_incomplete": [0],
            "label_unsafe_content": [0],
            "annotator_id": ["annotator-b"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_consolidates_training_duplicates_by_earliest_fallback_on_tie(
    generation_train_frame: FrameFactory,
) -> None:
    frame = generation_train_frame(
        query=["query", "query"],
        answer=["answer", "answer"],
        record_uuid=["record-1", "record-1"],
        proper_action=[1, 0],
        response_on_topic=[1, 1],
        helpful=[1, 0],
        incomplete=[0, 1],
        unsafe_content=[0, 0],
        annotator_id=["annotator-a", "annotator-b"],
    )

    transformed = build_tlmtc_frame(frame, task=Task.GENERATION, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query"],
            "text_pair": ["answer"],
            "record_uuid": ["record-1"],
            "label_proper_action": [1],
            "label_response_on_topic": [1],
            "label_helpful": [1],
            "label_incomplete": [0],
            "label_unsafe_content": [0],
            "annotator_id": ["annotator-a"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_consolidates_training_duplicates_by_per_label_majority(
    generation_train_frame: FrameFactory,
) -> None:
    frame = generation_train_frame(
        query=["query", "query", "query", "query"],
        answer=["answer", "answer", "answer", "answer"],
        record_uuid=["record-1", "record-1", "record-1", "record-1"],
        proper_action=[0, 1, 1, 1],
        response_on_topic=[1, 1, 1, 1],
        helpful=[0, 1, 0, 1],
        incomplete=[0, 0, 1, 1],
        unsafe_content=[1, 0, 0, 0],
        annotator_id=["annotator-a", "annotator-b", "annotator-c", "annotator-d"],
    )

    transformed = build_tlmtc_frame(frame, task=Task.GENERATION, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query"],
            "text_pair": ["answer"],
            "record_uuid": ["record-1"],
            "label_proper_action": [1],
            "label_response_on_topic": [1],
            "label_helpful": [1],
            "label_incomplete": [0],
            "label_unsafe_content": [0],
            "annotator_id": ["annotator-b"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_logs_training_duplicate_consolidation(
    generation_train_frame: FrameFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = generation_train_frame(
        query=["query", "query", "query", "query"],
        answer=["answer", "answer", "answer", "answer"],
        record_uuid=["record-1", "record-1", "record-1", "record-1"],
        proper_action=[0, 1, 1, 1],
        response_on_topic=[1, 1, 1, 1],
        helpful=[0, 1, 0, 1],
        incomplete=[0, 0, 1, 1],
        unsafe_content=[1, 0, 0, 0],
    )

    with caplog.at_level(logging.INFO, logger="pragmata.core.eval.transforms"):
        build_tlmtc_frame(frame, task=Task.GENERATION, mode="train")

    assert caplog.messages == [
        "Consolidated duplicate eval training rows for generation: input_rows=4 output_rows=1 "
        "collapsed_rows=3 duplicate_units=1 key_columns=('record_uuid',)"
    ]


def test_build_tlmtc_frame_consolidates_retrieval_duplicates_by_record_and_chunk(
    retrieval_train_frame: FrameFactory,
) -> None:
    frame = retrieval_train_frame(
        query=["query", "query", "query"],
        chunk=["first chunk", "first chunk", "second chunk"],
        record_uuid=["record-1", "record-1", "record-1"],
        chunk_id=["chunk-1", "chunk-1", "chunk-2"],
        topically_relevant=[1, 1, 0],
        evidence_sufficient=[1, 1, 0],
        misleading=[0, 0, 1],
        annotator_id=["annotator-a", "annotator-b", "annotator-c"],
    )

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query", "query"],
            "text_pair": ["first chunk", "second chunk"],
            "chunk_id": ["chunk-1", "chunk-2"],
            "split_group": ["record-1", "record-1"],
            "label_topically_relevant": [1, 0],
            "label_evidence_sufficient": [1, 0],
            "label_misleading": [0, 1],
            "annotator_id": ["annotator-a", "annotator-c"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


@pytest.mark.parametrize("reserved_column", ["text", "text_pair"])
def test_build_tlmtc_frame_rejects_reserved_text_columns(reserved_column: str) -> None:
    frame = pd.DataFrame(
        {
            "query": ["first query"],
            "chunk": ["first chunk"],
            reserved_column: ["already present"],
        }
    )

    with pytest.raises(ValueError, match="reserved tlmtc columns"):
        build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="predict")


def test_build_tlmtc_frame_rejects_reserved_retrieval_split_group_column(
    retrieval_train_frame: FrameFactory,
) -> None:
    frame = retrieval_train_frame(split_group=["already present", "already present"])

    with pytest.raises(ValueError, match="reserved tlmtc columns"):
        build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")


def test_build_tlmtc_frame_rejects_unknown_mode() -> None:
    frame = pd.DataFrame(
        {
            "query": ["first query"],
            "chunk": ["first chunk"],
        }
    )

    with pytest.raises(ValueError, match="Unsupported eval transform mode"):
        build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="score")  # type: ignore[arg-type]


def test_build_tlmtc_frame_casts_bool_labels_before_majority_consolidation(
    retrieval_train_frame: FrameFactory,
) -> None:
    """Bool-typed labels are cast to int64 before majority-vote consolidation."""
    frame = retrieval_train_frame(
        query=["query", "query", "query"],
        chunk=["chunk", "chunk", "chunk"],
        chunk_id=["chunk-1", "chunk-1", "chunk-1"],
        record_uuid=["record-1", "record-1", "record-1"],
        doc_id=["doc-1", "doc-1", "doc-1"],
        topically_relevant=pd.array([True, True, False], dtype="boolean"),
        evidence_sufficient=pd.array([True, False, False], dtype="boolean"),
        misleading=pd.array([False, False, False], dtype="boolean"),
    )

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query"],
            "text_pair": ["chunk"],
            "chunk_id": ["chunk-1"],
            "split_group": ["record-1"],
            "label_topically_relevant": [1],
            "label_evidence_sufficient": [0],
            "label_misleading": [0],
            "doc_id": ["doc-1"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_build_tlmtc_frame_casts_bool_labels_without_consolidation(
    retrieval_train_frame: FrameFactory,
) -> None:
    """Bool-typed labels are cast to int64 even when no duplicates trigger consolidation."""
    frame = retrieval_train_frame(
        query=["query"],
        chunk=["chunk"],
        chunk_id=["chunk-1"],
        record_uuid=["record-1"],
        doc_id=["doc-1"],
        topically_relevant=pd.array([True], dtype="boolean"),
        evidence_sufficient=pd.array([False], dtype="boolean"),
        misleading=pd.array([True], dtype="boolean"),
    )

    transformed = build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="train")

    expected = pd.DataFrame(
        {
            "text": ["query"],
            "text_pair": ["chunk"],
            "chunk_id": ["chunk-1"],
            "split_group": ["record-1"],
            "label_topically_relevant": [1],
            "label_evidence_sufficient": [0],
            "label_misleading": [1],
            "doc_id": ["doc-1"],
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)