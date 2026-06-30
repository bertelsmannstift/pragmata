"""Tests for eval dataframe transforms."""

import pandas as pd
import pytest

from pragmata.core.eval.transforms import build_tlmtc_frame
from pragmata.core.schemas.annotation_task import Task


@pytest.mark.parametrize(
    ("task", "frame", "expected"),
    [
        pytest.param(
            Task.RETRIEVAL,
            pd.DataFrame(
                {
                    "query": ["first query", "second query"],
                    "chunk": ["first chunk", "second chunk"],
                    "topically_relevant": [1, 0],
                    "evidence_sufficient": [1, 0],
                    "misleading": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first query", "second query"],
                    "text_pair": ["first chunk", "second chunk"],
                    "label_topically_relevant": [1, 0],
                    "label_evidence_sufficient": [1, 0],
                    "label_misleading": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
                }
            ),
            id="retrieval",
        ),
        pytest.param(
            Task.GROUNDING,
            pd.DataFrame(
                {
                    "answer": ["first answer", "second answer"],
                    "context_set": ["first context", "second context"],
                    "support_present": [1, 0],
                    "unsupported_claim_present": [0, 1],
                    "contradicted_claim_present": [0, 1],
                    "source_cited": [1, 0],
                    "fabricated_source": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first answer", "second answer"],
                    "text_pair": ["first context", "second context"],
                    "label_support_present": [1, 0],
                    "label_unsupported_claim_present": [0, 1],
                    "label_contradicted_claim_present": [0, 1],
                    "label_source_cited": [1, 0],
                    "label_fabricated_source": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
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
                    "proper_action": [1, 0],
                    "response_on_topic": [1, 0],
                    "helpful": [1, 0],
                    "incomplete": [0, 1],
                    "unsafe_content": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first query", "second query"],
                    "text_pair": ["first answer", "second answer"],
                    "label_proper_action": [1, 0],
                    "label_response_on_topic": [1, 0],
                    "label_helpful": [1, 0],
                    "label_incomplete": [0, 1],
                    "label_unsafe_content": [0, 1],
                    "record_uuid": ["record-1", "record-2"],
                }
            ),
            id="generation",
        ),
    ],
)
def test_build_tlmtc_train_frame_renames_task_columns(
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
            pd.DataFrame(
                {
                    "query": ["first query"],
                    "chunk": ["first chunk"],
                    "record_uuid": ["record-1"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first query"],
                    "text_pair": ["first chunk"],
                    "record_uuid": ["record-1"],
                }
            ),
            id="retrieval",
        ),
        pytest.param(
            Task.GROUNDING,
            pd.DataFrame(
                {
                    "answer": ["first answer"],
                    "context_set": ["first context"],
                    "record_uuid": ["record-1"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first answer"],
                    "text_pair": ["first context"],
                    "record_uuid": ["record-1"],
                }
            ),
            id="grounding",
        ),
        pytest.param(
            Task.GENERATION,
            pd.DataFrame(
                {
                    "query": ["first query"],
                    "answer": ["first answer"],
                    "record_uuid": ["record-1"],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["first query"],
                    "text_pair": ["first answer"],
                    "record_uuid": ["record-1"],
                }
            ),
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


@pytest.mark.parametrize("reserved_column", ["text", "text_pair"])
def test_build_tlmtc_frame_rejects_reserved_target_columns(reserved_column: str) -> None:
    frame = pd.DataFrame(
        {
            "query": ["first query"],
            "chunk": ["first chunk"],
            reserved_column: ["already present"],
        }
    )

    with pytest.raises(ValueError, match="reserved tlmtc columns"):
        build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="predict")


def test_build_tlmtc_frame_rejects_unknown_mode() -> None:
    frame = pd.DataFrame(
        {
            "query": ["first query"],
            "chunk": ["first chunk"],
        }
    )

    with pytest.raises(ValueError, match="Unsupported eval transform mode"):
        build_tlmtc_frame(frame, task=Task.RETRIEVAL, mode="score")  # type: ignore[arg-type]
