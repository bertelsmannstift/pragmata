"""Tests for eval input dataframe contracts."""

from collections.abc import Callable

import pandas as pd
import pytest

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import (
    EvalInputSchemaError,
    validate_eval_predict_frame,
    validate_eval_score_frame,
    validate_eval_train_frame,
)

FrameFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def retrieval_train_frame() -> FrameFactory:
    """Return a factory for valid retrieval training dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "chunk": ["first chunk", "second chunk"],
            "topically_relevant": [1, 0],
            "evidence_sufficient": [1, 0],
            "misleading": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def grounding_train_frame() -> FrameFactory:
    """Return a factory for valid grounding training dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "answer": ["first answer", "second answer"],
            "context_set": ["first context", "second context"],
            "support_present": [1, 0],
            "unsupported_claim_present": [0, 1],
            "contradicted_claim_present": [0, 1],
            "source_cited": [1, 0],
            "fabricated_source": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def generation_train_frame() -> FrameFactory:
    """Return a factory for valid generation training dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "answer": ["first answer", "second answer"],
            "proper_action": [1, 0],
            "response_on_topic": [1, 0],
            "helpful": [1, 0],
            "incomplete": [0, 1],
            "unsafe_content": [0, 1],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def retrieval_predict_frame() -> FrameFactory:
    """Return a factory for valid retrieval prediction dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "chunk": ["first chunk", "second chunk"],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def grounding_predict_frame() -> FrameFactory:
    """Return a factory for valid grounding prediction dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "answer": ["first answer", "second answer"],
            "context_set": ["first context", "second context"],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


@pytest.fixture
def generation_predict_frame() -> FrameFactory:
    """Return a factory for valid generation prediction dataframes."""

    def _factory(**overrides: object) -> pd.DataFrame:
        data: dict[str, object] = {
            "query": ["first query", "second query"],
            "answer": ["first answer", "second answer"],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    return _factory


class TestValidateEvalTrainFrame:
    """Tests for validating labeled eval training dataframes."""

    def test_validates_retrieval_train_frame(self, retrieval_train_frame: FrameFactory) -> None:
        df = retrieval_train_frame()

        validated = validate_eval_train_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(validated, df)

    def test_validates_grounding_train_frame(self, grounding_train_frame: FrameFactory) -> None:
        df = grounding_train_frame()

        validated = validate_eval_train_frame(df, task=Task.GROUNDING)

        pd.testing.assert_frame_equal(validated, df)

    def test_validates_generation_train_frame(self, generation_train_frame: FrameFactory) -> None:
        df = generation_train_frame()

        validated = validate_eval_train_frame(df, task=Task.GENERATION)

        pd.testing.assert_frame_equal(validated, df)

    def test_preserves_extra_columns(self, retrieval_train_frame: FrameFactory) -> None:
        df = retrieval_train_frame(record_uuid=["a", "b"], doc_id=["doc-a", "doc-b"])

        validated = validate_eval_train_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(validated, df)

    def test_coerces_label_values_to_integer(self, retrieval_train_frame: FrameFactory) -> None:
        df = retrieval_train_frame(
            topically_relevant=[1.0, 0.0],
            evidence_sufficient=[1.0, 0.0],
            misleading=[0.0, 1.0],
        )

        validated = validate_eval_train_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(
            validated[["topically_relevant", "evidence_sufficient", "misleading"]],
            pd.DataFrame(
                {
                    "topically_relevant": [1, 0],
                    "evidence_sufficient": [1, 0],
                    "misleading": [0, 1],
                },
                dtype="int64",
            ),
        )

    def test_rejects_non_dataframe_input(self) -> None:
        with pytest.raises(EvalInputSchemaError, match="Expected a pandas DataFrame"):
            validate_eval_train_frame("not a dataframe", task=Task.RETRIEVAL)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("task", "bad_df"),
        [
            (
                Task.RETRIEVAL,
                pd.DataFrame(
                    {
                        "query": ["first query", "second query"],
                        "topically_relevant": [1, 0],
                        "evidence_sufficient": [1, 0],
                        "misleading": [0, 1],
                    }
                ),
            ),
            (
                Task.GROUNDING,
                pd.DataFrame(
                    {
                        "answer": ["first answer", "second answer"],
                        "support_present": [1, 0],
                        "unsupported_claim_present": [0, 1],
                        "contradicted_claim_present": [0, 1],
                        "source_cited": [1, 0],
                        "fabricated_source": [0, 1],
                    }
                ),
            ),
            (
                Task.GENERATION,
                pd.DataFrame(
                    {
                        "query": ["first query", "second query"],
                        "proper_action": [1, 0],
                        "response_on_topic": [1, 0],
                        "helpful": [1, 0],
                        "incomplete": [0, 1],
                        "unsafe_content": [0, 1],
                    }
                ),
            ),
            (
                Task.RETRIEVAL,
                pd.DataFrame(
                    {
                        "query": [],
                        "chunk": [],
                        "topically_relevant": [],
                        "evidence_sufficient": [],
                        "misleading": [],
                    }
                ),
            ),
        ],
    )
    def test_rejects_structurally_invalid_frames(self, task: Task, bad_df: pd.DataFrame) -> None:
        with pytest.raises(EvalInputSchemaError, match="training data contract"):
            validate_eval_train_frame(bad_df, task=task)

    @pytest.mark.parametrize(
        ("task", "fixture_name", "overrides"),
        [
            (Task.RETRIEVAL, "retrieval_train_frame", {"query": ["first query", None]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"query": ["first query", "   "]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"chunk": ["first chunk", None]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"chunk": ["first chunk", "   "]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"topically_relevant": [1, None]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"topically_relevant": [1, 2]}),
            (Task.RETRIEVAL, "retrieval_train_frame", {"topically_relevant": [1, "yes"]}),
            (Task.GROUNDING, "grounding_train_frame", {"answer": ["first answer", None]}),
            (Task.GROUNDING, "grounding_train_frame", {"answer": ["first answer", "   "]}),
            (Task.GROUNDING, "grounding_train_frame", {"context_set": ["first context", None]}),
            (Task.GROUNDING, "grounding_train_frame", {"context_set": ["first context", "   "]}),
            (Task.GROUNDING, "grounding_train_frame", {"support_present": [1, None]}),
            (Task.GROUNDING, "grounding_train_frame", {"support_present": [1, 2]}),
            (Task.GROUNDING, "grounding_train_frame", {"support_present": [1, "yes"]}),
            (Task.GENERATION, "generation_train_frame", {"query": ["first query", None]}),
            (Task.GENERATION, "generation_train_frame", {"query": ["first query", "   "]}),
            (Task.GENERATION, "generation_train_frame", {"answer": ["first answer", None]}),
            (Task.GENERATION, "generation_train_frame", {"answer": ["first answer", "   "]}),
            (Task.GENERATION, "generation_train_frame", {"proper_action": [1, None]}),
            (Task.GENERATION, "generation_train_frame", {"proper_action": [1, 2]}),
            (Task.GENERATION, "generation_train_frame", {"proper_action": [1, "yes"]}),
        ],
    )
    def test_rejects_invalid_train_values(
        self,
        request: pytest.FixtureRequest,
        task: Task,
        fixture_name: str,
        overrides: dict[str, object],
    ) -> None:
        factory: FrameFactory = request.getfixturevalue(fixture_name)

        with pytest.raises(EvalInputSchemaError, match="training data contract"):
            validate_eval_train_frame(factory(**overrides), task=task)


class TestValidateEvalScoreFrame:
    """Tests for validating labeled eval scoring dataframes."""

    def test_reuses_train_contract(self, retrieval_train_frame: FrameFactory) -> None:
        df = retrieval_train_frame(record_uuid=["a", "b"])

        validated = validate_eval_score_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(validated, df)

    def test_rejects_invalid_labeled_score_frame(self, retrieval_train_frame: FrameFactory) -> None:
        df = retrieval_train_frame(topically_relevant=[1, 2])

        with pytest.raises(EvalInputSchemaError, match="training data contract"):
            validate_eval_score_frame(df, task=Task.RETRIEVAL)


class TestValidateEvalPredictFrame:
    """Tests for validating unlabeled eval prediction dataframes."""

    def test_validates_retrieval_predict_frame(self, retrieval_predict_frame: FrameFactory) -> None:
        df = retrieval_predict_frame()

        validated = validate_eval_predict_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(validated, df)

    def test_validates_grounding_predict_frame(self, grounding_predict_frame: FrameFactory) -> None:
        df = grounding_predict_frame()

        validated = validate_eval_predict_frame(df, task=Task.GROUNDING)

        pd.testing.assert_frame_equal(validated, df)

    def test_validates_generation_predict_frame(self, generation_predict_frame: FrameFactory) -> None:
        df = generation_predict_frame()

        validated = validate_eval_predict_frame(df, task=Task.GENERATION)

        pd.testing.assert_frame_equal(validated, df)

    def test_preserves_extra_columns(self, retrieval_predict_frame: FrameFactory) -> None:
        df = retrieval_predict_frame(record_uuid=["a", "b"], doc_id=["doc-a", "doc-b"])

        validated = validate_eval_predict_frame(df, task=Task.RETRIEVAL)

        pd.testing.assert_frame_equal(validated, df)

    def test_rejects_non_dataframe_prediction_input(self) -> None:
        with pytest.raises(EvalInputSchemaError, match="Expected a pandas DataFrame"):
            validate_eval_predict_frame("not a dataframe", task=Task.RETRIEVAL)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("task", "bad_df"),
        [
            (
                Task.RETRIEVAL,
                pd.DataFrame({"query": ["first query", "second query"]}),
            ),
            (
                Task.GROUNDING,
                pd.DataFrame({"answer": ["first answer", "second answer"]}),
            ),
            (
                Task.GENERATION,
                pd.DataFrame({"query": ["first query", "second query"]}),
            ),
            (
                Task.RETRIEVAL,
                pd.DataFrame({"query": [], "chunk": []}),
            ),
        ],
    )
    def test_rejects_structurally_invalid_prediction_frames(self, task: Task, bad_df: pd.DataFrame) -> None:
        with pytest.raises(EvalInputSchemaError, match="prediction data contract"):
            validate_eval_predict_frame(bad_df, task=task)

    @pytest.mark.parametrize(
        ("task", "fixture_name", "overrides"),
        [
            (Task.RETRIEVAL, "retrieval_predict_frame", {"query": ["first query", None]}),
            (Task.RETRIEVAL, "retrieval_predict_frame", {"query": ["first query", "   "]}),
            (Task.RETRIEVAL, "retrieval_predict_frame", {"chunk": ["first chunk", None]}),
            (Task.RETRIEVAL, "retrieval_predict_frame", {"chunk": ["first chunk", "   "]}),
            (Task.GROUNDING, "grounding_predict_frame", {"answer": ["first answer", None]}),
            (Task.GROUNDING, "grounding_predict_frame", {"answer": ["first answer", "   "]}),
            (Task.GROUNDING, "grounding_predict_frame", {"context_set": ["first context", None]}),
            (Task.GROUNDING, "grounding_predict_frame", {"context_set": ["first context", "   "]}),
            (Task.GENERATION, "generation_predict_frame", {"query": ["first query", None]}),
            (Task.GENERATION, "generation_predict_frame", {"query": ["first query", "   "]}),
            (Task.GENERATION, "generation_predict_frame", {"answer": ["first answer", None]}),
            (Task.GENERATION, "generation_predict_frame", {"answer": ["first answer", "   "]}),
        ],
    )
    def test_rejects_invalid_prediction_values(
        self,
        request: pytest.FixtureRequest,
        task: Task,
        fixture_name: str,
        overrides: dict[str, object],
    ) -> None:
        factory: FrameFactory = request.getfixturevalue(fixture_name)

        with pytest.raises(EvalInputSchemaError, match="prediction data contract"):
            validate_eval_predict_frame(factory(**overrides), task=task)

    def test_rejects_task_specific_label_columns(self, retrieval_predict_frame: FrameFactory) -> None:
        df = retrieval_predict_frame(topically_relevant=[1, 0])

        with pytest.raises(EvalInputSchemaError, match="Prediction input must be unlabeled"):
            validate_eval_predict_frame(df, task=Task.RETRIEVAL)

    def test_rejects_generic_label_columns(self, retrieval_predict_frame: FrameFactory) -> None:
        df = retrieval_predict_frame(label_relevant=[1, 0])

        with pytest.raises(EvalInputSchemaError, match="Prediction input must be unlabeled"):
            validate_eval_predict_frame(df, task=Task.RETRIEVAL)
