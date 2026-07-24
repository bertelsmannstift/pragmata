"""Unit tests for evaluation workflow settings."""

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.eval_settings import EvalPredictSettings, EvalScoreSettings, EvalTrainSettings


def _assert_uuid_hex(value: str) -> None:
    """Assert that a value is a UUID4-style hex string."""
    assert len(value) == 32
    parsed = UUID(hex=value)
    assert parsed.version == 4
    assert parsed.hex == value


@pytest.mark.parametrize(
    ("model_cls", "payload", "expected_message"),
    [
        pytest.param(
            EvalTrainSettings,
            {},
            "Field required",
            id="train-requires-task",
        ),
        pytest.param(
            EvalPredictSettings,
            {"task": "retrieval"},
            "Field required",
            id="predict-requires-unlabeled-data-path",
        ),
        pytest.param(
            EvalPredictSettings,
            {"unlabeled_data_path": "data/unlabeled.csv"},
            "Field required",
            id="predict-requires-task",
        ),
        pytest.param(
            EvalScoreSettings,
            {},
            "Field required",
            id="score-requires-task",
        ),
        pytest.param(
            EvalTrainSettings,
            {"task": "retrieval", "run_id": "not-a-pragmata-train-setting"},
            "Extra inputs are not permitted",
            id="train-rejects-extra-fields",
        ),
        pytest.param(
            EvalPredictSettings,
            {
                "unlabeled_data_path": "data/unlabeled.csv",
                "task": "retrieval",
                "run_id": "not-a-pragmata-predict-setting",
            },
            "Extra inputs are not permitted",
            id="predict-rejects-extra-fields",
        ),
        pytest.param(
            EvalScoreSettings,
            {"task": "retrieval", "label_source": "predicted"},
            "Extra inputs are not permitted",
            id="score-rejects-extra-fields",
        ),
    ],
)
def test_eval_settings_validation_constraints(
    model_cls: type[BaseModel],
    payload: dict[str, object],
    expected_message: str,
) -> None:
    """Eval settings enforce required fields and reject undeclared top-level fields."""
    with pytest.raises(ValidationError, match=expected_message):
        model_cls.model_validate(payload)


def test_eval_train_settings_construction_with_defaults() -> None:
    """EvalTrainSettings applies train-level defaults when task is provided."""
    settings = EvalTrainSettings.model_validate({"task": "retrieval"})

    assert settings.base_dir == Path.cwd()
    assert settings.labeled_data_path is None
    assert settings.export_id is None
    assert settings.task == Task.RETRIEVAL
    assert settings.target_name == "Retrieval evaluation"
    assert settings.checkpoint == "jhu-clsp/mmBERT-base"
    assert settings.proxy_checkpoint == "jhu-clsp/mmBERT-small"
    assert settings.scale_learning_rate is True
    assert settings.sequence_length == 1024
    assert settings.train_kwargs == {}


def test_eval_train_settings_accepts_labeled_data_path() -> None:
    """EvalTrainSettings accepts a direct labeled data path."""
    settings = EvalTrainSettings.model_validate(
        {
            "labeled_data_path": "data/labeled.csv",
            "task": "grounding",
        }
    )

    assert settings.labeled_data_path == Path("data/labeled.csv")
    assert settings.task == Task.GROUNDING
    assert settings.target_name == "Grounding evaluation"


def test_eval_train_settings_accepts_export_id_and_train_kwargs() -> None:
    """EvalTrainSettings accepts annotation export selection and tlmtc train kwargs."""
    settings = EvalTrainSettings.model_validate(
        {
            "export_id": "export-001",
            "task": "generation",
            "target_name": "Custom target",
            "checkpoint": "custom/checkpoint",
            "proxy_checkpoint": "custom/proxy",
            "scale_learning_rate": False,
            "sequence_length": 384,
            "train_kwargs": {
                "run_id": "custom-train-run",
                "batch_size": 8,
            },
        }
    )

    assert settings.export_id == "export-001"
    assert settings.task == Task.GENERATION
    assert settings.target_name == "Custom target"
    assert settings.checkpoint == "custom/checkpoint"
    assert settings.proxy_checkpoint == "custom/proxy"
    assert settings.scale_learning_rate is False
    assert settings.sequence_length == 384
    assert settings.train_kwargs == {
        "run_id": "custom-train-run",
        "batch_size": 8,
    }


def test_eval_predict_settings_construction_with_defaults() -> None:
    """EvalPredictSettings applies predict-level defaults when required fields are provided."""
    settings = EvalPredictSettings.model_validate(
        {
            "unlabeled_data_path": "data/unlabeled.csv",
            "task": "retrieval",
        }
    )

    assert settings.base_dir == Path.cwd()
    assert settings.unlabeled_data_path == Path("data/unlabeled.csv")
    assert settings.evaluator_run_id is None
    assert settings.task == Task.RETRIEVAL
    assert settings.predict_kwargs == {}


def test_eval_predict_settings_accepts_evaluator_run_id_and_predict_kwargs() -> None:
    """EvalPredictSettings accepts evaluator selection and tlmtc predict kwargs."""
    settings = EvalPredictSettings.model_validate(
        {
            "unlabeled_data_path": "data/unlabeled.csv",
            "evaluator_run_id": "train-run-001",
            "task": "grounding",
            "predict_kwargs": {
                "batch_size": 32,
                "use_cpu": True,
            },
        }
    )

    assert settings.unlabeled_data_path == Path("data/unlabeled.csv")
    assert settings.evaluator_run_id == "train-run-001"
    assert settings.task == Task.GROUNDING
    assert settings.predict_kwargs == {
        "batch_size": 32,
        "use_cpu": True,
    }


def test_eval_score_settings_construction_with_path() -> None:
    """EvalScoreSettings accepts a direct labeled input path."""
    settings = EvalScoreSettings.model_validate(
        {
            "path": "data/labeled.csv",
            "task": "retrieval",
        }
    )

    assert settings.base_dir == Path.cwd()
    assert settings.score_id
    assert settings.path == Path("data/labeled.csv")
    assert settings.prediction_id is None
    assert settings.task == Task.RETRIEVAL


def test_eval_score_settings_construction_with_prediction_id() -> None:
    """EvalScoreSettings accepts a prediction run selector."""
    settings = EvalScoreSettings.model_validate(
        {
            "prediction_id": "prediction-run-001",
            "task": "grounding",
        }
    )

    assert settings.base_dir == Path.cwd()
    assert settings.score_id
    assert settings.path is None
    assert settings.prediction_id == "prediction-run-001"
    assert settings.task == Task.GROUNDING


def test_eval_score_settings_generates_distinct_score_ids() -> None:
    """EvalScoreSettings generates a default score_id for each instance."""
    first = EvalScoreSettings.model_validate({"task": "retrieval"})
    second = EvalScoreSettings.model_validate({"task": "retrieval"})

    assert first.score_id != second.score_id
    _assert_uuid_hex(first.score_id)
    _assert_uuid_hex(second.score_id)


def test_eval_score_settings_resolve_score_id_override_precedence() -> None:
    """Resolve applies explicit score_id overrides over config values."""
    resolved = EvalScoreSettings.resolve(
        config={
            "score_id": "score-config",
            "task": "retrieval",
        },
        overrides={
            "score_id": "score-override",
        },
    )

    assert resolved.score_id == "score-override"
