"""Unit tests for eval export."""

import json
from datetime import UTC, datetime
from pathlib import Path

from pragmata.core.eval.export import export_eval_meta
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import EvalPredictMeta, EvalTrainMeta


def test_export_eval_train_meta_serializes_metadata_json_values(
    tmp_path: Path,
) -> None:
    """export_eval_train_meta should serialize metadata using JSON-compatible model_dump output."""
    meta = EvalTrainMeta(
        run_id="train-run-1",
        created_at=datetime(2026, 5, 28, 13, 30, tzinfo=UTC),
        task=Task.RETRIEVAL,
        annotation_export_id="export-1",
    )
    meta_path = tmp_path / "pragmata_train.meta.json"

    export_eval_meta(meta=meta, path=meta_path)

    assert json.loads(meta_path.read_text(encoding="utf-8")) == {
        "run_id": "train-run-1",
        "created_at": "2026-05-28T13:30:00Z",
        "task": "retrieval",
        "annotation_export_id": "export-1",
    }


def test_export_eval_predict_meta_serializes_metadata_json_values(
    tmp_path: Path,
) -> None:
    """export_eval_predict_meta should serialize metadata using JSON-compatible model_dump output."""
    meta = EvalPredictMeta(
        run_id="prediction-evaluator",
        created_at=datetime(2026, 5, 28, 13, 30, tzinfo=UTC),
        task=Task.RETRIEVAL,
        unlabeled_data_path="/work/eval/inputs/retrieval-predict.csv",
    )
    meta_path = tmp_path / "pragmata_predict.meta.json"

    export_eval_meta(meta=meta, path=meta_path)

    assert json.loads(meta_path.read_text(encoding="utf-8")) == {
        "run_id": "prediction-evaluator",
        "created_at": "2026-05-28T13:30:00Z",
        "task": "retrieval",
        "unlabeled_data_path": "/work/eval/inputs/retrieval-predict.csv",
    }
