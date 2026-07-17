"""Tests for the eval API orchestration."""

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import pragmata.api.eval as eval_api
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task


def _train_result(
    run_id: str = "train-run-1",
) -> SimpleNamespace:
    return SimpleNamespace(paths=SimpleNamespace(run_id=run_id))


def test_train_evaluator_orchestrates_direct_labeled_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_evaluator imports, transforms, and trains from a direct labeled CSV."""
    input_csv = tmp_path / "labeled.csv"
    input_csv.write_text("placeholder\nvalue\n", encoding="utf-8")
    eval_frame = pd.DataFrame({"source": ["eval"]})
    tlmtc_frame = pd.DataFrame({"text": ["query"], "text_pair": ["chunk"], "label_relevant": [1]})
    expected_result = _train_result()
    calls: dict[str, Any] = {}

    def import_eval_train_frame(
        *,
        path: Path,
        task: Task,
    ) -> pd.DataFrame:
        calls["import"] = {"path": path, "task": task}
        return eval_frame

    def build_tlmtc_frame(
        frame: pd.DataFrame,
        *,
        task: Task,
        mode: str,
    ) -> pd.DataFrame:
        calls["transform"] = {"frame": frame, "task": task, "mode": mode}
        return tlmtc_frame

    def run_tlmtc_train(
        **kwargs: Any,
    ) -> object:
        calls["train"] = kwargs
        return expected_result

    monkeypatch.setattr(eval_api, "import_eval_train_frame", import_eval_train_frame)
    monkeypatch.setattr(eval_api, "build_tlmtc_frame", build_tlmtc_frame)
    monkeypatch.setattr(eval_api, "run_tlmtc_train", run_tlmtc_train)
    monkeypatch.setattr(eval_api, "export_eval_train_meta", lambda **kwargs: calls.setdefault("export", kwargs))

    result = eval_api.train_evaluator(
        labeled_data_path=input_csv,
        task="retrieval",
        base_dir=tmp_path,
        train_kwargs={"run_id": "train-run-1", "verbosity": "quiet"},
    )

    assert result is expected_result
    assert calls["import"] == {"path": input_csv.resolve(), "task": Task.RETRIEVAL}
    assert calls["transform"]["frame"] is eval_frame
    assert calls["transform"]["task"] == Task.RETRIEVAL
    assert calls["transform"]["mode"] == "train"
    assert calls["train"]["labeled_data"] is tlmtc_frame
    assert {key: value for key, value in calls["train"].items() if key != "labeled_data"} == {
        "work_dir": tmp_path.resolve() / "eval",
        "target_name": "Retrieval evaluation",
        "checkpoint": "jhu-clsp/mmBERT-base",
        "proxy_checkpoint": "jhu-clsp/mmBERT-small",
        "scale_learning_rate": True,
        "sequence_length": 1024,
        "train_kwargs": {"run_id": "train-run-1", "verbosity": "quiet"},
    }
    assert calls["export"]["meta"].model_dump(exclude={"created_at"}) == {
        "run_id": "train-run-1",
        "task": Task.RETRIEVAL,
        "annotation_export_id": None,
    }
    assert (tmp_path / "eval").is_dir()


def test_train_evaluator_combines_config_and_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit train_evaluator args override config values before training."""
    input_csv = tmp_path / "labeled.csv"
    input_csv.write_text("placeholder\nvalue\n", encoding="utf-8")
    config_path = tmp_path / "eval.yml"
    config_path.write_text(
        dedent(
            """\
            task: grounding
            checkpoint: config/checkpoint
            proxy_checkpoint: config/proxy
            target_name: Config target
            scale_learning_rate: false
            sequence_length: 512
            train_kwargs:
              batch_size: 8
              trust_remote_code: false
            """
        ),
        encoding="utf-8",
    )
    tlmtc_frame = pd.DataFrame({"text": ["answer"], "text_pair": ["context"], "label_support_present": [1]})
    train_calls: list[dict[str, Any]] = []
    expected_result = _train_result("override-run")

    monkeypatch.setattr(eval_api, "import_eval_train_frame", lambda *, path, task: pd.DataFrame({"path": [path]}))
    monkeypatch.setattr(eval_api, "build_tlmtc_frame", lambda frame, *, task, mode: tlmtc_frame)
    monkeypatch.setattr(eval_api, "export_eval_train_meta", lambda **kwargs: None)

    def run_tlmtc_train(
        **kwargs: Any,
    ) -> SimpleNamespace:
        train_calls.append(kwargs)
        return expected_result

    monkeypatch.setattr(eval_api, "run_tlmtc_train", run_tlmtc_train)

    result = eval_api.train_evaluator(
        labeled_data_path=input_csv,
        base_dir=tmp_path,
        config_path=config_path,
        checkpoint="override/checkpoint",
        sequence_length=2048,
        train_kwargs={"batch_size": 16, "run_id": "override-run"},
    )

    assert result is expected_result
    assert len(train_calls) == 1
    assert train_calls[0]["labeled_data"] is tlmtc_frame
    assert {key: value for key, value in train_calls[0].items() if key != "labeled_data"} == {
        "work_dir": tmp_path.resolve() / "eval",
        "target_name": "Config target",
        "checkpoint": "override/checkpoint",
        "proxy_checkpoint": "config/proxy",
        "scale_learning_rate": False,
        "sequence_length": 2048,
        "train_kwargs": {"batch_size": 16, "trust_remote_code": False, "run_id": "override-run"},
    }


def test_train_evaluator_resolves_annotation_export_for_selected_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_evaluator asks path resolution for the selected export and task."""
    expected_result = _train_result("export-run")
    seen: dict[str, Any] = {}
    export_call: dict[str, Any] = {}
    tlmtc_frame = pd.DataFrame({"text": ["query"], "text_pair": ["answer"], "label_helpful": [1]})

    @dataclass(frozen=True, slots=True)
    class FakeTrainPaths:
        tool_root: Path
        training_input_csv: Path
        annotation_export_id: str | None

        def ensure_dirs(self) -> "FakeTrainPaths":
            seen["ensure_dirs_called"] = True
            return self

    def resolve_eval_train_paths(
        *,
        workspace: WorkspacePaths,
        task: Task,
        labeled_data_path: Path | None = None,
        export_id: str | None = None,
    ) -> FakeTrainPaths:
        seen["resolve_paths"] = {
            "workspace_base_dir": workspace.base_dir,
            "task": task,
            "labeled_data_path": labeled_data_path,
            "export_id": export_id,
        }
        return FakeTrainPaths(
            tool_root=workspace.tool_root("eval"),
            training_input_csv=tmp_path / "annotation" / "exports" / "export-1" / "generation.csv",
            annotation_export_id=export_id,
        )

    monkeypatch.setattr(eval_api, "resolve_eval_train_paths", resolve_eval_train_paths)
    monkeypatch.setattr(eval_api, "import_eval_train_frame", lambda *, path, task: pd.DataFrame({"path": [path]}))
    monkeypatch.setattr(eval_api, "build_tlmtc_frame", lambda frame, *, task, mode: tlmtc_frame)
    monkeypatch.setattr(eval_api, "run_tlmtc_train", lambda **kwargs: expected_result)
    monkeypatch.setattr(eval_api, "export_eval_train_meta", lambda **kwargs: export_call.update(kwargs))

    result = eval_api.train_evaluator(
        export_id="export-1",
        task=Task.GENERATION,
        base_dir=tmp_path,
    )

    assert result is expected_result
    assert seen == {
        "resolve_paths": {
            "workspace_base_dir": tmp_path.resolve(),
            "task": Task.GENERATION,
            "labeled_data_path": None,
            "export_id": "export-1",
        },
        "ensure_dirs_called": True,
    }
    assert export_call["meta"].model_dump(exclude={"created_at"}) == {
        "run_id": "export-run",
        "task": Task.GENERATION,
        "annotation_export_id": "export-1",
    }


def test_predict_labels_orchestrates_direct_unlabeled_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """predict_labels selects an evaluator, transforms input, and predicts."""
    input_csv = tmp_path / "unlabeled.csv"
    input_csv.write_text("placeholder\nvalue\n", encoding="utf-8")
    eval_frame = pd.DataFrame({"source": ["eval"]})
    tlmtc_frame = pd.DataFrame({"text": ["query"], "text_pair": ["chunk"]})
    expected_result = object()
    calls: dict[str, Any] = {}

    def resolve_eval_train_run_id(
        *,
        workspace: WorkspacePaths,
        task: Task,
        evaluator_run_id: str | None = None,
    ) -> str:
        calls["select"] = {
            "workspace_base_dir": workspace.base_dir,
            "task": task,
            "evaluator_run_id": evaluator_run_id,
        }
        return "evaluator-1"

    def import_eval_predict_frame(
        *,
        path: Path,
        task: Task,
    ) -> pd.DataFrame:
        calls["import"] = {"path": path, "task": task}
        return eval_frame

    def build_tlmtc_frame(
        frame: pd.DataFrame,
        *,
        task: Task,
        mode: str,
    ) -> pd.DataFrame:
        calls["transform"] = {"frame": frame, "task": task, "mode": mode}
        return tlmtc_frame

    def run_tlmtc_predict(
        **kwargs: Any,
    ) -> object:
        calls["predict"] = kwargs
        return expected_result

    monkeypatch.setattr(eval_api, "resolve_eval_train_run_id", resolve_eval_train_run_id)
    monkeypatch.setattr(eval_api, "import_eval_predict_frame", import_eval_predict_frame)
    monkeypatch.setattr(eval_api, "build_tlmtc_frame", build_tlmtc_frame)
    monkeypatch.setattr(eval_api, "run_tlmtc_predict", run_tlmtc_predict)

    result = eval_api.predict_labels(
        unlabeled_data_path=input_csv,
        evaluator_run_id="evaluator-1",
        task="retrieval",
        base_dir=tmp_path,
        predict_kwargs={"batch_size": 8, "use_cpu": True},
    )

    assert result is expected_result
    assert calls["select"] == {
        "workspace_base_dir": tmp_path.resolve(),
        "task": Task.RETRIEVAL,
        "evaluator_run_id": "evaluator-1",
    }
    assert calls["import"] == {"path": input_csv.resolve(), "task": Task.RETRIEVAL}
    assert calls["transform"] == {"frame": eval_frame, "task": Task.RETRIEVAL, "mode": "predict"}
    assert calls["predict"]["unlabeled_data"] is tlmtc_frame
    assert {key: value for key, value in calls["predict"].items() if key != "unlabeled_data"} == {
        "work_dir": tmp_path.resolve() / "eval",
        "evaluator_run_id": "evaluator-1",
        "predict_kwargs": {"batch_size": 8, "use_cpu": True},
    }
    assert (tmp_path / "eval").is_dir()


def test_predict_labels_combines_config_and_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit predict args override config before automatic evaluator selection."""
    config_input_csv = tmp_path / "config-unlabeled.csv"
    override_input_csv = tmp_path / "override-unlabeled.csv"
    override_input_csv.write_text("placeholder\nvalue\n", encoding="utf-8")
    config_path = tmp_path / "eval.yml"
    config_path.write_text(
        dedent(
            f"""\
            base_dir: {tmp_path / "config-workspace"}
            unlabeled_data_path: {config_input_csv}
            task: grounding
            predict_kwargs:
              batch_size: 8
              use_cpu: false
            """
        ),
        encoding="utf-8",
    )
    tlmtc_frame = pd.DataFrame({"text": ["query"], "text_pair": ["answer"]})
    expected_result = object()
    calls: dict[str, Any] = {}

    def resolve_eval_train_run_id(
        *,
        workspace: WorkspacePaths,
        task: Task,
        evaluator_run_id: str | None = None,
    ) -> str:
        calls["select"] = {
            "workspace_base_dir": workspace.base_dir,
            "task": task,
            "evaluator_run_id": evaluator_run_id,
        }
        return "latest-generation-evaluator"

    def run_tlmtc_predict(
        **kwargs: Any,
    ) -> object:
        calls["predict"] = kwargs
        return expected_result

    monkeypatch.setattr(eval_api, "resolve_eval_train_run_id", resolve_eval_train_run_id)
    monkeypatch.setattr(eval_api, "import_eval_predict_frame", lambda *, path, task: pd.DataFrame({"path": [path]}))
    monkeypatch.setattr(eval_api, "build_tlmtc_frame", lambda frame, *, task, mode: tlmtc_frame)
    monkeypatch.setattr(eval_api, "run_tlmtc_predict", run_tlmtc_predict)

    result = eval_api.predict_labels(
        unlabeled_data_path=override_input_csv,
        task="generation",
        base_dir=tmp_path,
        config_path=config_path,
        predict_kwargs={"batch_size": 16},
    )

    assert result is expected_result
    assert calls["select"] == {
        "workspace_base_dir": tmp_path.resolve(),
        "task": Task.GENERATION,
        "evaluator_run_id": None,
    }
    assert calls["predict"]["unlabeled_data"] is tlmtc_frame
    assert {key: value for key, value in calls["predict"].items() if key != "unlabeled_data"} == {
        "work_dir": tmp_path.resolve() / "eval",
        "evaluator_run_id": "latest-generation-evaluator",
        "predict_kwargs": {"batch_size": 16, "use_cpu": False},
    }
