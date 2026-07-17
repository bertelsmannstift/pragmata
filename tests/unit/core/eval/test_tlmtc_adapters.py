"""Unit tests for tlmtc eval adapters."""

import sys
from collections.abc import MutableMapping
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd
import pytest

from pragmata.core.eval.tlmtc_adapters import run_tlmtc_predict, run_tlmtc_train

_DEDICATED_RUN_TLMTC_TRAIN_ARGS = [name for name in signature(run_tlmtc_train).parameters if name != "train_kwargs"]
_RESERVED_RUN_TLMTC_PREDICT_KWARGS = ["unlabeled_data", "work_dir", "run_id"]


def _run_tlmtc_train_kwargs(
    *,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "labeled_data": Path("train.csv"),
        "work_dir": Path("eval"),
        "target_name": "Retrieval evaluation",
        "checkpoint": "jhu-clsp/mmBERT-base",
        "proxy_checkpoint": "jhu-clsp/mmBERT-small",
        "scale_learning_rate": True,
        "sequence_length": 1024,
        "train_kwargs": train_kwargs or {},
    }


def test_run_tlmtc_train_merges_curated_args_and_train_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_tlmtc_train forwards curated settings plus tlmtc-owned passthrough kwargs."""
    captured_kwargs: dict[str, Any] = {}
    fake_tlmtc = ModuleType("tlmtc")
    expected_result = object()

    def train_tlmtc(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return expected_result

    setattr(fake_tlmtc, "train_tlmtc", train_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc", fake_tlmtc)

    result = run_tlmtc_train(
        **_run_tlmtc_train_kwargs(
            train_kwargs={
                "run_id": "custom-run",
                "batch_size": 8,
                "verbosity": "quiet",
                "trust_remote_code": False,
            }
        )
    )

    assert result is expected_result
    assert captured_kwargs == {
        "labeled_data": Path("train.csv"),
        "work_dir": Path("eval"),
        "target_name": "Retrieval evaluation",
        "checkpoint": "jhu-clsp/mmBERT-base",
        "proxy_checkpoint": "jhu-clsp/mmBERT-small",
        "scale_learning_rate": True,
        "sequence_length": 1024,
        "run_id": "custom-run",
        "batch_size": 8,
        "verbosity": "quiet",
        "trust_remote_code": False,
    }


def test_run_tlmtc_train_forwards_dedicated_args_when_train_kwargs_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_tlmtc_train forwards exactly the dedicated args without passthrough kwargs."""
    captured_kwargs: dict[str, Any] = {}
    fake_tlmtc = ModuleType("tlmtc")
    expected_result = object()

    def train_tlmtc(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return expected_result

    setattr(fake_tlmtc, "train_tlmtc", train_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc", fake_tlmtc)

    result = run_tlmtc_train(**_run_tlmtc_train_kwargs(train_kwargs={}))

    assert result is expected_result
    assert captured_kwargs == {
        "labeled_data": Path("train.csv"),
        "work_dir": Path("eval"),
        "target_name": "Retrieval evaluation",
        "checkpoint": "jhu-clsp/mmBERT-base",
        "proxy_checkpoint": "jhu-clsp/mmBERT-small",
        "scale_learning_rate": True,
        "sequence_length": 1024,
    }


@pytest.mark.parametrize("reserved_key", _DEDICATED_RUN_TLMTC_TRAIN_ARGS)
def test_run_tlmtc_train_rejects_reserved_train_kwargs(
    reserved_key: str,
) -> None:
    """train_kwargs must not override settings managed by dedicated arguments."""
    with pytest.raises(ValueError, match=reserved_key):
        run_tlmtc_train(**_run_tlmtc_train_kwargs(train_kwargs={reserved_key: "override"}))


def test_run_tlmtc_train_imports_tlmtc_lazily_and_reports_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing tlmtc raises an install hint only when training is invoked."""
    modules = cast(MutableMapping[str, Any], sys.modules)
    monkeypatch.setitem(modules, "tlmtc", None)

    with pytest.raises(ImportError, match="Install pragmata with the 'eval' extra"):
        run_tlmtc_train(**_run_tlmtc_train_kwargs())


def test_run_tlmtc_predict_forwards_path_and_predict_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_tlmtc_predict maps the evaluator ID and forwards tlmtc-owned options."""
    captured_kwargs: dict[str, Any] = {}
    fake_tlmtc = ModuleType("tlmtc")
    expected_result = object()

    def predict_tlmtc(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return expected_result

    setattr(fake_tlmtc, "predict_tlmtc", predict_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc", fake_tlmtc)

    result = run_tlmtc_predict(
        unlabeled_data=Path("predict.csv"),
        work_dir=Path("eval"),
        evaluator_run_id="evaluator-001",
        predict_kwargs={
            "batch_size": 8,
            "use_cpu": True,
            "verbosity": "quiet",
            "inference_backend": "torch",
        },
    )

    assert result is expected_result
    assert captured_kwargs == {
        "unlabeled_data": Path("predict.csv"),
        "work_dir": Path("eval"),
        "run_id": "evaluator-001",
        "batch_size": 8,
        "use_cpu": True,
        "verbosity": "quiet",
        "inference_backend": "torch",
    }


def test_run_tlmtc_predict_forwards_dataframe_with_empty_predict_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_tlmtc_predict accepts in-memory data without passthrough options."""
    captured_kwargs: dict[str, Any] = {}
    fake_tlmtc = ModuleType("tlmtc")
    unlabeled_data = pd.DataFrame({"text": ["example"]})

    def predict_tlmtc(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return object()

    setattr(fake_tlmtc, "predict_tlmtc", predict_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc", fake_tlmtc)

    run_tlmtc_predict(
        unlabeled_data=unlabeled_data,
        work_dir=Path("eval"),
        evaluator_run_id="evaluator-001",
        predict_kwargs={},
    )

    assert captured_kwargs["unlabeled_data"] is unlabeled_data
    assert captured_kwargs["work_dir"] == Path("eval")
    assert captured_kwargs["run_id"] == "evaluator-001"
    assert set(captured_kwargs) == {"unlabeled_data", "work_dir", "run_id"}


@pytest.mark.parametrize("reserved_key", _RESERVED_RUN_TLMTC_PREDICT_KWARGS)
def test_run_tlmtc_predict_rejects_reserved_predict_kwargs(
    reserved_key: str,
) -> None:
    """predict_kwargs must not override arguments managed by Pragmata."""
    with pytest.raises(ValueError, match=reserved_key):
        run_tlmtc_predict(
            unlabeled_data=Path("predict.csv"),
            work_dir=Path("eval"),
            evaluator_run_id="evaluator-001",
            predict_kwargs={reserved_key: "override"},
        )


def test_run_tlmtc_predict_reports_all_reserved_collisions() -> None:
    """Reserved predict collisions are reported together in deterministic order."""
    with pytest.raises(ValueError, match="run_id, unlabeled_data, work_dir"):
        run_tlmtc_predict(
            unlabeled_data=Path("predict.csv"),
            work_dir=Path("eval"),
            evaluator_run_id="evaluator-001",
            predict_kwargs={
                "work_dir": "override",
                "unlabeled_data": "override",
                "run_id": "override",
            },
        )


def test_run_tlmtc_predict_imports_tlmtc_lazily_and_reports_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing tlmtc raises an install hint only when prediction is invoked."""
    modules = cast(MutableMapping[str, Any], sys.modules)
    monkeypatch.setitem(modules, "tlmtc", None)

    with pytest.raises(ImportError, match="Install pragmata with the 'eval' extra"):
        run_tlmtc_predict(
            unlabeled_data=Path("predict.csv"),
            work_dir=Path("eval"),
            evaluator_run_id="evaluator-001",
            predict_kwargs={},
        )
