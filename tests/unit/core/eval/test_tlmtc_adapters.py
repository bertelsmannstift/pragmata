"""Unit tests for tlmtc eval adapters."""

import sys
from collections.abc import MutableMapping
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from pragmata.core.eval.tlmtc_adapters import train_evaluator


_DEDICATED_TRAIN_EVALUATOR_ARGS = [
    name
    for name in signature(train_evaluator).parameters
    if name != "train_kwargs"
]


def _train_evaluator_kwargs(
    *,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "raw_csv": Path("train.csv"),
        "work_dir": Path("eval"),
        "target_name": "Retrieval evaluation",
        "checkpoint": "EuroBERT/EuroBERT-610m",
        "proxy_checkpoint": "EuroBERT/EuroBERT-210m",
        "scale_learning_rate": True,
        "sequence_length": 1024,
        "trust_remote_code": True,
        "train_kwargs": train_kwargs or {},
    }


def test_train_evaluator_merges_curated_args_and_train_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_evaluator forwards curated settings plus tlmtc-owned passthrough kwargs."""
    captured_kwargs: dict[str, Any] = {}
    fake_tlmtc = ModuleType("tlmtc")
    expected_result = object()

    def train_tlmtc(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return expected_result

    setattr(fake_tlmtc, "train_tlmtc", train_tlmtc)
    monkeypatch.setitem(sys.modules, "tlmtc", fake_tlmtc)

    result = train_evaluator(
        **_train_evaluator_kwargs(
            train_kwargs={
                "run_id": "custom-run",
                "batch_size": 8,
                "verbosity": "quiet",
            }
        )
    )

    assert result is expected_result
    assert captured_kwargs == {
        "raw_csv": Path("train.csv"),
        "work_dir": Path("eval"),
        "target_name": "Retrieval evaluation",
        "checkpoint": "EuroBERT/EuroBERT-610m",
        "proxy_checkpoint": "EuroBERT/EuroBERT-210m",
        "scale_learning_rate": True,
        "sequence_length": 1024,
        "trust_remote_code": True,
        "run_id": "custom-run",
        "batch_size": 8,
        "verbosity": "quiet",
    }


@pytest.mark.parametrize("reserved_key", _DEDICATED_TRAIN_EVALUATOR_ARGS)
def test_train_evaluator_rejects_reserved_train_kwargs(
    reserved_key: str,
) -> None:
    """train_kwargs must not override settings managed by dedicated arguments."""
    with pytest.raises(ValueError, match=reserved_key):
        train_evaluator(
            **_train_evaluator_kwargs(
                train_kwargs={reserved_key: "override"}
            )
        )


def test_train_evaluator_imports_tlmtc_lazily_and_reports_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing tlmtc raises an install hint only when training is invoked."""
    modules = cast(MutableMapping[str, Any], sys.modules)
    monkeypatch.setitem(modules, "tlmtc", None)

    with pytest.raises(ImportError, match="Install pragmata with the 'eval' extra"):
        train_evaluator(**_train_evaluator_kwargs())
