"""Tests for eval dataframe import workflows."""

from pathlib import Path

import pandas as pd
import pytest

from pragmata.core.eval.imports import import_eval_train_frame
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import EvalInputSchemaError


def test_import_eval_train_frame_reads_and_validates_csv(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.csv"
    path.write_text(
        "\n".join(
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,record_uuid",
                "first query,first chunk,1,1,0,record-1",
                "second query,second chunk,0,0,1,record-2",
            ]
        ),
        encoding="utf-8",
    )

    frame = import_eval_train_frame(path=path, task=Task.RETRIEVAL)

    expected = pd.DataFrame(
        {
            "query": ["first query", "second query"],
            "chunk": ["first chunk", "second chunk"],
            "topically_relevant": [1, 0],
            "evidence_sufficient": [1, 0],
            "misleading": [0, 1],
            "record_uuid": ["record-1", "record-2"],
        }
    )
    pd.testing.assert_frame_equal(frame, expected)


def test_import_eval_train_frame_accepts_export_style_lowercase_boolean_labels(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.csv"
    path.write_text(
        "\n".join(
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading",
                "first query,first chunk,true,true,false",
                "second query,second chunk,false,false,true",
            ]
        ),
        encoding="utf-8",
    )

    frame = import_eval_train_frame(path=path, task=Task.RETRIEVAL)

    expected_labels = pd.DataFrame(
        {
            "topically_relevant": [1, 0],
            "evidence_sufficient": [1, 0],
            "misleading": [0, 1],
        },
        dtype="int64",
    )
    pd.testing.assert_frame_equal(
        frame[["topically_relevant", "evidence_sufficient", "misleading"]],
        expected_labels,
    )


def test_import_eval_train_frame_rejects_invalid_csv(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.csv"
    path.write_text(
        "\n".join(
            [
                "query,topically_relevant,evidence_sufficient,misleading",
                "first query,1,1,0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalInputSchemaError, match="eval training data contract"):
        import_eval_train_frame(path=path, task=Task.RETRIEVAL)
