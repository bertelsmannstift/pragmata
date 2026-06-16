"""Tests for eval dataframe import workflows."""

from pathlib import Path

import pandas as pd
import pytest

from pragmata.core.eval.imports import import_eval_predict_frame, import_eval_train_frame
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import EvalInputSchemaError


class TestImportEvalTrainFrame:
    """Tests for labeled eval training dataframe imports."""

    def test_import_eval_train_frame_reads_and_validates_csv(
        self,
        tmp_path: Path,
    ) -> None:
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

    def test_import_eval_train_frame_accepts_export_style_lowercase_boolean_labels(
        self,
        tmp_path: Path,
    ) -> None:
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

    def test_import_eval_train_frame_rejects_invalid_csv(
        self,
        tmp_path: Path,
    ) -> None:
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


class TestImportEvalPredictFrame:
    """Tests for unlabeled eval prediction dataframe imports."""

    def test_import_eval_predict_frame_reads_and_validates_csv(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "retrieval_predict.csv"
        path.write_text(
            "\n".join(
                [
                    "query,chunk,record_uuid",
                    "first query,first chunk,record-1",
                    "second query,second chunk,record-2",
                ]
            ),
            encoding="utf-8",
        )

        frame = import_eval_predict_frame(path=path, task=Task.RETRIEVAL)

        expected = pd.DataFrame(
            {
                "query": ["first query", "second query"],
                "chunk": ["first chunk", "second chunk"],
                "record_uuid": ["record-1", "record-2"],
            }
        )
        pd.testing.assert_frame_equal(frame, expected)

    def test_import_eval_predict_frame_rejects_labeled_csv(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "retrieval_predict.csv"
        path.write_text(
            "\n".join(
                [
                    "query,chunk,topically_relevant",
                    "first query,first chunk,1",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(EvalInputSchemaError, match="Prediction input must be unlabeled"):
            import_eval_predict_frame(path=path, task=Task.RETRIEVAL)

    def test_import_eval_predict_frame_rejects_invalid_csv(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "retrieval_predict.csv"
        path.write_text(
            "\n".join(
                [
                    "query,record_uuid",
                    "first query,record-1",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(EvalInputSchemaError, match="eval prediction data contract"):
            import_eval_predict_frame(path=path, task=Task.RETRIEVAL)
