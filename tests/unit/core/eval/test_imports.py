"""Tests for eval dataframe import workflows."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from pragmata.core.eval.imports import (
    import_eval_predict_frame,
    import_eval_score_frame,
    import_eval_train_frame,
)
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import EvalInputSchemaError
from pragmata.core.schemas.eval_output import ScoreInputSource


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


class TestImportEvalScoreFrame:
    """Tests for labeled eval scoring dataframe imports."""

    _DIRECT = ScoreInputSource(kind="direct_path", ref="in.csv", resolved_path="in.csv")
    _PREDICTION = ScoreInputSource(
        kind="model_prediction", ref="run-1", resolved_path="eval/predictions/run-1/predictions.csv"
    )

    def _write(self, tmp_path: Path, rows: list[str]) -> Path:
        path = tmp_path / "score.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_reads_and_validates_pragmata_shaped_csv(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,c1,1,1,0,r1,ch1,1",
                "q1,c2,0,0,1,r1,ch2,2",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

        assert list(frame["query"]) == ["q1", "q1"]
        assert set(["query", "chunk", "record_uuid", "chunk_id", "chunk_rank"]).issubset(frame.columns)

    def test_rejects_invalid_csv(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "query,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,1,1,0,r1,ch1,1",
            ],
        )

        with pytest.raises(EvalInputSchemaError):
            import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

    def test_consolidates_multi_annotator_chunk_by_majority(self, tmp_path: Path) -> None:
        # Three annotators labeled the same (record_uuid, chunk_id); collapse to one majority row.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,c1,1,1,0,r1,ch1,1",
                "q1,c1,1,1,1,r1,ch1,1",
                "q1,c1,0,1,1,r1,ch1,1",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["record_uuid"] == "r1" and row["chunk_id"] == "ch1"
        assert row["topically_relevant"] == 1  # 2/3 positive
        assert row["evidence_sufficient"] == 1  # 3/3
        assert row["misleading"] == 1  # 2/3

    def test_rejects_non_collapsible_duplicate_chunk_rank(self, tmp_path: Path) -> None:
        # Two distinct chunk_ids sharing a chunk_rank cannot be majority-collapsed
        # (different units), so the post-collapse guard still hard-errors.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,c1,1,1,0,r1,ch1,1",
                "q1,c2,0,0,1,r1,ch2,1",
            ],
        )

        with pytest.raises(EvalInputSchemaError, match="duplicate chunk rank key"):
            import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

    def test_consolidates_multi_annotator_query_for_grounding(self, tmp_path: Path) -> None:
        # Three annotators labeled the same grounding query; collapse to one majority row.
        path = self._write(
            tmp_path,
            [
                "answer,context_set,support_present,unsupported_claim_present,"
                "contradicted_claim_present,source_cited,fabricated_source,record_uuid",
                "a1,ctx1,1,0,0,1,0,r1",
                "a1,ctx1,1,0,0,1,0,r1",
                "a1,ctx1,1,1,0,0,0,r1",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.GROUNDING, source=self._DIRECT)

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["record_uuid"] == "r1"
        assert row["support_present"] == 1  # 3/3
        assert row["unsupported_claim_present"] == 0  # 1/3
        assert row["source_cited"] == 1  # 2/3

    def test_restores_tlmtc_text_columns_for_prediction_input(self, tmp_path: Path) -> None:
        # A prediction artifact is tlmtc-shaped: task text columns arrive as text/text_pair.
        path = self._write(
            tmp_path,
            [
                "text,text_pair,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,c1,1,1,0,r1,ch1,1",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._PREDICTION)

        assert {"query", "chunk"}.issubset(frame.columns)
        assert "text" not in frame.columns and "text_pair" not in frame.columns
        assert frame.loc[0, "query"] == "q1"
        assert frame.loc[0, "chunk"] == "c1"

    def test_rejects_retrieval_panel_short_of_n_retrieved_chunks(self, tmp_path: Path) -> None:
        # Two labeled chunks of a K=5 retrieval: the @K denominators would be wrong.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,5",
                "q1,c2,0,0,1,r1,ch2,2,5",
            ],
        )

        with pytest.raises(EvalInputSchemaError, match="fewer labeled chunks"):
            import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

    def test_accepts_retrieval_panel_covering_n_retrieved_chunks(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,2",
                "q1,c2,0,0,1,r1,ch2,2,2",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

        assert len(frame) == 2

    def test_allow_incomplete_panels_skips_the_completeness_guard(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,5",
            ],
        )

        frame, n_panels_skipped = import_eval_score_frame(
            path=path, task=Task.RETRIEVAL, source=self._DIRECT, allow_incomplete_panels=True
        )

        assert len(frame) == 1
        assert n_panels_skipped == 0

    def test_skip_incomplete_panels_drops_short_panels_and_reports_the_count(
        self,
        tmp_path: Path,
    ) -> None:
        # One complete K=2 panel, one short K=5 panel: the short one is dropped and
        # counted so the score artifact records the reduced population.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,2",
                "q1,c2,0,0,1,r1,ch2,2,2",
                "q2,c3,1,0,0,r2,ch3,1,5",
            ],
        )

        frame, n_panels_skipped = import_eval_score_frame(
            path=path, task=Task.RETRIEVAL, source=self._DIRECT, skip_incomplete_panels=True
        )

        assert n_panels_skipped == 1
        assert set(frame["record_uuid"]) == {"r1"}

    def test_skip_incomplete_panels_rejects_an_input_with_no_complete_panel(
        self,
        tmp_path: Path,
    ) -> None:
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,5",
            ],
        )

        with pytest.raises(EvalInputSchemaError, match="no complete panel"):
            import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT, skip_incomplete_panels=True)

    def test_skip_and_allow_are_mutually_exclusive(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,1",
            ],
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            import_eval_score_frame(
                path=path,
                task=Task.RETRIEVAL,
                source=self._DIRECT,
                skip_incomplete_panels=True,
                allow_incomplete_panels=True,
            )

    def test_warns_instead_of_failing_without_an_n_retrieved_chunks_column(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Direct-path inputs are not required to carry export metadata; completeness is
        # unknowable without K, so the frame passes with a warning.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,record_uuid,chunk_id,chunk_rank",
                "q1,c1,1,1,0,r1,ch1,1",
            ],
        )

        with caplog.at_level(logging.WARNING, logger="pragmata.core.eval.imports"):
            frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

        assert len(frame) == 1
        assert any("n_retrieved_chunks" in message for message in caplog.messages)

    def test_warns_instead_of_failing_when_every_panel_carries_the_absent_k_sentinel(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Pre-backfill exports carry n_retrieved_chunks=-1; K is unknown, not violated.
        path = self._write(
            tmp_path,
            [
                "query,chunk,topically_relevant,evidence_sufficient,misleading,"
                "record_uuid,chunk_id,chunk_rank,n_retrieved_chunks",
                "q1,c1,1,1,0,r1,ch1,1,-1",
            ],
        )

        with caplog.at_level(logging.WARNING, logger="pragmata.core.eval.imports"):
            frame, _ = import_eval_score_frame(path=path, task=Task.RETRIEVAL, source=self._DIRECT)

        assert len(frame) == 1
        assert any("n_retrieved_chunks" in message for message in caplog.messages)

    def test_completeness_guard_does_not_apply_to_grounding(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "answer,context_set,support_present,unsupported_claim_present,"
                "contradicted_claim_present,source_cited,fabricated_source,record_uuid",
                "a1,ctx,1,0,0,1,0,r1",
            ],
        )

        frame, _ = import_eval_score_frame(path=path, task=Task.GROUNDING, source=self._DIRECT)

        assert len(frame) == 1
