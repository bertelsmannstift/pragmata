"""Integration tests for the public evaluator training surface."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pragmata import eval
from pragmata.core.paths.annotation_paths import resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_export import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task

pytestmark = [pytest.mark.integration, pytest.mark.eval]

_TINY_CHECKPOINT = "google/bert_uncased_L-2_H-128_A-2"

_RETRIEVAL_EXAMPLES = [
    {
        "query": "How can municipalities involve residents in local climate policy?",
        "chunk": (
            "The study compares municipal climate plans and finds that participation works best when residents are "
            "invited before policy options are finalized. Citizen panels, local workshops, and public response reports "
            "made it easier to see how recommendations changed the final plan."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
    {
        "query": "Which measures reduce educational inequality between students?",
        "chunk": (
            "The education monitor identifies early language support, all-day school programs, targeted tutoring, and "
            "cooperation with youth services as measures associated with weaker links between family background and "
            "student achievement."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
    {
        "query": "What supports social cohesion in diverse neighborhoods?",
        "chunk": (
            "Neighborhood interviews point to accessible meeting places, trusted local institutions, and low-threshold "
            "volunteering opportunities. The report stresses that everyday contact matters more than one-off symbolic "
            "events."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
    {
        "query": "How should public agencies use algorithms responsibly?",
        "chunk": (
            "The policy paper recommends documentation, human oversight, bias testing, and clear appeal mechanisms for "
            "automated public-sector decisions. It also calls for procurement rules that expose model limitations."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
    {
        "query": "What can regions do in response to demographic change?",
        "chunk": (
            "The regional case studies mention family-friendly infrastructure and age-friendly services, "
            "but the excerpt does not describe labor-market effects or migration strategies in enough detail "
            "to answer the question."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 0,
        "misleading": 0,
    },
    {
        "query": "How can foundations evaluate civic education programs?",
        "chunk": (
            "The evaluation note lists participant surveys and workshop attendance, but it does not define "
            "outcomes such as deliberation skills, political efficacy, or later participation behavior."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 0,
        "misleading": 0,
    },
    {
        "query": "Which indicators are useful for integration monitoring?",
        "chunk": (
            "The publication on municipal debt explains how local governments report investment backlogs, "
            "tax capacity, and borrowing limits. It does not discuss migration, language acquisition, "
            "or participation indicators."
        ),
        "topically_relevant": 0,
        "evidence_sufficient": 0,
        "misleading": 0,
    },
    {
        "query": "Why is fiscal transparency important for municipal governance?",
        "chunk": (
            "A chapter on school cooperation describes how teachers coordinate with youth centers and local clubs. The "
            "passage does not address budgets, public finance, or accountability for municipal spending."
        ),
        "topically_relevant": 0,
        "evidence_sufficient": 0,
        "misleading": 0,
    },
    {
        "query": "What strengthens youth trust in democracy?",
        "chunk": (
            "The report claims that youth trust improves mainly when formal participation channels are reduced and "
            "political education avoids controversial issues. This reverses the publication's conclusion about "
            "meaningful participation and practical civic learning."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 0,
        "misleading": 1,
    },
    {
        "query": "How can local governments improve access to preventive healthcare?",
        "chunk": (
            "The excerpt says outreach programs should focus exclusively on digital self-service portals and close "
            "in-person advisory points. This conflicts with the study's emphasis on mobile services and low-threshold "
            "local outreach."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 0,
        "misleading": 1,
    },
    {
        "query": "What helps workers adapt to digital transformation?",
        "chunk": (
            "The labor-market report highlights lifelong learning accounts, transparent certification of skills, and "
            "regional training networks, especially for workers in small and medium-sized enterprises."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
    {
        "query": "What makes cooperation between schools and civil society successful?",
        "chunk": (
            "The guide emphasizes stable contact persons, shared goals, realistic time budgets, "
            "safeguarding procedures, and formats that connect classroom learning with local social challenges."
        ),
        "topically_relevant": 1,
        "evidence_sufficient": 1,
        "misleading": 0,
    },
]


def _retrieval_rows(
    *,
    marker: str,
    n_rows: int = 24,
) -> list[dict[str, object]]:
    """Build a small retrieval training set with multiple binary label patterns."""
    rows: list[dict[str, object]] = []

    for idx in range(n_rows):
        example = _RETRIEVAL_EXAMPLES[idx % len(_RETRIEVAL_EXAMPLES)]
        rows.append(
            {
                "record_uuid": f"{marker}-record-{idx:02d}",
                "chunk_id": f"{marker}-chunk-{idx:02d}",
                "query": f"{marker}: {example['query']}",
                "chunk": f"{marker}: {example['chunk']}",
                "topically_relevant": example["topically_relevant"],
                "evidence_sufficient": example["evidence_sufficient"],
                "misleading": example["misleading"],
            }
        )

    return rows


def _write_retrieval_csv(
    path: Path,
    *,
    marker: str,
) -> None:
    """Write valid Pragmata retrieval evaluator training data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_retrieval_rows(marker=marker)).to_csv(path, index=False)


def _write_retrieval_export(
    *,
    workspace: WorkspacePaths,
    export_id: str,
    created_at: datetime,
    marker: str,
) -> None:
    """Write a minimal retrieval annotation export consumable by eval train."""
    export_paths = resolve_export_paths(
        workspace=workspace,
        export_id=export_id,
    ).ensure_dirs()
    _write_retrieval_csv(export_paths.retrieval_annotation_csv, marker=marker)

    meta = AnnotationExportMeta(
        export_id=export_id,
        created_at=created_at,
        dataset_id=None,
        tasks=[Task.RETRIEVAL],
        include_discarded=False,
        row_counts={Task.RETRIEVAL: 24},
        n_annotators={Task.RETRIEVAL: 1},
        calibration_enabled={},
        constraint_summary={},
    )
    export_paths.export_meta_json.write_text(
        meta.model_dump_json(),
        encoding="utf-8",
    )


def _fast_train_kwargs(
    *,
    run_id: str,
    transfer_learning: bool = False,
) -> dict[str, Any]:
    """Return tlmtc settings for quick integration-test training runs."""
    train_kwargs: dict[str, Any] = {
        "run_id": run_id,
        "validation_size": 0.25,
        "test_size": 0.25,
        "random_seed": 2469,
        "transfer_learning": transfer_learning,
        "hyperparameter_tuning": False,
        "threshold_optimization": False,
        "use_cpu": True,
        "verbosity": "quiet",
    }

    if transfer_learning:
        train_kwargs.update(
            {
                "batch_size": 4,
                "train_epochs": 1,
                "wrap_peft": False,
            }
        )

    return train_kwargs


def _train_evaluator(
    *,
    base_dir: Path,
    run_id: str,
    transfer_learning: bool = False,
    **kwargs: Any,
) -> Any:
    """Call the public eval API with fast integration-test defaults."""
    return eval.train_evaluator(
        base_dir=base_dir,
        task=Task.RETRIEVAL,
        checkpoint=_TINY_CHECKPOINT,
        proxy_checkpoint=_TINY_CHECKPOINT,
        scale_learning_rate=False,
        sequence_length=64,
        train_kwargs=_fast_train_kwargs(
            run_id=run_id,
            transfer_learning=transfer_learning,
        ),
        **kwargs,
    )


def _assert_prepared_splits(
    *,
    result: Any,
    base_dir: Path,
    run_id: str,
    expected_marker: str,
    unexpected_marker: str | None = None,
) -> None:
    """Assert tlmtc produced split artifacts from the expected in-memory frame."""
    paths = result.paths

    assert paths.work_dir == (base_dir / "eval").resolve()
    assert paths.run_id == run_id
    assert paths.run_dir == (base_dir / "eval" / "train_outputs" / run_id).resolve()
    assert getattr(paths, "labeled_data_path") is None

    split_paths = [
        paths.train_data_path,
        paths.val_data_path,
        paths.test_data_path,
    ]
    assert all(path.is_file() for path in split_paths)

    split_frame = pd.concat(
        [pd.read_parquet(path) for path in split_paths],
        ignore_index=True,
    )
    text_values = set(split_frame["text"])

    assert any(str(value).startswith(f"{expected_marker}: ") for value in text_values)
    if unexpected_marker is not None:
        assert not any(str(value).startswith(f"{unexpected_marker}: ") for value in text_values)


def _assert_pragmata_train_meta(
    *,
    result: Any,
    run_id: str,
    annotation_export_id: str | None,
) -> None:
    """Assert Pragmata persisted train-run metadata beside tlmtc artifacts."""
    meta_path = result.paths.run_dir / "pragmata_train.meta.json"

    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id
    assert meta["task"] == "retrieval"
    assert meta["annotation_export_id"] == annotation_export_id
    assert isinstance(meta["created_at"], str)


class TestTrainEvaluator:
    """Integration tests for evaluator training."""

    def test_runs_tlmtc_from_direct_labeled_csv(
        self,
        tmp_path: Path,
    ) -> None:
        """Direct CSV input is imported, transformed, and handed to real tlmtc training."""
        labeled_data_path = tmp_path / "inputs" / "retrieval.csv"
        _write_retrieval_csv(labeled_data_path, marker="direct")

        result = _train_evaluator(
            base_dir=tmp_path,
            run_id="direct-input",
            transfer_learning=True,
            labeled_data_path=labeled_data_path,
        )

        _assert_prepared_splits(
            result=result,
            base_dir=tmp_path,
            run_id="direct-input",
            expected_marker="direct",
        )
        assert result.paths.train_run_meta_path.is_file()
        _assert_pragmata_train_meta(
            result=result,
            run_id="direct-input",
            annotation_export_id=None,
        )
        assert result.paths.model_dir.is_dir()
        assert any(result.paths.model_dir.iterdir())

    def test_uses_explicit_annotation_export(
        self,
        tmp_path: Path,
    ) -> None:
        """Explicit export IDs resolve to the selected task CSV before tlmtc training."""
        workspace = WorkspacePaths.from_base_dir(tmp_path)
        _write_retrieval_export(
            workspace=workspace,
            export_id="export-a",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            marker="selected",
        )
        _write_retrieval_export(
            workspace=workspace,
            export_id="export-b",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            marker="unselected",
        )

        result = _train_evaluator(
            base_dir=tmp_path,
            run_id="explicit-export",
            export_id="export-a",
        )

        _assert_prepared_splits(
            result=result,
            base_dir=tmp_path,
            run_id="explicit-export",
            expected_marker="selected",
            unexpected_marker="unselected",
        )
        _assert_pragmata_train_meta(
            result=result,
            run_id="explicit-export",
            annotation_export_id="export-a",
        )

    def test_uses_latest_annotation_export_for_task(
        self,
        tmp_path: Path,
    ) -> None:
        """Omitting export_id selects the newest export containing the requested task CSV."""
        workspace = WorkspacePaths.from_base_dir(tmp_path)
        _write_retrieval_export(
            workspace=workspace,
            export_id="older-export",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            marker="older",
        )
        _write_retrieval_export(
            workspace=workspace,
            export_id="newer-export",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            marker="newer",
        )

        result = _train_evaluator(
            base_dir=tmp_path,
            run_id="latest-export",
        )

        _assert_prepared_splits(
            result=result,
            base_dir=tmp_path,
            run_id="latest-export",
            expected_marker="newer",
            unexpected_marker="older",
        )
        _assert_pragmata_train_meta(
            result=result,
            run_id="latest-export",
            annotation_export_id="newer-export",
        )
