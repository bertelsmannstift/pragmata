"""Unit tests for the always-suffixed Argilla dataset name helper."""

import pytest

from pragmata.core.annotation.argilla_task_definitions import dataset_name
from pragmata.core.schemas.annotation_task import Task


class TestDatasetName:
    @pytest.mark.parametrize(
        "task,expected_base",
        [
            (Task.RETRIEVAL, "retrieval"),
            (Task.GROUNDING, "grounding"),
            (Task.GENERATION, "generation"),
        ],
    )
    def test_production_suffix_no_dataset_id(self, task: Task, expected_base: str) -> None:
        assert dataset_name(task, calibration=False) == f"{expected_base}_production"

    @pytest.mark.parametrize(
        "task,expected_base",
        [
            (Task.RETRIEVAL, "retrieval"),
            (Task.GROUNDING, "grounding"),
            (Task.GENERATION, "generation"),
        ],
    )
    def test_calibration_suffix_no_dataset_id(self, task: Task, expected_base: str) -> None:
        assert dataset_name(task, calibration=True) == f"{expected_base}_calibration"

    def test_dataset_id_appended(self) -> None:
        assert dataset_name(Task.RETRIEVAL, calibration=False, dataset_id="run1") == "retrieval_production_run1"
        assert dataset_name(Task.RETRIEVAL, calibration=True, dataset_id="run1") == "retrieval_calibration_run1"

    def test_empty_dataset_id_no_extra_suffix(self) -> None:
        assert dataset_name(Task.GENERATION, calibration=True, dataset_id="") == "generation_calibration"
