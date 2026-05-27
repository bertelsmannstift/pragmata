"""Unit tests for fan_out_records: per-purpose dataset routing and logging.

These exercise the routing/branching contract directly (rather than via the
API tests, which patch fan_out_records out).
"""

from unittest.mock import MagicMock, patch

import pytest

from pragmata.core.annotation.record_builder import fan_out_records
from pragmata.core.schemas.annotation_import import Chunk, QueryResponsePair
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings, TaskSettings, WorkspaceSettings


def _make_pair(query: str) -> QueryResponsePair:
    return QueryResponsePair(
        query=query,
        answer="A.",
        chunks=[Chunk(chunk_id=f"c-{query}", doc_id="d", chunk_rank=1, text="t")],
        context_set="ctx",
    )


def _settings(*, calibration_enabled: bool = True) -> AnnotationSettings:
    def _task() -> TaskSettings:
        return TaskSettings(calibration_min_submitted=3 if calibration_enabled else None)

    return AnnotationSettings(
        dataset_id="run1",
        # When calibration_min_submitted=None for every task, calibration_fraction
        # must be 0 to satisfy the AnnotationSettings model validator.
        calibration_fraction=0.5 if calibration_enabled else 0.0,
        workspaces={
            "retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: _task()}),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: _task()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: _task()}),
        },
    )


@pytest.fixture
def fake_dataset_factory():
    """Builds a MagicMock dataset whose .name reflects the requested ds_name."""

    def _factory(ds_name: str) -> MagicMock:
        ds = MagicMock(name=f"dataset[{ds_name}]")
        ds.name = ds_name
        return ds

    return _factory


@pytest.fixture
def patched_create_dataset(fake_dataset_factory):
    """Patch create_dataset so each call returns a fresh fake dataset."""
    created: dict[tuple[str, str], MagicMock] = {}

    def _create(client, ds_name, ws_base, task_cfg):
        ds = fake_dataset_factory(ds_name)
        created[(ws_base, ds_name)] = ds
        return ds, True

    with patch("pragmata.core.annotation.record_builder.create_dataset", side_effect=_create) as m:
        yield m, created


class TestFanOutRecords:
    def test_empty_records_returns_empty_counts(self, patched_create_dataset) -> None:
        client = MagicMock()
        result = fan_out_records(client, records=[], settings=_settings(), assignments={})
        assert result == {}

    def test_production_only_logs_to_production_datasets(self, patched_create_dataset) -> None:
        client = MagicMock()
        _, created = patched_create_dataset
        records = [_make_pair(f"q{i}") for i in range(3)]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        assignments = {derive_record_uuid(r): False for r in records}

        result = fan_out_records(client, records=records, settings=_settings(), assignments=assignments)

        # One production dataset per task; no calibration datasets created.
        assert all(ds_name.endswith("_production_run1") for (_, ds_name) in created)
        assert len(result) == 3  # retrieval, grounding, generation
        for ds in created.values():
            ds.records.log.assert_called_once()

    def test_calibration_only_logs_to_calibration_datasets(self, patched_create_dataset) -> None:
        client = MagicMock()
        _, created = patched_create_dataset
        records = [_make_pair(f"q{i}") for i in range(2)]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        assignments = {derive_record_uuid(r): True for r in records}

        result = fan_out_records(client, records=records, settings=_settings(), assignments=assignments)

        assert all(ds_name.endswith("_calibration_run1") for (_, ds_name) in created)
        assert len(result) == 3

    def test_mixed_assignments_create_both_datasets_per_task(self, patched_create_dataset) -> None:
        client = MagicMock()
        _, created = patched_create_dataset
        records = [_make_pair(f"q{i}") for i in range(4)]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        # First two records calibration, last two production.
        assignments = {derive_record_uuid(r): (i < 2) for i, r in enumerate(records)}

        fan_out_records(client, records=records, settings=_settings(), assignments=assignments)

        purposes = {ds_name.split("_")[-2] for (_, ds_name) in created}
        assert purposes == {"calibration", "production"}
        # 3 tasks * 2 purposes = 6 datasets created.
        assert len(created) == 6

    def test_calibration_assigned_when_topology_disables_raises(self, patched_create_dataset) -> None:
        client = MagicMock()
        records = [_make_pair("q0")]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        assignments = {derive_record_uuid(records[0]): True}

        with pytest.raises(RuntimeError, match="topology disables calibration"):
            fan_out_records(
                client,
                records=records,
                settings=_settings(calibration_enabled=False),
                assignments=assignments,
            )

    def test_dataset_counts_reflect_per_dataset_record_counts(self, patched_create_dataset) -> None:
        client = MagicMock()
        records = [_make_pair(f"q{i}") for i in range(5)]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        # Three calibration, two production - same split per task.
        assignments = {derive_record_uuid(r): (i < 3) for i, r in enumerate(records)}

        result = fan_out_records(client, records=records, settings=_settings(), assignments=assignments)

        # Each task contributes one production count and one calibration count;
        # each grouping reflects the assignment split.
        assert sum(result.values()) == len(records) * 3  # 3 tasks
        cal_total = sum(c for ds, c in result.items() if "_calibration_" in ds)
        prod_total = sum(c for ds, c in result.items() if "_production_" in ds)
        assert cal_total == 3 * 3  # 3 cal records * 3 tasks
        assert prod_total == 2 * 3  # 2 prod records * 3 tasks

    def test_skips_tasks_not_in_workspaces_topology(self, patched_create_dataset, caplog) -> None:
        client = MagicMock()
        _, created = patched_create_dataset

        # Topology only declares retrieval; grounding/generation are absent.
        partial_settings = AnnotationSettings(
            dataset_id="run1",
            workspaces={"retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        records = [_make_pair("q0")]
        from pragmata.core.annotation.record_builder import derive_record_uuid

        assignments = {derive_record_uuid(records[0]): False}

        with caplog.at_level("WARNING"):
            result = fan_out_records(client, records=records, settings=partial_settings, assignments=assignments)

        # Only retrieval gets a dataset; grounding and generation are skipped with a warning.
        assert len(result) == 1
        assert any(ds_name.startswith("retrieval_") for (_, ds_name) in created)
        assert "not in workspaces topology" in caplog.text


class TestImportLocaleConflict:
    """Re-importing into an existing dataset with a different locale logs a warning and proceeds.

    Label *values* are locale-invariant, so the data is still safe to append;
    only the displayed text differs. The mismatch is surfaced as a warning so
    operators can investigate, not raised.
    """

    def test_relocale_warn_append(self, fake_dataset_factory, caplog) -> None:
        import argilla as rg

        from pragmata.core.annotation.argilla_task_definitions import build_task_settings, dataset_name
        from pragmata.core.annotation.record_builder import derive_record_uuid

        # Build an EN-flavoured existing dataset; _detect_dataset_locale will
        # match its label displays against the EN catalog.
        from pragmata.core.settings.annotation_settings import AnnotationSettings

        en_settings = build_task_settings(AnnotationSettings(locale="en"))[Task.RETRIEVAL]
        ds_name = dataset_name(Task.RETRIEVAL, calibration=False, dataset_id="run1")
        fake_dataset = fake_dataset_factory(ds_name)
        fake_dataset.settings = rg.Settings(
            fields=en_settings.fields,
            questions=en_settings.questions,
            metadata=en_settings.metadata,
            guidelines=en_settings.guidelines,
        )

        def _create(client, name, ws_base, task_cfg):
            # Simulate an existing dataset: ds_created=False, returning the EN-flavoured one.
            return fake_dataset, False

        records = [_make_pair("q0")]
        assignments = {derive_record_uuid(records[0]): False}

        # DE-locale settings — mismatch against the EN existing dataset.
        settings_de = AnnotationSettings(
            dataset_id="run1",
            locale="de",
            calibration_fraction=0.0,
            workspaces={"retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )

        with patch("pragmata.core.annotation.record_builder.create_dataset", side_effect=_create):
            with caplog.at_level("WARNING", logger="pragmata.core.annotation.record_builder"):
                result = fan_out_records(MagicMock(), records=records, settings=settings_de, assignments=assignments)

        # Records were appended (no error raised), and a warning was logged.
        # Assert against locale .name rather than the %r-formatted .value so the
        # test isn't tied to the log formatter's quote style.
        assert len(result) == 1
        assert "Locale mismatch" in caplog.text
        assert "en" in caplog.text
        assert "de" in caplog.text
        fake_dataset.records.log.assert_called_once()
