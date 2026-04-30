"""Unit tests for api/ annotation orchestration layer.

Tests settings resolution, delegation to core/, and result assembly.
No Argilla server required — all SDK calls are mocked.
"""

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pragmata.api.annotation_import import ImportResult, import_records
from pragmata.api.annotation_setup import setup, teardown
from pragmata.core.annotation.setup import SetupResult
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec


@pytest.fixture(autouse=True)
def _isolated_argilla_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from ambient ARGILLA_API_URL; provide a fake ARGILLA_API_KEY."""
    monkeypatch.delenv("ARGILLA_API_URL", raising=False)
    monkeypatch.setenv("ARGILLA_API_KEY", "test-key")


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace resolve_argilla_client in both API modules with a stub returning this mock."""
    fake_client = MagicMock()

    import pragmata.api.annotation_import as import_module
    import pragmata.api.annotation_setup as setup_module

    monkeypatch.setattr(setup_module, "resolve_argilla_client", lambda api_url, api_key: fake_client)
    monkeypatch.setattr(import_module, "resolve_argilla_client", lambda api_url, api_key: fake_client)
    return fake_client


@pytest.fixture(autouse=True)
def _autouse_mock_client(mock_client: MagicMock) -> MagicMock:
    """Force mock_client to be active for every test in this module."""
    return mock_client


@pytest.fixture(autouse=True)
def _isolate_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every test under a tmp directory so partition manifests etc. write there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_raw() -> dict:
    return {
        "query": "What is X?",
        "answer": "X is Y.",
        "chunks": [{"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "text": "Chunk text."}],
        "context_set": "ctx-001",
    }


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------


class TestSetup:
    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_workspaces")
    def test_delegates_to_core(
        self,
        mock_ds: MagicMock,
        mock_users: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        mock_ds.return_value = SetupResult(created_workspaces=["ws1"])
        mock_users.return_value = SetupResult(created_users=["alice"])

        setup()

        mock_ds.assert_called_once()
        mock_users.assert_called_once()
        assert mock_client is mock_ds.call_args[0][0]

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_workspaces")
    def test_merges_workspace_and_user_results(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult(created_workspaces=["ws1"])
        mock_users.return_value = SetupResult(created_users=["alice"])

        result = setup()

        assert result.created_workspaces == ["ws1"]
        assert result.created_users == ["alice"]

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_workspaces")
    def test_passes_users_to_provision(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()
        users = [UserSpec(username="alice", role="annotator")]

        setup(users)

        assert mock_users.call_args[0][1] == users

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_workspaces")
    def test_none_users_passes_empty_list(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()

        setup()

        assert mock_users.call_args[0][1] == []


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


class TestTeardown:
    @patch("pragmata.api.annotation_setup.teardown_resources")
    def test_delegates_to_core(self, mock_teardown: MagicMock, mock_client: MagicMock) -> None:
        teardown(dataset_id="test")
        mock_teardown.assert_called_once()
        assert mock_client is mock_teardown.call_args[0][0]

    @patch("pragmata.api.annotation_setup.teardown_resources")
    def test_resolves_dataset_id(self, mock_teardown: MagicMock) -> None:
        teardown(dataset_id="myprefix")
        settings: AnnotationSettings = mock_teardown.call_args[0][1]
        assert settings.dataset_id == "myprefix"


# ---------------------------------------------------------------------------
# import_records()
# ---------------------------------------------------------------------------


class TestImportRecords:
    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_delegates_to_core(self, mock_fan_out: MagicMock, mock_client: MagicMock) -> None:
        mock_fan_out.return_value = ({"ds1": 2}, 0, 1)
        raw = [_make_raw()]

        import_records(raw, dataset_id="test")

        mock_fan_out.assert_called_once()
        assert mock_client is mock_fan_out.call_args[0][0]
        assert len(mock_fan_out.call_args[0][1]) == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_resolves_dataset_id(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({}, 0, 0)

        import_records([], dataset_id="myprefix")

        settings: AnnotationSettings = mock_fan_out.call_args[0][2]
        assert settings.dataset_id == "myprefix"

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_returns_import_result(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({"ds1": 3, "ds2": 1}, 0, 2)
        raw = [_make_raw(), _make_raw()]

        result = import_records(raw, dataset_id="test", calibration_fraction=0.0)

        assert isinstance(result, ImportResult)
        assert result.total_records == 2
        assert result.dataset_counts == {"ds1": 3, "ds2": 1}
        assert result.calibration_count == 0
        assert result.production_count == 2
        assert result.errors == []

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_empty_records_returns_zero_totals(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({}, 0, 0)

        result = import_records([], dataset_id="test")

        assert result.total_records == 0
        assert result.dataset_counts == {}
        assert result.errors == []

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_validation_errors_reported(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)
        raw = [_make_raw(), {"query": "missing required fields"}]

        result = import_records(raw, dataset_id="test")

        assert result.total_records == 2
        assert len(result.errors) == 1
        assert result.errors[0].index == 1
        # Only the valid record was passed to fan_out
        assert len(mock_fan_out.call_args[0][1]) == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_all_invalid_skips_fan_out(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({}, 0, 0)
        raw = [{"bad": "data"}, {"also": "bad"}]

        result = import_records(raw, dataset_id="test")

        assert result.total_records == 2
        assert len(result.errors) == 2
        # fan_out called with empty list
        assert mock_fan_out.call_args[0][1] == []

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_accepts_json_file_path(self, mock_fan_out: MagicMock, tmp_path: Path) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)
        f = tmp_path / "data.json"
        f.write_text(json.dumps([_make_raw()]))

        result = import_records(str(f), dataset_id="test")

        assert result.total_records == 1
        mock_fan_out.assert_called_once()

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_accepts_path_object(self, mock_fan_out: MagicMock, tmp_path: Path) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)
        f = tmp_path / "data.json"
        f.write_text(json.dumps([_make_raw()]))

        result = import_records(f, dataset_id="test")

        assert result.total_records == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_accepts_jsonl_file(self, mock_fan_out: MagicMock, tmp_path: Path) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(_make_raw()) + "\n")

        result = import_records(str(f), dataset_id="test")

        assert result.total_records == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_format_override(self, mock_fan_out: MagicMock, tmp_path: Path) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)
        f = tmp_path / "data.txt"
        f.write_text(json.dumps([_make_raw()]))

        result = import_records(str(f), format="json", dataset_id="test")

        assert result.total_records == 1

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            import_records("/nonexistent/data.json", dataset_id="test")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.parquet"
        f.write_text("")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            import_records(str(f), dataset_id="test")

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_accepts_hf_dataset(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)

        FakeDataset = type("Dataset", (), {"to_list": lambda self: [_make_raw()]})
        fake_ds = FakeDataset()
        fake_mod = types.ModuleType("datasets")
        fake_mod.Dataset = FakeDataset  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"datasets": fake_mod}):
            result = import_records(fake_ds, dataset_id="test")

        assert result.total_records == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_accepts_dataframe(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = ({"ds1": 1}, 0, 1)

        FakeDataFrame = type("DataFrame", (), {"to_dict": lambda self, orient: [_make_raw()]})
        fake_df = FakeDataFrame()
        fake_mod = types.ModuleType("pandas")
        fake_mod.DataFrame = FakeDataFrame  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"pandas": fake_mod}):
            result = import_records(fake_df, dataset_id="test")

        assert result.total_records == 1


class TestImportRecordsCalibrationValidation:
    """Hard-error semantics for calibration_fraction vs topology mismatch."""

    def test_fraction_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration_fraction"):
            import_records([_make_raw()], dataset_id="t", calibration_fraction=1.5)

    def test_negative_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration_fraction"):
            import_records([_make_raw()], dataset_id="t", calibration_fraction=-0.1)

    @staticmethod
    def _no_calibration_config(tmp_path: Path) -> Path:
        """Write a YAML config that disables calibration on every task."""
        yaml_text = (
            "workspace_dataset_map:\n"
            "  retrieval:\n"
            "    retrieval:\n"
            "      production_min_submitted: 1\n"
            "      calibration_min_submitted: null\n"
            "  grounding:\n"
            "    grounding:\n"
            "      production_min_submitted: 1\n"
            "      calibration_min_submitted: null\n"
            "  generation:\n"
            "    generation:\n"
            "      production_min_submitted: 1\n"
            "      calibration_min_submitted: null\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml_text)
        return path

    def test_fraction_zero_with_no_calibration_topology_ok(self, tmp_path: Path) -> None:
        """`calibration_fraction=0` is fine even when topology disables calibration."""
        config_path = self._no_calibration_config(tmp_path)

        with patch("pragmata.api.annotation_import.fan_out_records") as mock_fan_out:
            mock_fan_out.return_value = ({"retrieval_production": 1}, 0, 1)
            result = import_records([_make_raw()], dataset_id="t", calibration_fraction=0.0, config_path=config_path)
        assert result.calibration_count == 0

    def test_fraction_positive_with_no_calibration_topology_raises(self, tmp_path: Path) -> None:
        config_path = self._no_calibration_config(tmp_path)

        with pytest.raises(ValueError, match="calibration_fraction"):
            import_records([_make_raw()], dataset_id="t", calibration_fraction=0.1, config_path=config_path)
