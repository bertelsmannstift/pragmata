"""Unit tests for api/ annotation orchestration layer.

Tests settings resolution, delegation to core/, and result assembly.
No Argilla server required — all SDK calls are mocked.
"""

from unittest.mock import MagicMock, patch

from pragmata.api.annotation_import import ImportResult, import_records
from pragmata.api.annotation_setup import setup, teardown
from pragmata.core.annotation.setup import SetupResult
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec


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
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_delegates_to_core(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult(created_workspaces=["ws1"])
        mock_users.return_value = SetupResult(created_users=["alice"])
        client = MagicMock()

        setup(client, workspace_prefix="test")

        mock_ds.assert_called_once()
        mock_users.assert_called_once()
        assert client is mock_ds.call_args[0][0]

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_merges_dataset_and_user_results(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult(created_workspaces=["ws1"], created_datasets=["ds1"])
        mock_users.return_value = SetupResult(created_users=["alice"])
        client = MagicMock()

        result = setup(client, workspace_prefix="test")

        assert result.created_workspaces == ["ws1"]
        assert result.created_datasets == ["ds1"]
        assert result.created_users == ["alice"]

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_resolves_workspace_prefix(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()
        client = MagicMock()

        setup(client, workspace_prefix="myprefix")

        settings: AnnotationSettings = mock_ds.call_args[0][1]
        assert settings.workspace_prefix == "myprefix"

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_resolves_min_submitted(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()
        client = MagicMock()

        setup(client, workspace_prefix="test", min_submitted=3)

        settings: AnnotationSettings = mock_ds.call_args[0][1]
        assert settings.min_submitted == 3

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_passes_users_to_provision(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()
        client = MagicMock()
        users = [UserSpec(username="alice", role="annotator")]

        setup(client, users, workspace_prefix="test")

        assert mock_users.call_args[0][1] == users

    @patch("pragmata.api.annotation_setup.provision_users")
    @patch("pragmata.api.annotation_setup.setup_datasets")
    def test_none_users_passes_empty_list(self, mock_ds: MagicMock, mock_users: MagicMock) -> None:
        mock_ds.return_value = SetupResult()
        mock_users.return_value = SetupResult()
        client = MagicMock()

        setup(client, workspace_prefix="test")

        assert mock_users.call_args[0][1] == []


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


class TestTeardown:
    @patch("pragmata.api.annotation_setup.teardown_resources")
    def test_delegates_to_core(self, mock_teardown: MagicMock) -> None:
        client = MagicMock()
        teardown(client, workspace_prefix="test")
        mock_teardown.assert_called_once()
        assert client is mock_teardown.call_args[0][0]

    @patch("pragmata.api.annotation_setup.teardown_resources")
    def test_resolves_workspace_prefix(self, mock_teardown: MagicMock) -> None:
        client = MagicMock()
        teardown(client, workspace_prefix="myprefix")
        settings: AnnotationSettings = mock_teardown.call_args[0][1]
        assert settings.workspace_prefix == "myprefix"


# ---------------------------------------------------------------------------
# import_records()
# ---------------------------------------------------------------------------


class TestImportRecords:
    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_delegates_to_core(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {"ds1": 2}
        client = MagicMock()
        raw = [_make_raw()]

        import_records(client, raw, workspace_prefix="test")

        mock_fan_out.assert_called_once()
        assert client is mock_fan_out.call_args[0][0]
        assert len(mock_fan_out.call_args[0][1]) == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_resolves_workspace_prefix(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {}
        client = MagicMock()

        import_records(client, [], workspace_prefix="myprefix")

        settings: AnnotationSettings = mock_fan_out.call_args[0][2]
        assert settings.workspace_prefix == "myprefix"

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_returns_import_result(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {"ds1": 3, "ds2": 1}
        client = MagicMock()
        raw = [_make_raw(), _make_raw()]

        result = import_records(client, raw, workspace_prefix="test")

        assert isinstance(result, ImportResult)
        assert result.total_records == 2
        assert result.dataset_counts == {"ds1": 3, "ds2": 1}
        assert result.errors == []

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_empty_records_returns_zero_totals(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {}
        client = MagicMock()

        result = import_records(client, [], workspace_prefix="test")

        assert result.total_records == 0
        assert result.dataset_counts == {}
        assert result.errors == []

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_validation_errors_reported(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {"ds1": 1}
        client = MagicMock()
        raw = [_make_raw(), {"query": "missing required fields"}]

        result = import_records(client, raw, workspace_prefix="test")

        assert result.total_records == 2
        assert len(result.errors) == 1
        assert result.errors[0].index == 1
        # Only the valid record was passed to fan_out
        assert len(mock_fan_out.call_args[0][1]) == 1

    @patch("pragmata.api.annotation_import.fan_out_records")
    def test_all_invalid_skips_fan_out(self, mock_fan_out: MagicMock) -> None:
        mock_fan_out.return_value = {}
        client = MagicMock()
        raw = [{"bad": "data"}, {"also": "bad"}]

        result = import_records(client, raw, workspace_prefix="test")

        assert result.total_records == 2
        assert len(result.errors) == 2
        # fan_out called with empty list
        assert mock_fan_out.call_args[0][1] == []
