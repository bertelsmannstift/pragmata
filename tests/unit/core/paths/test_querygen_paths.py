"""Tests for synthetic query generation run path bundle."""

from pathlib import Path

import pytest

from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.paths.querygen_paths import (
    QueryGenRunPaths,
    resolve_querygen_paths,
)


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    """Workspace path bundle rooted in a temporary directory."""
    return WorkspacePaths.from_base_dir(tmp_path)


def test_resolve_querygen_paths_returns_expected_bundle(
    workspace: WorkspacePaths,
) -> None:
    """Resolver returns the expected run and artifact paths."""
    run_id = "run-26"
    spec_fingerprint = "abc123fingerprint"

    paths = resolve_querygen_paths(
        workspace=workspace,
        run_id=run_id,
        spec_fingerprint=spec_fingerprint,
    )
    expected_tool_root = workspace.base_dir / "querygen"
    expected_run_dir = workspace.base_dir / "querygen" / "runs" / run_id

    assert paths == QueryGenRunPaths(
        tool_root=expected_tool_root,
        run_dir=expected_run_dir,
        synthetic_queries_csv=expected_run_dir / "synthetic_queries.csv",
        synthetic_queries_meta_json=expected_run_dir / "synthetic_queries.meta.json",
        planning_summary_artifact_json=expected_tool_root / f"{spec_fingerprint}.json",
    )


def test_ensure_dirs_creates_run_directory_and_returns_self(
    workspace: WorkspacePaths,
) -> None:
    """ensure_dirs creates the run directory scaffold and returns self."""
    paths = resolve_querygen_paths(
        workspace=workspace,
        run_id="new-run",
        spec_fingerprint="fingerprint-001",
    )

    assert not paths.run_dir.exists()
    assert not paths.tool_root.exists()

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.tool_root.is_dir()
    assert paths.run_dir.is_dir()


def test_planning_summary_artifact_path_is_fingerprint_specific_and_run_independent(
    workspace: WorkspacePaths,
) -> None:
    """Reusable planning-summary path depends on fingerprint, not run directory."""
    first = resolve_querygen_paths(
        workspace=workspace,
        run_id="run-a",
        spec_fingerprint="fingerprint-001",
    )
    second = resolve_querygen_paths(
        workspace=workspace,
        run_id="run-b",
        spec_fingerprint="fingerprint-001",
    )
    third = resolve_querygen_paths(
        workspace=workspace,
        run_id="run-a",
        spec_fingerprint="fingerprint-002",
    )

    assert first.planning_summary_artifact_json == second.planning_summary_artifact_json
    assert first.planning_summary_artifact_json != third.planning_summary_artifact_json
    assert first.planning_summary_artifact_json.parent == first.tool_root


def test_ensure_dirs_keeps_planning_summary_parent_available(
    workspace: WorkspacePaths,
) -> None:
    """ensure_dirs creates the reusable planning-summary parent directory implicitly."""
    paths = resolve_querygen_paths(
        workspace=workspace,
        run_id="run-ensure",
        spec_fingerprint="fingerprint-001",
    )

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.planning_summary_artifact_json.parent.is_dir()
