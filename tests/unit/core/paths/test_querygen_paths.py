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

    paths = resolve_querygen_paths(workspace=workspace, run_id=run_id)
    expected_run_dir = workspace.base_dir / "querygen" / "runs" / run_id

    assert paths == QueryGenRunPaths(
        run_dir=expected_run_dir,
        synthetic_queries_csv=expected_run_dir / "synthetic_queries.csv",
        synthetic_queries_meta_json=expected_run_dir / "synthetic_queries.meta.json",
    )


def test_ensure_dirs_creates_run_directory_and_returns_self(
    workspace: WorkspacePaths,
) -> None:
    """ensure_dirs creates the run directory scaffold and returns self."""
    paths = resolve_querygen_paths(workspace=workspace, run_id="new-run")

    assert not paths.run_dir.exists()

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.run_dir.is_dir()
