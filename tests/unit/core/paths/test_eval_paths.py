"""Tests for eval score run path bundle."""

from pathlib import Path

import pytest

from pragmata.core.paths.eval_paths import EvalPaths, resolve_eval_score_paths
from pragmata.core.paths.paths import WorkspacePaths


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    """Workspace path bundle rooted in a temporary directory."""
    return WorkspacePaths.from_base_dir(tmp_path)


def test_resolve_eval_score_paths_returns_expected_bundle(
    workspace: WorkspacePaths,
) -> None:
    """Resolver returns the expected score run and artifact paths."""
    score_id = "score-26"

    paths = resolve_eval_score_paths(
        workspace=workspace,
        score_id=score_id,
    )
    expected_tool_root = workspace.base_dir / "eval"
    expected_score_dir = workspace.base_dir / "eval" / "scores" / score_id

    assert paths == EvalPaths(
        tool_root=expected_tool_root,
        score_dir=expected_score_dir,
        retrieval_scores_json=expected_score_dir / "retrieval_scores.json",
        grounding_scores_json=expected_score_dir / "grounding_scores.json",
        generation_scores_json=expected_score_dir / "generation_scores.json",
        scores_meta_json=expected_score_dir / "scores.meta.json",
    )


def test_ensure_dirs_creates_score_directory_and_returns_self(
    workspace: WorkspacePaths,
) -> None:
    """ensure_dirs creates the score directory scaffold and returns self."""
    paths = resolve_eval_score_paths(
        workspace=workspace,
        score_id="new-score",
    )

    assert not paths.score_dir.exists()
    assert not paths.tool_root.exists()

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.tool_root.is_dir()
    assert paths.score_dir.is_dir()


def test_score_artifact_paths_are_score_id_specific(
    workspace: WorkspacePaths,
) -> None:
    """Score artifact paths depend on the score ID."""
    first = resolve_eval_score_paths(
        workspace=workspace,
        score_id="score-a",
    )
    second = resolve_eval_score_paths(
        workspace=workspace,
        score_id="score-b",
    )

    assert first.score_dir != second.score_dir
    assert first.retrieval_scores_json != second.retrieval_scores_json
    assert first.grounding_scores_json != second.grounding_scores_json
    assert first.generation_scores_json != second.generation_scores_json
    assert first.scores_meta_json != second.scores_meta_json

    assert first.score_dir.parent == second.score_dir.parent == first.tool_root / "scores"
