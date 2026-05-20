"""Path bundles for eval score runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pragmata.core.paths.paths import WorkspacePaths


@dataclass(frozen=True, slots=True)
class EvalPaths:
    """Path bundle for an eval score run.

    Attributes:
        tool_root: Root directory for the eval tool.
        score_dir: Root directory for a specific eval score run.
        scores_retrieval_json: Output path for retrieval score metrics.
        scores_grounding_json: Output path for grounding score metrics.
        scores_generation_json: Output path for generation score metrics.
        scores_meta_json: Output path for score run metadata.
    """

    tool_root: Path
    score_dir: Path
    retrieval_scores_json: Path
    grounding_scores_json: Path
    generation_scores_json: Path
    scores_meta_json: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the eval score directory scaffold."""
        self.score_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_eval_score_paths(
    *,
    workspace: WorkspacePaths,
    score_id: str,
) -> EvalPaths:
    """Build the path bundle for an eval score run.

    Args:
        workspace: Workspace path bundle.
        score_id: Unique score run identifier.

    Returns:
        Path bundle for the eval score run.
    """
    tool_root = workspace.tool_root("eval")
    score_dir = tool_root / "scores" / score_id

    return EvalPaths(
        tool_root=tool_root,
        score_dir=score_dir,
        retrieval_scores_json=score_dir / "retrieval_scores.json",
        grounding_scores_json=score_dir / "grounding_scores.json",
        generation_scores_json=score_dir / "generation_scores.json",
        scores_meta_json=score_dir / "scores_meta.json",
    )
