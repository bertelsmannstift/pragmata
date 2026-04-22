"""Path bundle and artifact paths for synthetic query generation runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pragmata.core.paths.paths import WorkspacePaths


@dataclass(frozen=True, slots=True)
class QueryGenRunPaths:
    """Path bundle for a synthetic query generation run.

    Attributes:
       tool_root: Root directory for the query generation tool.
       run_dir: Root directory for the query generation run.
       synthetic_queries_csv: Output path for the generated query rows CSV.
       synthetic_queries_meta_json: Output path for the dataset metadata.
       planning_summary_artifact_json: Output path for the planning-summary artifact.
    """

    tool_root: Path
    run_dir: Path
    synthetic_queries_csv: Path
    synthetic_queries_meta_json: Path
    planning_summary_artifact_json: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the run directory scaffold."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_querygen_paths(
    *,
    workspace: WorkspacePaths,
    run_id: str,
    spec_fingerprint: str,
) -> QueryGenRunPaths:
    """Build the path bundle for a synthetic query generation run.

    Args:
        workspace: Workspace path bundle.
        run_id: Unique run identifier.
        spec_fingerprint: Fingerprint of the resolved querygen spec.

    Returns:
        Path bundle for the query generation run.
    """
    tool_root = workspace.tool_root("querygen")
    run_dir = workspace.run_root(tool="querygen", run_id=run_id)

    return QueryGenRunPaths(
        tool_root=tool_root,
        run_dir=run_dir,
        synthetic_queries_csv=run_dir / "synthetic_queries.csv",
        synthetic_queries_meta_json=run_dir / "synthetic_queries.meta.json",
        planning_summary_artifact_json=tool_root / f"{spec_fingerprint}.json",
    )
