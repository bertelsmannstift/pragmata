"""Shared workspace path resolver."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class WorkspacePaths:
    """Workspace path bundle.

    Attributes:
        base_dir: Workspace root stored as an expanded, absolute path.
    """

    base_dir: Path

    @classmethod
    def from_base_dir(
        cls,
        base_dir: Path,
    ) -> Self:
        """Create a workspace path bundle from a base directory.

        Args:
            base_dir: Workspace root as a `str` or `Path`.

        Returns:
            A `WorkspacePaths` instance with a canonicalized `base_dir`.
        """
        normalized_base_dir = base_dir.expanduser().resolve()
        return cls(base_dir=normalized_base_dir)

    def tool_root(
        self,
        tool: str,
    ) -> Path:
        """Return the root directory for a tool within the workspace.

        Args:
            tool: Tool name used as the direct child directory under `base_dir`.

        Returns:
            The tool root path.
        """
        return self.base_dir / tool

    def run_root(
        self,
        *,
        tool: str,
        run_id: str,
        runs_dirname: str = "runs",
    ) -> Path:
        """Return the root directory for a specific tool run.

        Args:
            tool: Tool name under the workspace.
            run_id: Unique run identifier.
            runs_dirname: Name of the intermediate runs directory.

        Returns:
            The run root path.
        """
        return self.tool_root(tool) / runs_dirname / run_id
