"""Annotation IAA API — compute inter-annotator agreement from export CSVs."""

import logging
from pathlib import Path

from pragmata.api._error_log import error_log
from pragmata.core.annotation.iaa_runner import run_iaa
from pragmata.core.paths.annotation_paths import resolve_export_paths, resolve_iaa_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.iaa_report import IaaReport
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file

logger = logging.getLogger(__name__)


def compute_iaa(
    export_id: str,
    *,
    base_dir: str | Path | Unset = UNSET,
    tasks: list[Task] | None = None,
    workspace_prefix: str | Unset = UNSET,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
    config_path: str | Path | Unset = UNSET,
) -> IaaReport:
    """Compute inter-annotator agreement metrics from an existing export.

    Reads the export CSVs for the given ``export_id``, computes per-label
    Krippendorff's alpha with bootstrap CIs and pairwise Cohen's kappa,
    and writes a JSON report.

    Args:
        export_id: Identifier of a previous export run whose CSVs to analyse.
        base_dir: Workspace base directory. Defaults to cwd.
        tasks: Tasks to analyse. Defaults to all three tasks.
        workspace_prefix: Prefix used when the environment was created.
        n_resamples: Number of bootstrap iterations for CIs.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Optional RNG seed for reproducible bootstrap.
        config_path: Path to YAML config file for settings resolution.

    Returns:
        IaaReport with per-label alpha, CIs, and pairwise kappas.
    """
    settings = AnnotationSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        overrides={"workspace_prefix": workspace_prefix, "base_dir": base_dir},
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    export_paths = resolve_export_paths(workspace=workspace, export_id=export_id)
    iaa_paths = resolve_iaa_paths(export_paths=export_paths).ensure_dirs()
    resolved_tasks = tasks if tasks is not None else list(Task)

    with error_log(export_paths.tool_root):
        report = run_iaa(export_paths, iaa_paths, resolved_tasks, n_resamples=n_resamples, ci=ci, seed=seed)

    logger.info(
        "IAA complete: %d task(s) analysed, report at %s",
        len(report.tasks),
        iaa_paths.report,
    )
    return report
