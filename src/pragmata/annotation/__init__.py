"""Public annotation namespace.

Acts as the import boundary between the public annotation surface and its
argilla-dependent internals (spanning ``pragmata.api.annotation_*`` and
``pragmata.core.annotation.*``). Re-exports are resolved lazily via
:pep:`562` ``__getattr__`` so importing ``pragmata.annotation`` does not
trigger ``import argilla`` until an attribute is actually accessed. This
keeps the ``annotation`` optional extra genuinely optional: users who
installed pragmata without it can still import the package and run
non-annotation commands.
"""

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "CompletenessReport",
    "CompletenessSummary",
    "ExportResult",
    "HeadlineTotals",
    "IaaReport",
    "ImportResult",
    "KBucketStat",
    "Locale",
    "PanelCompleteness",
    "PanelStatus",
    "SetupResult",
    "StatusReport",
    "Task",
    "UserSpec",
    "compute_iaa",
    "export_annotations",
    "import_records",
    "report_status",
    "setup",
    "teardown",
]

_LAZY: dict[str, tuple[str, str]] = {
    "CompletenessReport": ("pragmata.core.annotation.completeness", "CompletenessReport"),
    "CompletenessSummary": ("pragmata.core.schemas.annotation_export", "CompletenessSummary"),
    "ExportResult": ("pragmata.core.annotation.export_runner", "ExportResult"),
    "HeadlineTotals": ("pragmata.core.annotation.panel_status", "HeadlineTotals"),
    "IaaReport": ("pragmata.core.schemas.iaa_report", "IaaReport"),
    "ImportResult": ("pragmata.api.annotation_import", "ImportResult"),
    "KBucketStat": ("pragmata.core.schemas.annotation_export", "KBucketStat"),
    "Locale": ("pragmata.core.schemas.annotation_task", "Locale"),
    "PanelCompleteness": ("pragmata.core.annotation.completeness", "PanelCompleteness"),
    "PanelStatus": ("pragmata.core.annotation.panel_status", "PanelStatus"),
    "SetupResult": ("pragmata.core.annotation.setup", "SetupResult"),
    "StatusReport": ("pragmata.core.annotation.panel_status", "StatusReport"),
    "Task": ("pragmata.core.schemas.annotation_task", "Task"),
    "UserSpec": ("pragmata.core.settings.annotation_settings", "UserSpec"),
    "compute_iaa": ("pragmata.api.annotation_iaa", "compute_iaa"),
    "export_annotations": ("pragmata.api.annotation_export", "export_annotations"),
    "import_records": ("pragmata.api.annotation_import", "import_records"),
    "report_status": ("pragmata.api.annotation_status", "report_status"),
    "setup": ("pragmata.api.annotation_setup", "setup"),
    "teardown": ("pragmata.api.annotation_setup", "teardown"),
}


def __getattr__(name: str) -> object:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as err:
        raise AttributeError(f"module 'pragmata.annotation' has no attribute {name!r}") from err
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from pragmata.api.annotation_export import export_annotations as export_annotations
    from pragmata.api.annotation_iaa import compute_iaa as compute_iaa
    from pragmata.api.annotation_import import ImportResult as ImportResult
    from pragmata.api.annotation_import import import_records as import_records
    from pragmata.api.annotation_setup import setup as setup
    from pragmata.api.annotation_setup import teardown as teardown
    from pragmata.api.annotation_status import report_status as report_status
    from pragmata.core.annotation.completeness import CompletenessReport as CompletenessReport
    from pragmata.core.annotation.completeness import PanelCompleteness as PanelCompleteness
    from pragmata.core.annotation.export_runner import ExportResult as ExportResult
    from pragmata.core.annotation.panel_status import HeadlineTotals as HeadlineTotals
    from pragmata.core.annotation.panel_status import PanelStatus as PanelStatus
    from pragmata.core.annotation.panel_status import StatusReport as StatusReport
    from pragmata.core.annotation.setup import SetupResult as SetupResult
    from pragmata.core.schemas.annotation_export import CompletenessSummary as CompletenessSummary
    from pragmata.core.schemas.annotation_export import KBucketStat as KBucketStat
    from pragmata.core.schemas.annotation_task import Locale as Locale
    from pragmata.core.schemas.annotation_task import Task as Task
    from pragmata.core.schemas.iaa_report import IaaReport as IaaReport
    from pragmata.core.settings.annotation_settings import UserSpec as UserSpec
