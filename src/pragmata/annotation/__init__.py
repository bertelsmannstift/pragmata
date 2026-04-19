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
    "DiscardReason",
    "ExportResult",
    "IaaReport",
    "ImportResult",
    "SetupResult",
    "Task",
    "UserSpec",
    "compute_iaa",
    "export_annotations",
    "import_records",
    "setup",
    "teardown",
]

_LAZY: dict[str, tuple[str, str]] = {
    "DiscardReason": ("pragmata.core.schemas.annotation_task", "DiscardReason"),
    "ExportResult": ("pragmata.core.annotation.export_runner", "ExportResult"),
    "IaaReport": ("pragmata.core.schemas.iaa_report", "IaaReport"),
    "ImportResult": ("pragmata.api.annotation_import", "ImportResult"),
    "SetupResult": ("pragmata.core.annotation.setup", "SetupResult"),
    "Task": ("pragmata.core.schemas.annotation_task", "Task"),
    "UserSpec": ("pragmata.core.settings.annotation_settings", "UserSpec"),
    "compute_iaa": ("pragmata.api.annotation_iaa", "compute_iaa"),
    "export_annotations": ("pragmata.api.annotation_export", "export_annotations"),
    "import_records": ("pragmata.api.annotation_import", "import_records"),
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
    from pragmata.core.annotation.export_runner import ExportResult as ExportResult
    from pragmata.core.annotation.setup import SetupResult as SetupResult
    from pragmata.core.schemas.annotation_task import DiscardReason as DiscardReason
    from pragmata.core.schemas.annotation_task import Task as Task
    from pragmata.core.schemas.iaa_report import IaaReport as IaaReport
    from pragmata.core.settings.annotation_settings import UserSpec as UserSpec
