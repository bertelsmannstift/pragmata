"""Internal API layer."""

from pragmata.core.schemas.annotation_task import Locale
from pragmata.core.settings.settings_base import UNSET

from .version import get_version

__all__ = ["get_version", "Locale", "UNSET"]
