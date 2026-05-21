"""Internal API layer."""

from pragmata.core.settings.settings_base import UNSET, Unset

from .version import get_version

__all__ = ["get_version", "UNSET", "Unset"]
