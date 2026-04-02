"""Top-level public interface for pragmata.

Only curated, stable symbols should be exposed here.
"""

from . import querygen
from .api import get_version

__all__ = ["get_version", "querygen"]
