"""Top-level public interface for pragmata.

Only curated, stable symbols should be exposed here.
"""

import logging

from . import querygen
from .api import get_version

__all__ = ["get_version", "querygen"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
