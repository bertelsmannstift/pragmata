"""Version resolution utilities for chatboteval."""

from importlib import metadata


def get_version(dist_name: str = "chatboteval") -> str:
    """Return installed distribution version."""
    return metadata.version(dist_name)