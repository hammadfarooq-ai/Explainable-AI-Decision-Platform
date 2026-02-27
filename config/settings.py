from __future__ import annotations

"""Shared configuration helpers for external scripts or CLIs.

The primary application configuration lives in ``backend.app.core.config``.
This module simply re-exports those settings for convenience.
"""

from backend.app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

