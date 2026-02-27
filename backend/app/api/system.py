from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """Basic health endpoint for readiness/liveness checks."""

    return {"status": "ok"}


@router.get("/config")
def public_config() -> Dict[str, Any]:
    """Return non-sensitive configuration for debugging."""

    settings = get_settings()
    return {
        "project_name": settings.project_name,
        "env": settings.env,
        "debug": settings.debug,
        "api_prefix": settings.api_prefix,
    }

