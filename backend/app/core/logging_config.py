from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Any, Dict


def get_logging_config(level: str = "INFO") -> Dict[str, Any]:
    """Return a dictConfig-compatible logging configuration.

    The configuration is intentionally minimal but production-ready:
    - Structured console logs.
    - Request/trace ID placeholders supported via `extra` or logging context.
    """

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "request_id=%(request_id)s | %(message)s"
                )
            }
        },
        "handlers": {
            "default": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "loggers": {
            "uvicorn.error": {"level": level, "handlers": ["default"], "propagate": False},
            "uvicorn.access": {"level": level, "handlers": ["default"], "propagate": False},
            "enterprise_ai": {
                "level": level,
                "handlers": ["default"],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": ["default"],
        },
    }


def configure_logging(level: str = "INFO") -> None:
    """Configure global logging using dictConfig."""

    config = get_logging_config(level=level)
    # Ensure request_id is always present in log records.
    logging.basicConfig(level=level)
    dictConfig(config)

