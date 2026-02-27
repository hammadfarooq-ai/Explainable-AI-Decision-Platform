from __future__ import annotations

import logging
import uuid
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import ml as ml_router
from backend.app.api import rag as rag_router
from backend.app.api import system as system_router
from backend.app.core.config import get_settings
from backend.app.core.logging_config import configure_logging
from backend.app.infrastructure.db.base import Base, engine


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    settings = get_settings()
    configure_logging(level="DEBUG" if settings.debug else "INFO")

    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description=(
            "Enterprise AI Decision Platform: automated ML model training, "
            "explainability, data drift detection, and RAG-based recommendations."
        ),
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Callable):  # type: ignore[type-arg]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        logger = logging.getLogger("enterprise_ai.request")
        logger.info("Request start", extra={"request_id": request_id})

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        logger.info("Request end", extra={"request_id": request_id})
        return response

    # Routers
    app.include_router(system_router.router, prefix=settings.api_prefix)
    app.include_router(ml_router.router, prefix=settings.api_prefix)
    app.include_router(rag_router.router, prefix=settings.api_prefix)

    # Create DB schema (for demo; in production use migrations)
    Base.metadata.create_all(bind=engine)

    return app


app = create_app()

