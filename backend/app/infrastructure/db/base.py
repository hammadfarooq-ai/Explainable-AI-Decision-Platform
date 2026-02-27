from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


def _create_engine():
    settings = get_settings()
    return create_engine(settings.database_url, echo=settings.debug, future=True)


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency providing a database session."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

