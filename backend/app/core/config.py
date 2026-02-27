from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This centralizes configuration for the entire platform and should be the
    single source of truth for environment-specific values.
    """

    # Environment
    env: Literal["local", "dev", "staging", "prod"] = Field(
        default="local", description="Deployment environment identifier."
    )
    debug: bool = Field(default=False, description="Enable debug mode.")

    # API
    api_prefix: str = Field(default="/api", description="Base API prefix.")
    project_name: str = Field(
        default="Enterprise AI Decision Platform",
        description="Human-readable project name.",
    )
    backend_host: str = Field(default="0.0.0.0", description="Backend bind host.")
    backend_port: int = Field(default=8000, description="Backend bind port.")

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+psycopg2://user:password@db:5432/enterprise_ai",
        description="SQLAlchemy-compatible PostgreSQL DSN.",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000", description="MLflow tracking URI."
    )
    mlflow_artifacts_uri: Optional[str] = Field(
        default=None,
        description="Optional MLflow artifacts URI; falls back to tracking URI.",
    )

    # Paths
    data_dir: str = Field(default="data", description="Directory for data files.")
    models_dir: str = Field(
        default="ml_models", description="Directory for local model artifacts."
    )
    faiss_index_dir: str = Field(
        default="faiss_indexes", description="Directory for FAISS indexes."
    )

    # Redis / caching
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL used for caching and background jobs.",
    )
    cache_ttl_seconds: int = Field(
        default=600,
        description="Default TTL (in seconds) for cached items such as EDA summaries.",
    )

    # RAG / LLM
    llm_provider: Literal["openai", "huggingface", "azure_openai"] = Field(
        default="openai", description="LLM provider identifier."
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini", description="Default LLM model name."
    )
    # API keys are intentionally generic; concrete env names are documented in README.
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the configured LLM provider.",
    )

    # Security / CORS
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins for the frontend.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return application settings instance (singleton via LRU cache)."""

    return Settings()  # type: ignore[call-arg]

