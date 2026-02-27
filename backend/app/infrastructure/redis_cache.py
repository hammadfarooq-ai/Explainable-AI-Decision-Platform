from __future__ import annotations

import json
from typing import Any, Optional

import redis

from backend.app.core.config import get_settings


class RedisCache:
    """Thin Redis wrapper for JSON-based caching and lightweight job queues.

    This implementation is intentionally minimal and synchronous. For CPU-bound
    work, background tasks should still be used to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = redis.Redis.from_url(settings.redis_url)
        self._default_ttl = settings.cache_ttl_seconds

    @property
    def client(self) -> redis.Redis:
        return self._client

    # Generic JSON cache -----------------------------------------------------

    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        payload = json.dumps(value)
        expire_seconds = ttl or self._default_ttl
        self._client.setex(key, expire_seconds, payload)

    # Simple background job queue --------------------------------------------

    def enqueue_job(self, queue: str, payload: dict[str, Any]) -> None:
        """Push a JSON-serializable job payload onto a Redis list."""

        self._client.lpush(queue, json.dumps(payload))

    def dequeue_job(self, queue: str, timeout: int = 5) -> Optional[dict[str, Any]]:
        """Block for up to ``timeout`` seconds waiting for a job."""

        result = self._client.brpop(queue, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


redis_cache = RedisCache()

