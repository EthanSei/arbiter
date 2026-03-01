"""Token-bucket rate limiter wrapping httpx.AsyncClient."""

from __future__ import annotations

import httpx


class RateLimitedClient:
    """Async HTTP client with token-bucket rate limiting.

    Wraps httpx.AsyncClient to enforce a maximum requests-per-minute (RPM) limit.
    Respects Retry-After headers on 429 responses.
    """

    def __init__(self, client: httpx.AsyncClient, rpm: int = 60) -> None:
        raise NotImplementedError  # Phase 2

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        raise NotImplementedError  # Phase 2

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        raise NotImplementedError  # Phase 2

    async def close(self) -> None:
        raise NotImplementedError  # Phase 2
