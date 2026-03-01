"""Token-bucket rate limiter wrapping httpx.AsyncClient."""

from __future__ import annotations

import asyncio
import time

import httpx

_MAX_RETRIES = 3
_DEFAULT_RETRY_AFTER = 1.0


class RateLimitedClient:
    """Async HTTP client with token-bucket rate limiting.

    Wraps httpx.AsyncClient to enforce a maximum requests-per-minute (RPM) limit.
    Respects Retry-After headers on 429 responses.
    """

    def __init__(self, client: httpx.AsyncClient, rpm: int = 60) -> None:
        self._client = client
        self._rpm = rpm
        self._tokens = float(rpm)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def _acquire(self) -> None:
        """Wait until a token is available, refilling based on elapsed time."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # No token available — wait a fraction of the refill interval
            await asyncio.sleep(60.0 / self._rpm)

    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        await self._acquire()
        call = getattr(self._client, method)
        for _attempt in range(_MAX_RETRIES):
            resp: httpx.Response = await call(url, **kwargs)
            if resp.status_code != 429:
                return resp
            retry_after = float(resp.headers.get("Retry-After", _DEFAULT_RETRY_AFTER))
            await asyncio.sleep(retry_after)
            await self._acquire()
        # Final attempt — still rate-limited
        await self._acquire()
        result: httpx.Response = await call(url, **kwargs)
        return result

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        return await self._request("get", url, **kwargs)

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        return await self._request("post", url, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()
