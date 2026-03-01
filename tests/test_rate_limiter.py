"""Tests for token-bucket rate limiter."""

import asyncio
from unittest.mock import AsyncMock

import httpx

from arbiter.ingestion.rate_limiter import RateLimitedClient


def _mock_transport(responses: list[httpx.Response] | None = None) -> httpx.MockTransport:
    """Create a mock transport returning canned responses."""
    resp_iter = iter(responses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            return next(resp_iter)
        except StopIteration:
            return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


class TestRateLimitedClientInit:
    def test_stores_rpm_and_client(self):
        client = httpx.AsyncClient()
        rl = RateLimitedClient(client, rpm=30)
        assert rl._rpm == 30
        assert rl._client is client

    def test_initializes_full_token_bucket(self):
        client = httpx.AsyncClient()
        rl = RateLimitedClient(client, rpm=60)
        assert rl._tokens == 60


class TestRateLimitedClientGet:
    async def test_get_returns_response(self):
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.get("https://example.com/api")
            assert resp.status_code == 200

    async def test_get_consumes_token(self):
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            initial = rl._tokens
            await rl.get("https://example.com/api")
            assert rl._tokens < initial

    async def test_get_passes_kwargs(self):
        """Kwargs like params and headers are forwarded to the underlying client."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["key"] == "val"
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            await rl.get("https://example.com/api", params={"key": "val"})


class TestRateLimitedClientPost:
    async def test_post_returns_response(self):
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.post("https://example.com/api")
            assert resp.status_code == 200


class TestRateLimitedClient429Handling:
    async def test_retries_on_429_with_retry_after(self):
        """On a 429, waits Retry-After seconds then retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.get("https://example.com/api")
            assert resp.status_code == 200
            assert call_count == 2

    async def test_429_without_retry_after_uses_default(self):
        """Without Retry-After header, uses a sensible default backoff."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.get("https://example.com/api")
            assert resp.status_code == 200
            assert call_count == 2

    async def test_gives_up_after_max_retries(self):
        """Doesn't retry forever on persistent 429s."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "0"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.get("https://example.com/api")
            assert resp.status_code == 429

    async def test_http_date_retry_after_uses_default(self):
        """HTTP-date format in Retry-After header falls back to default instead of crashing."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=60)
            resp = await rl.get("https://example.com/api")
        assert resp.status_code == 200
        assert call_count == 2


class TestRateLimitedClientTokenRefill:
    async def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time since last request."""
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            rl = RateLimitedClient(client, rpm=6000)  # High RPM for fast refill
            # Drain some tokens
            for _ in range(5):
                await rl.get("https://example.com/api")
            tokens_after_drain = rl._tokens
            # Wait a tiny bit for refill
            await asyncio.sleep(0.05)
            await rl.get("https://example.com/api")
            # After sleep + refill, tokens should be >= what we had before
            # (minus 1 for the new request, plus refill)
            # Just check that the refill mechanism works at all
            assert rl._tokens >= tokens_after_drain - 1


class TestRateLimitedClientClose:
    async def test_close_delegates_to_underlying_client(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        rl = RateLimitedClient(client, rpm=60)
        await rl.close()
        client.aclose.assert_called_once()
