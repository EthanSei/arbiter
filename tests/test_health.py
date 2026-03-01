"""Tests for the health check HTTP server."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from arbiter.health import HealthState, _handle, _state, update_health

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_health_state() -> None:
    """Reset the module-level health singleton before every test."""
    _state.__dict__.update(HealthState().__dict__)


# ---------------------------------------------------------------------------
# update_health
# ---------------------------------------------------------------------------


def test_update_health_sets_status() -> None:
    update_health(status="degraded")
    assert _state.status == "degraded"


def test_update_health_sets_markets_scanned() -> None:
    update_health(markets_scanned=42)
    assert _state.markets_scanned == 42


def test_update_health_sets_opportunities_found() -> None:
    update_health(opportunities_found=3)
    assert _state.opportunities_found == 3


def test_update_health_sets_duration() -> None:
    update_health(last_cycle_duration_seconds=12.5)
    assert _state.last_cycle_duration_seconds == 12.5


def test_update_health_sets_errors() -> None:
    update_health(errors=["oops"])
    assert _state.errors == ["oops"]


def test_update_health_sets_last_cycle_timestamp() -> None:
    from datetime import UTC, datetime

    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    update_health(last_cycle_completed_at=ts)
    assert _state.last_cycle_completed_at == ts.isoformat()


# ---------------------------------------------------------------------------
# HTTP handler — unit tests using fake reader/writer
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self._buf = b""

    def write(self, data: bytes) -> None:
        self._buf += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    @property
    def response(self) -> str:
        return self._buf.decode()


async def _make_reader(request: str) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(request.encode())
    reader.feed_eof()
    return reader


async def test_handler_get_health_returns_200() -> None:
    reader = await _make_reader("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    writer = _FakeWriter()
    await _handle(reader, writer)  # type: ignore[arg-type]
    assert "200 OK" in writer.response


async def test_handler_get_health_returns_json() -> None:
    reader = await _make_reader("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    writer = _FakeWriter()
    await _handle(reader, writer)  # type: ignore[arg-type]
    # Extract body (after \r\n\r\n)
    _, _, body = writer.response.partition("\r\n\r\n")
    data = json.loads(body)
    assert "status" in data
    assert "markets_scanned" in data
    assert "opportunities_found" in data
    assert "errors" in data


async def test_handler_get_health_reflects_state() -> None:
    update_health(status="degraded", markets_scanned=7, opportunities_found=2)
    reader = await _make_reader("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    writer = _FakeWriter()
    await _handle(reader, writer)  # type: ignore[arg-type]
    _, _, body = writer.response.partition("\r\n\r\n")
    data = json.loads(body)
    assert data["status"] == "degraded"
    assert data["markets_scanned"] == 7
    assert data["opportunities_found"] == 2


async def test_handler_unknown_path_returns_404() -> None:
    reader = await _make_reader("GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
    writer = _FakeWriter()
    await _handle(reader, writer)  # type: ignore[arg-type]
    assert "404" in writer.response


# ---------------------------------------------------------------------------
# Integration — real server on OS-assigned port
# ---------------------------------------------------------------------------


async def test_health_server_integration() -> None:
    """Start a real server on port 0 and hit /health via httpx."""
    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
