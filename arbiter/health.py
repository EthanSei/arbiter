"""Health check endpoint for Cloud Run liveness probes.

Exposes a minimal HTTP server on the configured port (default 8080)
that returns cycle metadata for monitoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HealthState:
    status: str = "healthy"
    last_cycle_completed_at: str | None = None
    last_cycle_duration_seconds: float | None = None
    markets_scanned: int = 0
    opportunities_found: int = 0
    errors: list[str] = field(default_factory=list)


_state = HealthState()


def update_health(
    *,
    status: str = "healthy",
    last_cycle_completed_at: datetime | None = None,
    last_cycle_duration_seconds: float | None = None,
    markets_scanned: int = 0,
    opportunities_found: int = 0,
    errors: list[str] | None = None,
) -> None:
    """Update global health state after a cycle completes."""
    _state.status = status
    _state.last_cycle_completed_at = (
        last_cycle_completed_at.isoformat() if last_cycle_completed_at else None
    )
    _state.last_cycle_duration_seconds = last_cycle_duration_seconds
    _state.markets_scanned = markets_scanned
    _state.opportunities_found = opportunities_found
    _state.errors = errors if errors is not None else []


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle one HTTP connection and return a JSON response."""
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        # Drain headers
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if line in (b"\r\n", b"\n", b""):
                break

        parts = request_line.split(b" ")
        path = parts[1].decode("utf-8", errors="replace") if len(parts) > 1 else "/"

        if path == "/health" or path.startswith("/health?"):
            body = json.dumps(
                {
                    "status": _state.status,
                    "last_cycle_completed_at": _state.last_cycle_completed_at,
                    "last_cycle_duration_seconds": _state.last_cycle_duration_seconds,
                    "markets_scanned": _state.markets_scanned,
                    "opportunities_found": _state.opportunities_found,
                    "errors": _state.errors,
                }
            ).encode()
            status_line = b"200 OK"
        else:
            body = b'{"error": "not found"}'
            status_line = b"404 Not Found"

        writer.write(
            b"HTTP/1.1 " + status_line + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: close\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n" + body
        )
        await writer.drain()
    except (TimeoutError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as exc:
        logger.warning("Health handler error: %s", exc)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def start_health_server(port: int = 8080) -> None:
    """Start the health check HTTP server.

    GET /health returns JSON with:
    - status: "healthy" or "degraded"
    - last_cycle_completed_at: ISO timestamp
    - last_cycle_duration_seconds: float
    - markets_scanned: int
    - opportunities_found: int
    - errors: list[str]
    """
    server = await asyncio.start_server(_handle, "0.0.0.0", port)
    logger.info("Health server listening on :%d", port)
    async with server:
        await server.serve_forever()
