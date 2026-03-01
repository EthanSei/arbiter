"""Health check endpoint for Cloud Run liveness probes.

Exposes a minimal HTTP server on the configured port (default 8080)
that returns cycle metadata for monitoring.
"""

from __future__ import annotations


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
    raise NotImplementedError  # Phase 6
