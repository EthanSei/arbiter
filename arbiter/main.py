"""Application entrypoint — wires all components and runs the scan loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

import httpx

from arbiter.alerts.base import AlertChannel
from arbiter.alerts.discord import DiscordChannel
from arbiter.config import settings
from arbiter.db.session import async_session_factory, init_db
from arbiter.health import start_health_server
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.polymarket import PolymarketClient
from arbiter.models.lgbm import LGBMEstimator
from arbiter.scheduler import ScanPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Initialise all components and run the scan pipeline."""
    logger.info("Arbiter starting")

    await init_db()

    # Each client borrows its own HTTP connection; closed in finally block.
    kalshi_http = httpx.AsyncClient(timeout=30.0)
    poly_http = httpx.AsyncClient(timeout=30.0)

    kalshi = KalshiClient(kalshi_http, base_url=settings.kalshi_api_base)
    polymarket = PolymarketClient(poly_http, gamma_base_url=settings.polymarket_gamma_base)

    # LGBMEstimator falls back to market midpoint when model file is absent.
    model_path: str | None = settings.model_weights_path
    if model_path and not Path(model_path).exists():
        logger.warning("Model weights not found at %s — using midpoint fallback", model_path)
        model_path = None
    estimator = LGBMEstimator(model_path)

    channels: list[AlertChannel] = [DiscordChannel(settings.discord_webhook_url)]

    pipeline = ScanPipeline(
        clients=[kalshi, polymarket],
        estimator=estimator,
        channels=channels,
        session_factory=async_session_factory,
        ev_threshold=settings.ev_threshold,
        fee_rate=settings.fee_rate,
    )

    health_task = asyncio.create_task(start_health_server(settings.health_port))

    try:
        await pipeline.run_forever(settings.poll_interval_seconds)
    except asyncio.CancelledError:
        logger.info("Scan loop cancelled — shutting down")
    finally:
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task

        await kalshi_http.aclose()
        await poly_http.aclose()
        for channel in channels:
            await channel.close()

        logger.info("Arbiter shutdown complete")
