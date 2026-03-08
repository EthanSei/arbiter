"""Application entrypoint — wires all components and runs the scan loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from pathlib import Path

import httpx

from arbiter.alerts.base import AlertChannel
from arbiter.alerts.discord import DiscordChannel
from arbiter.alerts.stdout import StdoutChannel
from arbiter.config import settings
from arbiter.data.providers.base import FeatureProvider
from arbiter.data.providers.bls import BLSComponentProvider
from arbiter.data.providers.fred import FREDSurpriseProvider
from arbiter.db.session import async_session_factory, engine, init_db
from arbiter.health import start_health_server
from arbiter.ingestion.base import MarketClient
from arbiter.ingestion.collector import DataCollector
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.polymarket import PolymarketClient
from arbiter.ingestion.rate_limiter import RateLimitedClient
from arbiter.models.lgbm import LGBMEstimator
from arbiter.scheduler import ScanPipeline
from arbiter.scoring.fees import FeeFn, flat_fee_rate, kalshi_fee
from arbiter.scoring.strategy import build_default_strategies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main(stdout_alerts: bool = False) -> None:
    """Initialise all components and run the scan pipeline."""
    logger.info("Arbiter starting")

    await init_db()

    # Each client borrows its own HTTP connection; closed in finally block.
    kalshi_http = httpx.AsyncClient(timeout=30.0)

    # Kalshi's public API rate-limits rapid pagination; 20 RPM keeps requests ~3 s apart.
    kalshi_rl = RateLimitedClient(kalshi_http, rpm=20)
    series_tickers = (
        [s.strip() for s in settings.kalshi_target_series.split(",") if s.strip()]
        if settings.kalshi_target_series
        else None
    )
    kalshi = KalshiClient(
        kalshi_rl,
        base_url=settings.kalshi_api_base,
        max_markets=settings.max_markets_per_poll,
        min_volume_24h=settings.min_volume_24h,
        max_empty_pages=settings.kalshi_max_empty_pages,
        series_tickers=series_tickers,
    )

    # Second rate-limited client for background data collection (order book snapshots).
    # Separate RPM budget so data collection doesn't starve market fetches.
    data_http = httpx.AsyncClient(timeout=30.0)
    data_rl = RateLimitedClient(data_http, rpm=settings.data_collection_rpm)
    data_kalshi = KalshiClient(data_rl, base_url=settings.kalshi_api_base)
    data_collector = DataCollector(
        data_kalshi, async_session_factory, top_n=settings.orderbook_top_n
    )

    clients: list[MarketClient] = [kalshi]
    poly_http: httpx.AsyncClient | None = None
    if settings.polymarket_enabled:
        poly_http = httpx.AsyncClient(timeout=30.0)
        polymarket = PolymarketClient(
            poly_http,
            gamma_base_url=settings.polymarket_gamma_base,
            min_volume_24h=settings.min_volume_24h,
        )
        clients.append(polymarket)

    # LGBMEstimator falls back to market midpoint when model file is absent.
    model_path: str | None = settings.model_weights_path
    if model_path and not Path(model_path).exists():
        logger.warning("Model weights not found at %s — using midpoint fallback", model_path)
        model_path = None
    estimator = LGBMEstimator(model_path)

    channels: list[AlertChannel] = (
        [StdoutChannel()] if stdout_alerts else [DiscordChannel(settings.discord_webhook_url)]
    )

    # Anchor providers: FREDSurpriseProvider + BLSComponentProvider load from
    # data/features/{fred,bls}/ JSON caches populated by `make fetch-data`.
    anchor_providers: list[FeatureProvider] = [
        FREDSurpriseProvider(),
        BLSComponentProvider(),
    ]

    # Fee model: parabolic (Kalshi real fees) or flat (legacy)
    fee_fn: FeeFn | None = None
    if settings.fee_model == "kalshi":
        fee_fn = kalshi_fee
        logger.info("Using Kalshi parabolic fee model")
    elif settings.fee_model == "flat":
        fee_fn = flat_fee_rate(settings.fee_rate)
        logger.info("Using flat fee model (rate=%.4f)", settings.fee_rate)
    else:
        logger.warning("Unknown fee_model=%r — fees disabled", settings.fee_model)

    # IndicatorRouter routes contracts matching the INDICATORS registry
    # (KXCPI, KXPAYROLLS, etc.) to [Consistency + Anchor] and others to [EV].
    # Skip YesOnlyEVStrategy when the model is not calibrated — raw uncalibrated
    # probabilities produce too many false signals.
    calibrators_path = Path("models/anchor_calibrators.pkl")
    strategies = build_default_strategies(
        fee_rate=settings.fee_rate,
        anchor_providers=anchor_providers,
        include_ev=estimator.calibrated,
        calibrators_path=str(calibrators_path) if calibrators_path.exists() else None,
        fee_fn=fee_fn,
    )

    pipeline = ScanPipeline(
        clients=clients,
        estimator=estimator,
        channels=channels,
        session_factory=async_session_factory,
        ev_threshold=settings.ev_threshold,
        fee_rate=settings.fee_rate,
        kelly_fraction=settings.kelly_fraction,
        strategies=strategies,
        fee_fn=fee_fn,
        data_collector=data_collector,
    )

    health_task = asyncio.create_task(start_health_server(settings.health_port))

    # Register SIGTERM handler so Cloud Run graceful shutdown cancels the loop.
    loop = asyncio.get_running_loop()
    pipeline_task = asyncio.create_task(pipeline.run_forever(settings.poll_interval_seconds))
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, pipeline_task.cancel)

    try:
        await pipeline_task
    except asyncio.CancelledError:
        logger.info("Scan loop cancelled — shutting down")
    finally:
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task

        await kalshi_http.aclose()
        await data_http.aclose()
        if poly_http is not None:
            await poly_http.aclose()
        for channel in channels:
            await channel.close()
        await engine.dispose()

        logger.info("Arbiter shutdown complete")
