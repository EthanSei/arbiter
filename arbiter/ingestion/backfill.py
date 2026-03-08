"""Offline candlestick backfill using the batch candlesticks API."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arbiter.db.models import CandlestickBar, Source
from arbiter.ingestion.kalshi import KalshiClient

logger = logging.getLogger(__name__)


async def backfill_candlesticks(
    kalshi_client: KalshiClient,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    tickers: list[str],
    series_ticker: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    period_interval: int = 60,
) -> int:
    """Fetch historical candlesticks and persist as CandlestickBar rows.

    Skips rows that already exist (dedup on contract_id + period_start + period_interval).
    Returns the total number of new bars written.
    """
    try:
        batch = await kalshi_client.fetch_candlesticks_batch(
            tickers,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=period_interval,
        )
    except Exception:
        logger.warning("Failed to fetch candlesticks for %s", series_ticker, exc_info=True)
        return 0

    interval_seconds = period_interval * 60
    total = 0

    async with session_factory() as session:
        for ticker, candles in batch.items():
            for candle in candles:
                try:
                    price = candle.get("price", {})
                    end_ts_val = candle["end_period_ts"]
                except (KeyError, TypeError):
                    logger.warning("Skipping malformed candle for %s: %s", ticker, candle)
                    continue
                period_start = datetime.fromtimestamp(end_ts_val - interval_seconds, tz=UTC)

                # Check for existing row to avoid unique constraint violation
                exists = await session.execute(
                    select(CandlestickBar.id).where(
                        CandlestickBar.contract_id == ticker,
                        CandlestickBar.period_start == period_start,
                        CandlestickBar.period_interval == period_interval,
                    )
                )
                if exists.scalar_one_or_none() is not None:
                    continue

                bar = CandlestickBar(
                    source=Source.KALSHI,
                    contract_id=ticker,
                    series_ticker=series_ticker,
                    period_start=period_start,
                    period_interval=period_interval,
                    open=(price.get("open") or 0) / 100.0,
                    high=(price.get("high") or 0) / 100.0,
                    low=(price.get("low") or 0) / 100.0,
                    close=(price.get("close") or 0) / 100.0,
                    volume=candle.get("volume"),
                )
                session.add(bar)
                total += 1
        await session.commit()

    return total
