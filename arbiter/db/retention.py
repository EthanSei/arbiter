"""Data retention: delete old order book snapshots and candlestick bars."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.db.models import CandlestickBar, OrderBookSnapshot


async def cleanup_old_data(
    session: AsyncSession,
    *,
    ob_max_days: int = 7,
    candle_max_days: int = 90,
) -> dict[str, int]:
    """Delete data older than the specified retention windows.

    Returns a dict with counts of deleted rows per table.
    """
    now = datetime.now(UTC)

    ob_cutoff = now - timedelta(days=ob_max_days)
    ob_result = await session.execute(
        delete(OrderBookSnapshot).where(OrderBookSnapshot.snapshot_at < ob_cutoff)
    )

    candle_cutoff = now - timedelta(days=candle_max_days)
    candle_result = await session.execute(
        delete(CandlestickBar).where(CandlestickBar.period_start < candle_cutoff)
    )

    await session.flush()

    return {
        "order_book_snapshots": ob_result.rowcount,
        "candlestick_bars": candle_result.rowcount,
    }
