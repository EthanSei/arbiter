"""Tests for data retention cleanup."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from arbiter.db.models import CandlestickBar, OrderBookSnapshot, Source
from arbiter.db.retention import cleanup_old_data


async def test_deletes_old_order_book_snapshots(db_session):
    """Should delete OrderBookSnapshot rows older than max_age_days."""
    old = OrderBookSnapshot(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.0",
        bids=[],
        asks=[],
        snapshot_at=datetime.now(UTC) - timedelta(days=10),
    )
    recent = OrderBookSnapshot(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.5",
        bids=[],
        asks=[],
        snapshot_at=datetime.now(UTC) - timedelta(days=1),
    )
    db_session.add_all([old, recent])
    await db_session.commit()

    deleted = await cleanup_old_data(db_session, ob_max_days=7, candle_max_days=90)

    result = await db_session.execute(select(OrderBookSnapshot))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].contract_id == "KXCPI-T3.5"
    assert deleted["order_book_snapshots"] == 1


async def test_deletes_old_candlestick_bars(db_session):
    """Should delete CandlestickBar rows older than candle_max_days."""
    old = CandlestickBar(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.0",
        series_ticker="KXCPI",
        period_start=datetime.now(UTC) - timedelta(days=100),
        period_interval=60,
        open=0.50,
        high=0.55,
        low=0.48,
        close=0.52,
        volume=100,
    )
    recent = CandlestickBar(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.5",
        series_ticker="KXCPI",
        period_start=datetime.now(UTC) - timedelta(days=30),
        period_interval=60,
        open=0.50,
        high=0.55,
        low=0.48,
        close=0.52,
        volume=100,
    )
    db_session.add_all([old, recent])
    await db_session.commit()

    deleted = await cleanup_old_data(db_session, ob_max_days=7, candle_max_days=90)

    result = await db_session.execute(select(CandlestickBar))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].contract_id == "KXCPI-T3.5"
    assert deleted["candlestick_bars"] == 1


async def test_no_data_returns_zero_counts(db_session):
    """Should return zero counts when nothing to delete."""
    deleted = await cleanup_old_data(db_session, ob_max_days=7, candle_max_days=90)
    assert deleted["order_book_snapshots"] == 0
    assert deleted["candlestick_bars"] == 0
