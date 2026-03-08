"""Tests for maker data infrastructure DB models: OrderBookSnapshot and CandlestickBar."""

from datetime import UTC, datetime

from sqlalchemy import select

from arbiter.db.models import CandlestickBar, OrderBookSnapshot, Source


class TestOrderBookSnapshot:
    async def test_create_and_read(self, db_session):
        snapshot = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            series_ticker="KXCPI",
            event_ticker="KXCPI-26MAR",
            bids=[{"price": 0.55, "quantity": 100}, {"price": 0.54, "quantity": 200}],
            asks=[{"price": 0.57, "quantity": 150}],
        )
        db_session.add(snapshot)
        await db_session.commit()

        result = await db_session.execute(
            select(OrderBookSnapshot).where(OrderBookSnapshot.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.source == Source.KALSHI
        assert row.series_ticker == "KXCPI"
        assert row.event_ticker == "KXCPI-26MAR"
        assert len(row.bids) == 2
        assert row.bids[0]["price"] == 0.55
        assert len(row.asks) == 1

    async def test_uuid_auto_generated(self, db_session):
        snapshot = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            bids=[],
            asks=[],
        )
        db_session.add(snapshot)
        await db_session.commit()

        result = await db_session.execute(
            select(OrderBookSnapshot).where(OrderBookSnapshot.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.id is not None
        assert len(row.id) == 36

    async def test_snapshot_at_auto_set(self, db_session):
        snapshot = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            bids=[],
            asks=[],
        )
        db_session.add(snapshot)
        await db_session.commit()

        result = await db_session.execute(
            select(OrderBookSnapshot).where(OrderBookSnapshot.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.snapshot_at is not None

    async def test_empty_book(self, db_session):
        """Empty order books (illiquid markets) are valid."""
        snapshot = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id="KXMEDIA-26MAR-T1",
            bids=[],
            asks=[],
        )
        db_session.add(snapshot)
        await db_session.commit()

        result = await db_session.execute(
            select(OrderBookSnapshot).where(OrderBookSnapshot.contract_id == "KXMEDIA-26MAR-T1")
        )
        row = result.scalar_one()
        assert row.bids == []
        assert row.asks == []

    async def test_series_ticker_defaults_empty(self, db_session):
        snapshot = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            bids=[],
            asks=[],
        )
        db_session.add(snapshot)
        await db_session.commit()

        result = await db_session.execute(
            select(OrderBookSnapshot).where(OrderBookSnapshot.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.series_ticker == ""
        assert row.event_ticker == ""


class TestCandlestickBar:
    async def test_create_and_read(self, db_session):
        bar = CandlestickBar(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            series_ticker="KXCPI",
            period_start=datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC),
            period_interval=60,
            open=0.55,
            high=0.60,
            low=0.52,
            close=0.58,
            volume=1200,
        )
        db_session.add(bar)
        await db_session.commit()

        result = await db_session.execute(
            select(CandlestickBar).where(CandlestickBar.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.source == Source.KALSHI
        assert row.series_ticker == "KXCPI"
        assert row.open == 0.55
        assert row.high == 0.60
        assert row.low == 0.52
        assert row.close == 0.58
        assert row.volume == 1200
        assert row.period_interval == 60

    async def test_uuid_auto_generated(self, db_session):
        bar = CandlestickBar(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            period_start=datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC),
            period_interval=60,
            open=0.55,
            high=0.60,
            low=0.52,
            close=0.58,
            volume=1200,
        )
        db_session.add(bar)
        await db_session.commit()

        result = await db_session.execute(
            select(CandlestickBar).where(CandlestickBar.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.id is not None
        assert len(row.id) == 36

    async def test_nullable_volume(self, db_session):
        """Volume can be None for periods with no trades."""
        bar = CandlestickBar(
            source=Source.KALSHI,
            contract_id="KXCPI-26MAR-T3.0",
            period_start=datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC),
            period_interval=60,
            open=0.55,
            high=0.55,
            low=0.55,
            close=0.55,
            volume=None,
        )
        db_session.add(bar)
        await db_session.commit()

        result = await db_session.execute(
            select(CandlestickBar).where(CandlestickBar.contract_id == "KXCPI-26MAR-T3.0")
        )
        row = result.scalar_one()
        assert row.volume is None

    async def test_multiple_bars_per_contract(self, db_session):
        """Multiple candle bars for the same contract at different periods."""
        for hour in range(3):
            bar = CandlestickBar(
                source=Source.KALSHI,
                contract_id="KXCPI-26MAR-T3.0",
                period_start=datetime(2026, 3, 7, hour, 0, 0, tzinfo=UTC),
                period_interval=60,
                open=0.50 + hour * 0.01,
                high=0.55 + hour * 0.01,
                low=0.48 + hour * 0.01,
                close=0.52 + hour * 0.01,
                volume=100 * (hour + 1),
            )
            db_session.add(bar)
        await db_session.commit()

        result = await db_session.execute(
            select(CandlestickBar).where(CandlestickBar.contract_id == "KXCPI-26MAR-T3.0")
        )
        rows = result.scalars().all()
        assert len(rows) == 3
