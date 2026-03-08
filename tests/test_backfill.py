"""Tests for candlestick backfill logic."""

from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.db.models import Base, CandlestickBar, Source
from arbiter.ingestion.backfill import backfill_candlesticks
from arbiter.ingestion.kalshi import KalshiClient

BATCH_RESPONSE = {
    "markets": [
        {
            "market_ticker": "KXCPI-26MAR-T3.0",
            "candlesticks": [
                {
                    "end_period_ts": 1709300400,
                    "price": {"open": 55, "high": 60, "low": 52, "close": 58},
                    "volume": 1200,
                },
                {
                    "end_period_ts": 1709304000,
                    "price": {"open": 58, "high": 65, "low": 57, "close": 63},
                    "volume": 850,
                },
            ],
        },
        {
            "market_ticker": "KXCPI-26MAR-T3.5",
            "candlesticks": [
                {
                    "end_period_ts": 1709300400,
                    "price": {"open": 40, "high": 44, "low": 38, "close": 42},
                    "volume": 500,
                },
            ],
        },
    ],
}


@pytest.fixture
async def db_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestBackfillCandlesticks:
    async def test_persists_candlestick_bars(self, db_factory):
        """Should write CandlestickBar rows for each candle returned by the API."""
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=BATCH_RESPONSE))
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            count = await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0", "KXCPI-26MAR-T3.5"],
                series_ticker="KXCPI",
            )

        assert count == 3
        async with db_factory() as session:
            result = await session.execute(select(CandlestickBar))
            rows = result.scalars().all()
        assert len(rows) == 3
        tickers = {r.contract_id for r in rows}
        assert tickers == {"KXCPI-26MAR-T3.0", "KXCPI-26MAR-T3.5"}

    async def test_maps_ohlcv_fields_correctly(self, db_factory):
        """Should correctly map price OHLCV and volume."""
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=BATCH_RESPONSE))
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0"],
                series_ticker="KXCPI",
            )

        async with db_factory() as session:
            result = await session.execute(
                select(CandlestickBar).order_by(CandlestickBar.period_start)
            )
            rows = result.scalars().all()
        bar = rows[0]
        assert bar.open == pytest.approx(0.55)
        assert bar.high == pytest.approx(0.60)
        assert bar.low == pytest.approx(0.52)
        assert bar.close == pytest.approx(0.58)
        assert bar.volume == 1200
        assert bar.series_ticker == "KXCPI"
        assert bar.source == Source.KALSHI
        assert bar.period_interval == 60

    async def test_sets_period_start_from_end_ts_minus_interval(self, db_factory):
        """period_start = end_period_ts - period_interval_seconds."""
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=BATCH_RESPONSE))
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0"],
                series_ticker="KXCPI",
                period_interval=60,
            )

        async with db_factory() as session:
            result = await session.execute(
                select(CandlestickBar).order_by(CandlestickBar.period_start)
            )
            rows = result.scalars().all()
        # end_period_ts=1709300400, interval=60min=3600s → start=1709296800
        expected_start = datetime.fromtimestamp(1709300400 - 3600, tz=UTC)
        # SQLite strips tzinfo, so compare naive values
        assert rows[0].period_start.replace(tzinfo=None) == expected_start.replace(tzinfo=None)

    async def test_empty_response_returns_zero(self, db_factory):
        """Should handle empty candlestick response gracefully."""
        empty = {"markets": []}
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=empty))
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            count = await backfill_candlesticks(
                kalshi, db_factory, tickers=["KXCPI-26MAR-T3.0"], series_ticker="KXCPI"
            )

        assert count == 0

    async def test_passes_time_range_to_api(self, db_factory):
        """Should forward start_ts and end_ts to the batch API."""
        request_log: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_log.append(request)
            return httpx.Response(200, json={"markets": []})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0"],
                series_ticker="KXCPI",
                start_ts=1709200000,
                end_ts=1709400000,
            )

        params = request_log[0].url.params
        assert params["start_ts"] == "1709200000"
        assert params["end_ts"] == "1709400000"

    async def test_backfill_deduplicates_on_rerun(self, db_factory):
        """Running backfill twice with same data should not double the rows."""
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=BATCH_RESPONSE))

        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            first = await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0", "KXCPI-26MAR-T3.5"],
                series_ticker="KXCPI",
            )

        # Second run with fresh transport (same data)
        transport2 = httpx.MockTransport(lambda _: httpx.Response(200, json=BATCH_RESPONSE))
        async with httpx.AsyncClient(transport=transport2) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            second = await backfill_candlesticks(
                kalshi,
                db_factory,
                tickers=["KXCPI-26MAR-T3.0", "KXCPI-26MAR-T3.5"],
                series_ticker="KXCPI",
            )

        assert first == 3
        assert second == 0  # all duplicates skipped

        async with db_factory() as session:
            result = await session.execute(select(CandlestickBar))
            rows = result.scalars().all()
        assert len(rows) == 3  # no duplicates
