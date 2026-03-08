"""Tests for background data collector."""

from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.db.models import Base, OrderBookSnapshot, Source
from arbiter.ingestion.base import Contract
from arbiter.ingestion.collector import DataCollector
from arbiter.ingestion.kalshi import KalshiClient


def _make_contract(ticker: str, volume: float = 1000.0) -> Contract:
    return Contract(
        source="kalshi",
        contract_id=ticker,
        title=f"Test {ticker}",
        category="Economics",
        yes_price=0.50,
        no_price=0.50,
        yes_bid=0.48,
        yes_ask=0.52,
        last_price=0.50,
        volume_24h=volume,
        open_interest=100.0,
        expires_at=datetime(2026, 3, 14, tzinfo=UTC),
        url=f"https://kalshi.com/markets/{ticker}",
        status="active",
        series_ticker="KXCPI",
        event_ticker="KXCPI-26MAR",
    )


ORDERBOOK_RESPONSE = {
    "orderbook": {
        "yes": [[65, 100], [60, 200]],
        "no": [[38, 150]],
    },
}


def _orderbook_transport() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    request_log: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        return httpx.Response(200, json=ORDERBOOK_RESPONSE)

    return httpx.MockTransport(handler), request_log


@pytest.fixture
async def db_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestDataCollector:
    async def test_collects_orderbooks_for_top_n_contracts(self, db_factory):
        """Should fetch order books for top-N contracts by volume and persist them."""
        contracts = [
            _make_contract("KXCPI-T3.0", volume=5000),
            _make_contract("KXCPI-T3.5", volume=3000),
            _make_contract("KXCPI-T4.0", volume=1000),
        ]
        transport, requests = _orderbook_transport()
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, db_factory, top_n=2)
            count = await collector.collect_orderbooks(contracts)

        assert count == 2
        assert len(requests) == 2
        # Should fetch the two highest-volume contracts
        urls = [str(r.url) for r in requests]
        assert any("KXCPI-T3.0" in u for u in urls)
        assert any("KXCPI-T3.5" in u for u in urls)

    async def test_persists_order_book_snapshots(self, db_factory):
        """Should write OrderBookSnapshot rows to the database."""
        contracts = [_make_contract("KXCPI-T3.0", volume=5000)]
        transport, _ = _orderbook_transport()
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, db_factory, top_n=5)
            await collector.collect_orderbooks(contracts)

        async with db_factory() as session:
            result = await session.execute(select(OrderBookSnapshot))
            rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].contract_id == "KXCPI-T3.0"
        assert rows[0].source == Source.KALSHI
        assert rows[0].series_ticker == "KXCPI"
        assert len(rows[0].bids) == 2
        assert len(rows[0].asks) == 1

    async def test_skips_failed_fetches(self, db_factory):
        """Should skip contracts that fail to fetch and continue with others."""
        contracts = [
            _make_contract("KXCPI-T3.0", volume=5000),
            _make_contract("KXCPI-T3.5", volume=3000),
        ]
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if "KXCPI-T3.0" in str(request.url):
                return httpx.Response(500)
            return httpx.Response(200, json=ORDERBOOK_RESPONSE)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, db_factory, top_n=5)
            count = await collector.collect_orderbooks(contracts)

        assert count == 1  # only the second contract succeeded

    async def test_empty_contracts_returns_zero(self, db_factory):
        """Should handle empty contract list gracefully."""
        transport, requests = _orderbook_transport()
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, db_factory, top_n=5)
            count = await collector.collect_orderbooks([])

        assert count == 0
        assert len(requests) == 0

    async def test_uses_single_session_for_batch(self, db_factory):
        """Should use a single DB session for all snapshots (batch write)."""
        contracts = [
            _make_contract("KXCPI-T3.0", volume=5000),
            _make_contract("KXCPI-T3.5", volume=3000),
            _make_contract("KXCPI-T4.0", volume=1000),
        ]
        transport, _ = _orderbook_transport()

        session_count = 0
        original_factory = db_factory

        async def counting_factory():
            nonlocal session_count
            session_count += 1
            return await original_factory().__aenter__()

        # Wrap factory to count session creations
        class CountingFactory:
            def __init__(self, factory):
                self._factory = factory
                self.call_count = 0

            def __call__(self):
                self.call_count += 1
                return self._factory()

        counting = CountingFactory(db_factory)

        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, counting, top_n=5)
            count = await collector.collect_orderbooks(contracts)

        assert count == 3
        # Should open exactly 1 session, not 3 (one per snapshot)
        assert counting.call_count == 1

    async def test_filters_non_kalshi_contracts(self, db_factory):
        """Should only collect order books for Kalshi contracts."""
        kalshi_contract = _make_contract("KXCPI-T3.0", volume=5000)
        poly_contract = Contract(
            source="polymarket",
            contract_id="poly-123",
            title="Poly contract",
            category="Economics",
            yes_price=0.50,
            no_price=0.50,
            yes_bid=0.48,
            yes_ask=0.52,
            last_price=0.50,
            volume_24h=10000,
            open_interest=100.0,
            expires_at=None,
            url="https://polymarket.com/123",
            status="active",
        )
        transport, requests = _orderbook_transport()
        async with httpx.AsyncClient(transport=transport) as http:
            kalshi = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            collector = DataCollector(kalshi, db_factory, top_n=5)
            count = await collector.collect_orderbooks([kalshi_contract, poly_contract])

        assert count == 1
        assert len(requests) == 1
