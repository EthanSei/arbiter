"""Tests for Kalshi API client."""

from datetime import UTC, datetime

import httpx
import pytest

from arbiter.ingestion.kalshi import KalshiClient

# --- Realistic Kalshi API response fixtures ---

KALSHI_MARKET_RESPONSE = {
    "markets": [
        {
            "ticker": "KXBTC-26MAR14-T100000",
            "title": "Bitcoin above $100,000 on March 14?",
            "category": "Crypto",
            "yes_bid": "0.62",
            "yes_ask": "0.65",
            "no_bid": "0.35",
            "no_ask": "0.38",
            "last_price": "0.63",
            "volume_24h": 15420,
            "open_interest": 52000,
            "close_time": "2026-03-14T23:59:59Z",
            "result": "",
            "status": "open",
            "url": "https://kalshi.com/markets/KXBTC-26MAR14-T100000",
        },
        {
            "ticker": "KXFED-26MAR-T425",
            "title": "Fed funds rate above 4.25% after March meeting?",
            "category": "Economics",
            "yes_bid": "0.40",
            "yes_ask": "0.44",
            "no_bid": "0.56",
            "no_ask": "0.60",
            "last_price": "0.42",
            "volume_24h": 8300,
            "open_interest": 31000,
            "close_time": "2026-03-19T18:00:00Z",
            "result": "",
            "status": "open",
            "url": "https://kalshi.com/markets/KXFED-26MAR-T425",
        },
    ],
    "cursor": "abc123",
}

KALSHI_MARKET_PAGE2 = {
    "markets": [
        {
            "ticker": "KXELECTION-26NOV-TRUMP",
            "title": "Trump wins 2026 midterm?",
            "category": "Politics",
            "yes_bid": "0.30",
            "yes_ask": "0.33",
            "no_bid": "0.67",
            "no_ask": "0.70",
            "last_price": "0.31",
            "volume_24h": 250000,
            "open_interest": 920000,
            "close_time": "2026-11-03T23:59:59Z",
            "result": "",
            "status": "open",
            "url": "https://kalshi.com/markets/KXELECTION-26NOV-TRUMP",
        }
    ],
    "cursor": "",
}

KALSHI_CLOSED_MARKET = {
    "markets": [
        {
            "ticker": "KXOLD-CLOSED",
            "title": "Old closed market",
            "category": "Other",
            "yes_bid": "0.90",
            "yes_ask": "0.95",
            "no_bid": "0.05",
            "no_ask": "0.10",
            "last_price": "0.92",
            "volume_24h": 0,
            "open_interest": 100,
            "close_time": "2025-01-01T00:00:00Z",
            "result": "yes",
            "status": "closed",
            "url": "https://kalshi.com/markets/KXOLD-CLOSED",
        }
    ],
    "cursor": "",
}


def _kalshi_transport(pages: list[dict]) -> httpx.MockTransport:
    """Transport returning paginated Kalshi responses."""
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            return httpx.Response(200, json=next(page_iter))
        except StopIteration:
            return httpx.Response(200, json={"markets": [], "cursor": ""})

    return httpx.MockTransport(handler)


class TestKalshiClientFetchMarkets:
    async def test_returns_contracts(self):
        transport = _kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets(limit=100)

        assert len(contracts) == 2
        assert all(c.source == "kalshi" for c in contracts)

    async def test_normalizes_prices_to_floats(self):
        transport = _kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.yes_bid == pytest.approx(0.62)
        assert btc.yes_ask == pytest.approx(0.65)
        assert btc.yes_price == pytest.approx(0.635)  # midpoint of bid/ask
        assert btc.no_price == pytest.approx(1.0 - 0.635)

    async def test_parses_expiry_datetime(self):
        transport = _kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.expires_at == datetime(2026, 3, 14, 23, 59, 59, tzinfo=UTC)

    async def test_maps_contract_fields(self):
        transport = _kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.contract_id == "KXBTC-26MAR14-T100000"
        assert btc.title == "Bitcoin above $100,000 on March 14?"
        assert btc.category == "Crypto"
        assert btc.volume_24h == 15420
        assert btc.open_interest == 52000
        assert btc.last_price == pytest.approx(0.63)
        assert btc.url == "https://kalshi.com/markets/KXBTC-26MAR14-T100000"
        assert btc.status == "open"

    async def test_cursor_pagination_fetches_all_pages(self):
        transport = _kalshi_transport([KALSHI_MARKET_RESPONSE, KALSHI_MARKET_PAGE2])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets(limit=2)

        # 2 from page 1 + 1 from page 2
        assert len(contracts) == 3

    async def test_filters_non_open_markets(self):
        transport = _kalshi_transport([KALSHI_CLOSED_MARKET])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_empty_response(self):
        transport = _kalshi_transport([{"markets": [], "cursor": ""}])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert contracts == []

    async def test_handles_missing_last_price(self):
        market = {**KALSHI_MARKET_RESPONSE["markets"][0], "last_price": None}
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert contracts[0].last_price is None

    async def test_raises_on_http_error(self):
        transport = httpx.MockTransport(lambda _req: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_markets()

    async def test_skips_market_with_null_volume(self):
        market = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h": None}
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert contracts[0].volume_24h == 0.0

    async def test_skips_market_with_missing_bid_ask(self):
        market = {**KALSHI_MARKET_RESPONSE["markets"][0]}
        del market["yes_bid"]
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0


class TestKalshiClientClose:
    async def test_close_is_noop_on_borrowed_client(self):
        """KalshiClient borrows the http client; close() is safe."""
        transport = _kalshi_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.close()  # Should not raise
