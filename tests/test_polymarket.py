"""Tests for Polymarket API client."""

from datetime import UTC, datetime

import httpx
import pytest

from arbiter.ingestion.polymarket import PolymarketClient

# --- Realistic Polymarket Gamma API response fixtures ---

GAMMA_MARKET_RESPONSE = [
    {
        "condition_id": "0xabc123",
        "question": "Will Bitcoin hit $100k by March 2026?",
        "category": "Crypto",
        "tokens": [
            {"outcome": "Yes", "price": "0.72"},
            {"outcome": "No", "price": "0.28"},
        ],
        "volume24hr": 340000.0,
        "liquidity": 120000.0,
        "end_date": "2026-03-31T23:59:59Z",
        "market_slug": "will-bitcoin-hit-100k-march-2026",
        "active": True,
        "closed": False,
        "accepting_orders": True,
    },
    {
        "condition_id": "0xdef456",
        "question": "Fed cuts rates in March 2026?",
        "category": "Economics",
        "tokens": [
            {"outcome": "Yes", "price": "0.35"},
            {"outcome": "No", "price": "0.65"},
        ],
        "volume24hr": 89000.0,
        "liquidity": 45000.0,
        "end_date": "2026-03-19T18:00:00Z",
        "market_slug": "fed-cuts-rates-march-2026",
        "active": True,
        "closed": False,
        "accepting_orders": True,
    },
]

GAMMA_MARKET_PAGE2 = [
    {
        "condition_id": "0xghi789",
        "question": "Trump wins 2026 midterm?",
        "category": "Politics",
        "tokens": [
            {"outcome": "Yes", "price": "0.31"},
            {"outcome": "No", "price": "0.69"},
        ],
        "volume24hr": 1200000.0,
        "liquidity": 800000.0,
        "end_date": "2026-11-03T23:59:59Z",
        "market_slug": "trump-wins-2026-midterm",
        "active": True,
        "closed": False,
        "accepting_orders": True,
    }
]

GAMMA_CLOSED_MARKET = [
    {
        "condition_id": "0xclosed",
        "question": "Old resolved market",
        "category": "Other",
        "tokens": [
            {"outcome": "Yes", "price": "1.00"},
            {"outcome": "No", "price": "0.00"},
        ],
        "volume24hr": 0.0,
        "liquidity": 0.0,
        "end_date": "2025-01-01T00:00:00Z",
        "market_slug": "old-resolved-market",
        "active": False,
        "closed": True,
        "accepting_orders": False,
    }
]


def _gamma_transport(pages: list[list[dict]]) -> httpx.MockTransport:
    """Transport returning paginated Gamma API responses."""
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            return httpx.Response(200, json=next(page_iter))
        except StopIteration:
            return httpx.Response(200, json=[])

    return httpx.MockTransport(handler)


class TestPolymarketClientFetchMarkets:
    async def test_returns_contracts(self):
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets(limit=100)

        assert len(contracts) == 2
        assert all(c.source == "polymarket" for c in contracts)

    async def test_normalizes_token_prices(self):
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.yes_price == pytest.approx(0.72)
        assert btc.no_price == pytest.approx(0.28)

    async def test_parses_expiry_datetime(self):
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.expires_at == datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)

    async def test_maps_contract_fields(self):
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.contract_id == "0xabc123"
        assert btc.title == "Will Bitcoin hit $100k by March 2026?"
        assert btc.category == "Crypto"
        assert btc.volume_24h == 340000.0
        assert btc.open_interest == 120000.0  # liquidity maps to open_interest
        assert btc.status == "open"
        assert "polymarket.com" in btc.url

    async def test_offset_pagination_fetches_all_pages(self):
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE, GAMMA_MARKET_PAGE2])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets(limit=2)

        # 2 from page 1 + 1 from page 2 (< limit, so pagination stops)
        assert len(contracts) == 3

    async def test_filters_closed_markets(self):
        transport = _gamma_transport([GAMMA_CLOSED_MARKET])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_empty_response(self):
        transport = _gamma_transport([[]])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert contracts == []

    async def test_bid_ask_derived_from_price(self):
        """Since Gamma API only provides a price, bid/ask approximate around it."""
        transport = _gamma_transport([GAMMA_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        # Bid should be <= price, ask should be >= price
        assert btc.yes_bid <= btc.yes_price
        assert btc.yes_ask >= btc.yes_price

    async def test_raises_on_http_error(self):
        transport = httpx.MockTransport(lambda _req: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_markets()

    async def test_handles_single_yes_token_only(self):
        """Market with only a Yes token derives No price from 1 - yes_price."""
        market = {
            **GAMMA_MARKET_RESPONSE[0],
            "tokens": [{"outcome": "Yes", "price": "0.70"}],
        }
        transport = _gamma_transport([[market]])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].yes_price == pytest.approx(0.70)
        assert contracts[0].no_price == pytest.approx(0.30)

    async def test_skips_market_with_no_tokens(self):
        market = {**GAMMA_MARKET_RESPONSE[0], "tokens": []}
        transport = _gamma_transport([[market]])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_handles_null_volume(self):
        market = {**GAMMA_MARKET_RESPONSE[0], "volume24hr": None}
        transport = _gamma_transport([[market]])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com", min_volume_24h=0)
            contracts = await client.fetch_markets()

        assert contracts[0].volume_24h == 0.0


class TestPolymarketClientClose:
    async def test_close_is_noop_on_borrowed_client(self):
        transport = _gamma_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            await client.close()  # Should not raise
