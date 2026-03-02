"""Tests for Polymarket API client."""

from datetime import UTC, datetime

import httpx
import pytest

from arbiter.ingestion.polymarket import _MIN_TOTAL_VOLUME_PREFILTER, PolymarketClient

# --- Realistic Polymarket Gamma API response fixtures ---
# Field names match the real Gamma API camelCase schema.
# outcomes/outcomePrices are JSON-encoded strings as returned by the API.

GAMMA_MARKET_RESPONSE = [
    {
        "conditionId": "0xabc123",
        "question": "Will Bitcoin hit $100k by March 2026?",
        "category": "Crypto",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.72", "0.28"]',
        "volume24hr": 340000.0,
        "liquidity": 120000.0,
        "endDate": "2026-03-31T23:59:59Z",
        "slug": "will-bitcoin-hit-100k-march-2026",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
    },
    {
        "conditionId": "0xdef456",
        "question": "Fed cuts rates in March 2026?",
        "category": "Economics",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.35", "0.65"]',
        "volume24hr": 89000.0,
        "liquidity": 45000.0,
        "endDate": "2026-03-19T18:00:00Z",
        "slug": "fed-cuts-rates-march-2026",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
    },
]

GAMMA_CLOSED_MARKET = [
    {
        "conditionId": "0xclosed",
        "question": "Old resolved market",
        "category": "Other",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["1.00", "0.00"]',
        "volume24hr": 0.0,
        "liquidity": 0.0,
        "endDate": "2025-01-01T00:00:00Z",
        "slug": "old-resolved-market",
        "active": False,
        "closed": True,
        "acceptingOrders": False,
    }
]


def _gamma_transport(response: list[dict]) -> httpx.MockTransport:
    """Transport returning a single Gamma API response."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handler)


def _capturing_transport(response: list[dict]) -> tuple[httpx.MockTransport, list]:
    """Transport that captures each request; useful for asserting params."""
    request_log: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handler), request_log


class TestPolymarketClientFetchMarkets:
    async def test_returns_contracts(self):
        transport = _gamma_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 2
        assert all(c.source == "polymarket" for c in contracts)

    async def test_normalizes_token_prices(self):
        transport = _gamma_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.yes_price == pytest.approx(0.72)
        assert btc.no_price == pytest.approx(0.28)

    async def test_parses_expiry_datetime(self):
        transport = _gamma_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.expires_at == datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)

    async def test_maps_contract_fields(self):
        transport = _gamma_transport(GAMMA_MARKET_RESPONSE)
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

    async def test_sends_volume_sort_params(self):
        """Must send order=volume24hr&ascending=false to get top markets in one request."""
        transport, requests = _capturing_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            await client.fetch_markets()

        assert len(requests) == 1
        params = dict(requests[0].url.params)
        assert params["order"] == "volume24hr"
        assert params["ascending"] == "false"

    async def test_sends_volume_prefilter_param(self):
        """Must send volume_num_min=_MIN_TOTAL_VOLUME_PREFILTER server-side."""
        transport, requests = _capturing_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            await client.fetch_markets()

        params = dict(requests[0].url.params)
        assert float(params["volume_num_min"]) == _MIN_TOTAL_VOLUME_PREFILTER

    async def test_single_request_no_pagination(self):
        """Client must issue exactly one HTTP request regardless of response size."""
        transport, requests = _capturing_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            await client.fetch_markets()

        assert len(requests) == 1

    async def test_filters_closed_markets(self):
        transport = _gamma_transport(GAMMA_CLOSED_MARKET)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_empty_response(self):
        transport = _gamma_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert contracts == []

    async def test_bid_ask_derived_from_price(self):
        """Since Gamma API only provides a price, bid/ask approximate around it."""
        transport = _gamma_transport(GAMMA_MARKET_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        btc = contracts[0]
        assert btc.yes_bid <= btc.yes_price
        assert btc.yes_ask >= btc.yes_price

    async def test_raises_on_http_error(self):
        transport = httpx.MockTransport(lambda _req: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_markets()

    async def test_handles_single_yes_outcome_only(self):
        """Market with only a Yes outcome derives No price from 1 - yes_price."""
        market = {
            **GAMMA_MARKET_RESPONSE[0],
            "outcomes": '["Yes"]',
            "outcomePrices": '["0.70"]',
        }
        transport = _gamma_transport([market])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].yes_price == pytest.approx(0.70)
        assert contracts[0].no_price == pytest.approx(0.30)

    async def test_skips_market_with_no_outcomes(self):
        market = {
            **GAMMA_MARKET_RESPONSE[0],
            "outcomes": "[]",
            "outcomePrices": "[]",
        }
        transport = _gamma_transport([market])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_handles_null_volume(self):
        market = {**GAMMA_MARKET_RESPONSE[0], "volume24hr": None}
        transport = _gamma_transport([market])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(
                http, gamma_base_url="https://gamma-api.polymarket.com", min_volume_24h=0
            )
            contracts = await client.fetch_markets()

        assert contracts[0].volume_24h == 0.0

    async def test_filters_low_24h_volume(self):
        """Markets below min_volume_24h are excluded by client-side filter."""
        low_vol = {**GAMMA_MARKET_RESPONSE[0], "volume24hr": 50.0}
        high_vol = {**GAMMA_MARKET_RESPONSE[1], "volume24hr": 500.0}
        transport = _gamma_transport([low_vol, high_vol])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(
                http, gamma_base_url="https://gamma-api.polymarket.com", min_volume_24h=100.0
            )
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].volume_24h == 500.0


class TestPolymarketClientClose:
    async def test_close_is_noop_on_borrowed_client(self):
        transport = _gamma_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = PolymarketClient(http, gamma_base_url="https://gamma-api.polymarket.com")
            await client.close()  # Should not raise
