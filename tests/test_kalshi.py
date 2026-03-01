"""Tests for Kalshi API client."""

from datetime import UTC, datetime

import httpx
import pytest

from arbiter.ingestion.kalshi import KalshiClient

# --- Realistic Kalshi API response fixtures ---
# Field names match the current Kalshi v2 API (post-March 2026 _dollars schema).

KALSHI_MARKET_RESPONSE = {
    "markets": [
        {
            "ticker": "KXBTC-26MAR14-T100000",
            "title": "Bitcoin above $100,000 on March 14?",
            "category": "Crypto",
            "yes_bid_dollars": "0.62",
            "yes_ask_dollars": "0.65",
            "no_bid_dollars": "0.35",
            "no_ask_dollars": "0.38",
            "last_price_dollars": "0.63",
            "volume_24h_fp": 15420,
            "open_interest": 52000,
            "close_time": "2026-03-14T23:59:59Z",
            "result": "",
            "status": "active",
            "url": "https://kalshi.com/markets/KXBTC-26MAR14-T100000",
        },
        {
            "ticker": "KXFED-26MAR-T425",
            "title": "Fed funds rate above 4.25% after March meeting?",
            "category": "Economics",
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.44",
            "no_bid_dollars": "0.56",
            "no_ask_dollars": "0.60",
            "last_price_dollars": "0.42",
            "volume_24h_fp": 8300,
            "open_interest": 31000,
            "close_time": "2026-03-19T18:00:00Z",
            "result": "",
            "status": "active",
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
            "yes_bid_dollars": "0.30",
            "yes_ask_dollars": "0.33",
            "no_bid_dollars": "0.67",
            "no_ask_dollars": "0.70",
            "last_price_dollars": "0.31",
            "volume_24h_fp": 250000,
            "open_interest": 920000,
            "close_time": "2026-11-03T23:59:59Z",
            "result": "",
            "status": "active",
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
            "yes_bid_dollars": "0.90",
            "yes_ask_dollars": "0.95",
            "no_bid_dollars": "0.05",
            "no_ask_dollars": "0.10",
            "last_price_dollars": "0.92",
            "volume_24h_fp": 0,
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

    def handler(_request: httpx.Request) -> httpx.Response:
        try:
            return httpx.Response(200, json=next(page_iter))
        except StopIteration:
            return httpx.Response(200, json={"markets": [], "cursor": ""})

    return httpx.MockTransport(handler)


def _counting_kalshi_transport(pages: list[dict]) -> tuple[httpx.MockTransport, list]:
    """Transport that records each request and returns pages in sequence."""
    request_log: list[httpx.Request] = []
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        try:
            return httpx.Response(200, json=next(page_iter))
        except StopIteration:
            return httpx.Response(200, json={"markets": [], "cursor": ""})

    return httpx.MockTransport(handler), request_log


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
        assert btc.status == "active"

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
        market = {**KALSHI_MARKET_RESPONSE["markets"][0], "last_price_dollars": None}
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

    async def test_filters_market_with_null_volume(self):
        """Null volume_24h_fp coerces to 0.0, which is below the default 5.0 threshold."""
        market = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": None}
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0

    async def test_falls_back_to_last_price_when_bid_ask_missing(self):
        """Markets with no active orders use last_price as bid=ask midpoint."""
        market = {
            **KALSHI_MARKET_RESPONSE["markets"][0],
            "yes_bid_dollars": None,
            "yes_ask_dollars": None,
        }
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].yes_price == pytest.approx(0.63)  # last_price used as midpoint
        assert contracts[0].yes_bid == pytest.approx(0.63)
        assert contracts[0].yes_ask == pytest.approx(0.63)

    async def test_skips_market_with_no_bid_ask_and_no_last_price(self):
        market = {
            **KALSHI_MARKET_RESPONSE["markets"][0],
            "yes_bid_dollars": None,
            "yes_ask_dollars": None,
            "last_price_dollars": None,
        }
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert len(contracts) == 0


    async def test_filters_low_volume_markets(self):
        """Markets below min_volume_24h threshold are excluded."""
        low_vol = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": 3.0}
        high_vol = {**KALSHI_MARKET_RESPONSE["markets"][1], "volume_24h_fp": 50.0}
        resp = {"markets": [low_vol, high_vol], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2", min_volume_24h=5.0)
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].volume_24h == 50.0

    async def test_zero_min_volume_returns_all_markets(self):
        """min_volume_24h=0 disables the filter and includes all parseable markets."""
        zero_vol = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": 0.0}
        resp = {"markets": [zero_vol], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2", min_volume_24h=0.0)
            contracts = await client.fetch_markets()

        assert len(contracts) == 1


    async def test_default_limit_is_1000(self):
        """Default limit must be 1000 (API max) to minimise request count."""
        transport, requests = _counting_kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.fetch_markets()

        assert requests[0].url.params["limit"] == "1000"

    async def test_sends_mve_filter_exclude(self):
        """Must send mve_filter=exclude to drop sports combo markets server-side."""
        transport, requests = _counting_kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.fetch_markets()

        assert requests[0].url.params["mve_filter"] == "exclude"

    async def test_stops_after_max_empty_pages(self):
        """Stops when max_empty_pages consecutive pages yield no qualifying contracts."""
        dead_market = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": 0.0}
        dead_page = {"markets": [dead_market, dead_market], "cursor": "cursor_xyz"}
        liquid_page = KALSHI_MARKET_PAGE2  # high volume — should never be fetched

        transport, requests = _counting_kalshi_transport([
            dead_page,
            dead_page,
            dead_page,
            liquid_page,  # unreachable
        ])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                max_empty_pages=3,
                min_volume_24h=5.0,
            )
            contracts = await client.fetch_markets(limit=2)

        assert contracts == []
        assert len(requests) == 3  # stops after 3 consecutive empty pages

    async def test_qualifying_page_resets_empty_counter(self):
        """A page with qualifying contracts resets the consecutive-empty counter."""
        dead_market = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": 0.0}
        dead_page = {"markets": [dead_market], "cursor": "cursor_xyz"}
        liquid_page = {"markets": [KALSHI_MARKET_RESPONSE["markets"][1]], "cursor": "cursor_abc"}

        transport, requests = _counting_kalshi_transport([
            dead_page,
            liquid_page,   # resets counter
            dead_page,
            dead_page,
            dead_page,     # 3 consecutive empty → stops
            liquid_page,   # unreachable
        ])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                max_empty_pages=3,
                min_volume_24h=5.0,
            )
            contracts = await client.fetch_markets(limit=1)

        assert len(contracts) == 1
        assert len(requests) == 5

    async def test_empty_page_with_cursor_breaks_immediately(self):
        """Empty page breaks even if cursor is non-empty (stale cursor guard)."""
        stale_response = {"markets": [], "cursor": "stale_cursor"}
        transport, requests = _counting_kalshi_transport([stale_response])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert contracts == []
        assert len(requests) == 1  # breaks immediately, no infinite loop


class TestKalshiClientClose:
    async def test_close_is_noop_on_borrowed_client(self):
        """KalshiClient borrows the http client; close() is safe."""
        transport = _kalshi_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.close()  # Should not raise
