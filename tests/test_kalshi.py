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

KALSHI_SETTLED_MARKET_RESPONSE = {
    "markets": [
        {
            "ticker": "KXBTC-26JAN15-T95000",
            "event_ticker": "KXBTC-26JAN15",
            "title": "Bitcoin above $95,000 on January 15?",
            "category": "Crypto",
            "last_price": 85,
            "result": "no",
            "volume": 12400,
            "volume_24h": 0,
            "floor_strike": 95000.0,
            "cap_strike": None,
            "close_time": "2026-01-15T23:59:59Z",
            "status": "finalized",
            "settlement_value": 0,
        },
        {
            "ticker": "KXBTC-26JAN15-T90000",
            "event_ticker": "KXBTC-26JAN15",
            "title": "Bitcoin above $90,000 on January 15?",
            "category": "Crypto",
            "last_price": 72,
            "result": "yes",
            "volume": 8900,
            "volume_24h": 0,
            "floor_strike": 90000.0,
            "cap_strike": None,
            "close_time": "2026-01-15T23:59:59Z",
            "status": "finalized",
            "settlement_value": 100,
        },
    ],
    "cursor": "settled_cursor_1",
}

KALSHI_SETTLED_PAGE2 = {
    "markets": [
        {
            "ticker": "KXBTC-26JAN15-T85000",
            "event_ticker": "KXBTC-26JAN15",
            "title": "Bitcoin above $85,000 on January 15?",
            "category": "Crypto",
            "last_price": 90,
            "result": "yes",
            "volume": 5200,
            "volume_24h": 0,
            "floor_strike": 85000.0,
            "cap_strike": None,
            "close_time": "2026-01-15T23:59:59Z",
            "status": "finalized",
            "settlement_value": 100,
        },
    ],
    "cursor": "",
}

KALSHI_SETTLED_ZERO_VOLUME = {
    "markets": [
        {
            "ticker": "KXBTC-26JAN15-T120000",
            "event_ticker": "KXBTC-26JAN15",
            "title": "Bitcoin above $120,000 on January 15?",
            "category": "Crypto",
            "last_price": 5,
            "result": "no",
            "volume": 0,
            "volume_24h": 0,
            "floor_strike": 120000.0,
            "cap_strike": None,
            "close_time": "2026-01-15T23:59:59Z",
            "status": "finalized",
            "settlement_value": 0,
        },
        {
            "ticker": "KXBTC-26JAN15-T100000",
            "event_ticker": "KXBTC-26JAN15",
            "title": "Bitcoin above $100,000 on January 15?",
            "category": "Crypto",
            "last_price": 60,
            "result": "no",
            "volume": 7300,
            "volume_24h": 0,
            "floor_strike": 100000.0,
            "cap_strike": None,
            "close_time": "2026-01-15T23:59:59Z",
            "status": "finalized",
            "settlement_value": 0,
        },
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
            client = KalshiClient(
                http, base_url="https://api.kalshi.com/trade-api/v2", min_volume_24h=5.0
            )
            contracts = await client.fetch_markets()

        assert len(contracts) == 1
        assert contracts[0].volume_24h == 50.0

    async def test_zero_min_volume_returns_all_markets(self):
        """min_volume_24h=0 disables the filter and includes all parseable markets."""
        zero_vol = {**KALSHI_MARKET_RESPONSE["markets"][0], "volume_24h_fp": 0.0}
        resp = {"markets": [zero_vol], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                min_volume_24h=0.0,
            )
            contracts = await client.fetch_markets()

        assert len(contracts) == 1

    async def test_missing_url_generates_kalshi_deep_link(self):
        """When API returns no url, generate a two-segment kalshi.com deep link."""
        market = {**KALSHI_MARKET_RESPONSE["markets"][0], "url": None}
        resp = {"markets": [market], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        # Two-segment format: /markets/{series_prefix}/{ticker}
        # Series prefix = text before first '-' in ticker
        assert contracts[0].url == "https://kalshi.com/markets/KXBTC/KXBTC-26MAR14-T100000"

    async def test_provided_url_is_preserved(self):
        """When API returns a url, use it as-is."""
        resp = {"markets": [KALSHI_MARKET_RESPONSE["markets"][0]], "cursor": ""}
        transport = _kalshi_transport([resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            contracts = await client.fetch_markets()

        assert contracts[0].url == "https://kalshi.com/markets/KXBTC-26MAR14-T100000"

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

        transport, requests = _counting_kalshi_transport(
            [
                dead_page,
                dead_page,
                dead_page,
                liquid_page,  # unreachable
            ]
        )
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

        transport, requests = _counting_kalshi_transport(
            [
                dead_page,
                liquid_page,  # resets counter
                dead_page,
                dead_page,
                dead_page,  # 3 consecutive empty → stops
                liquid_page,  # unreachable
            ]
        )
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


class TestKalshiClientFetchSettled:
    async def test_fetch_settled_returns_raw_dicts(self):
        transport = _kalshi_transport([KALSHI_SETTLED_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            markets = await client.fetch_settled(series_ticker="KXBTC")

        assert isinstance(markets, list)
        assert len(markets) == 2
        assert all(isinstance(m, dict) for m in markets)
        assert markets[0]["ticker"] == "KXBTC-26JAN15-T95000"
        assert markets[0]["result"] == "no"
        assert markets[1]["settlement_value"] == 100

    async def test_fetch_settled_sends_correct_params(self):
        transport, requests = _counting_kalshi_transport([KALSHI_SETTLED_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.fetch_settled(series_ticker="KXBTC", limit=500)

        params = requests[0].url.params
        assert params["status"] == "settled"
        assert params["series_ticker"] == "KXBTC"
        assert params["mve_filter"] == "exclude"
        assert params["limit"] == "500"

    async def test_fetch_settled_paginates(self):
        transport = _kalshi_transport([KALSHI_SETTLED_MARKET_RESPONSE, KALSHI_SETTLED_PAGE2])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            markets = await client.fetch_settled(series_ticker="KXBTC")

        # 2 from page 1 + 1 from page 2
        assert len(markets) == 3

    async def test_fetch_settled_filters_zero_volume(self):
        transport = _kalshi_transport([KALSHI_SETTLED_ZERO_VOLUME])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            markets = await client.fetch_settled(series_ticker="KXBTC")

        # Only the market with volume=7300 should remain
        assert len(markets) == 1
        assert markets[0]["ticker"] == "KXBTC-26JAN15-T100000"

    async def test_fetch_settled_passes_time_filters(self):
        transport, requests = _counting_kalshi_transport([KALSHI_SETTLED_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.fetch_settled(
                series_ticker="KXBTC",
                min_close_ts=1704067200,
                max_close_ts=1706745600,
            )

        params = requests[0].url.params
        assert params["min_close_ts"] == "1704067200"
        assert params["max_close_ts"] == "1706745600"

    async def test_fetch_settled_stops_at_max_markets(self):
        transport, requests = _counting_kalshi_transport(
            [KALSHI_SETTLED_MARKET_RESPONSE, KALSHI_SETTLED_PAGE2]
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                max_markets=2,
            )
            markets = await client.fetch_settled(series_ticker="KXBTC")

        # Should stop after page 1 (2 markets fetched >= max_markets=2)
        assert len(requests) == 1
        assert len(markets) == 2


class TestKalshiClientSeriesTickers:
    async def test_sends_series_ticker_param_per_series(self):
        """With series_tickers set, makes one request per series with series_ticker param."""
        cpi_resp = {"markets": [KALSHI_MARKET_RESPONSE["markets"][0]], "cursor": ""}
        payrolls_resp = {"markets": [KALSHI_MARKET_RESPONSE["markets"][1]], "cursor": ""}
        transport, requests = _counting_kalshi_transport([cpi_resp, payrolls_resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                series_tickers=["KXCPI", "KXPAYROLLS"],
            )
            await client.fetch_markets()

        assert len(requests) == 2
        assert requests[0].url.params["series_ticker"] == "KXCPI"
        assert requests[1].url.params["series_ticker"] == "KXPAYROLLS"

    async def test_combines_results_from_multiple_series(self):
        """Results from all targeted series are combined into one list."""
        cpi_market = {
            **KALSHI_MARKET_RESPONSE["markets"][0],
            "ticker": "KXCPI-26MAR-T3.0",
            "category": "Economics",
        }
        payrolls_market = {
            **KALSHI_MARKET_RESPONSE["markets"][0],
            "ticker": "KXPAYROLLS-26MAR-T200K",
            "category": "Economics",
        }
        cpi_resp = {"markets": [cpi_market], "cursor": ""}
        payrolls_resp = {"markets": [payrolls_market], "cursor": ""}
        transport = _kalshi_transport([cpi_resp, payrolls_resp])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                series_tickers=["KXCPI", "KXPAYROLLS"],
            )
            contracts = await client.fetch_markets()

        assert len(contracts) == 2
        tickers = {c.contract_id for c in contracts}
        assert tickers == {"KXCPI-26MAR-T3.0", "KXPAYROLLS-26MAR-T200K"}

    async def test_no_series_tickers_omits_param(self):
        """Without series_tickers, no series_ticker param is sent (current behavior)."""
        transport, requests = _counting_kalshi_transport([KALSHI_MARKET_RESPONSE])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.fetch_markets()

        assert "series_ticker" not in requests[0].url.params

    async def test_paginates_within_each_series(self):
        """Each series is paginated independently."""
        transport, requests = _counting_kalshi_transport(
            [
                KALSHI_MARKET_RESPONSE,  # series 1, page 1 (cursor=abc123)
                KALSHI_MARKET_PAGE2,  # series 1, page 2 (cursor="")
            ]
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(
                http,
                base_url="https://api.kalshi.com/trade-api/v2",
                series_tickers=["KXCPI"],
            )
            contracts = await client.fetch_markets()

        assert len(requests) == 2
        # Both requests target the same series
        assert requests[0].url.params["series_ticker"] == "KXCPI"
        assert requests[1].url.params["series_ticker"] == "KXCPI"
        assert len(contracts) == 3  # 2 from page 1 + 1 from page 2


class TestKalshiClientClose:
    async def test_close_is_noop_on_borrowed_client(self):
        """KalshiClient borrows the http client; close() is safe."""
        transport = _kalshi_transport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = KalshiClient(http, base_url="https://api.kalshi.com/trade-api/v2")
            await client.close()  # Should not raise
