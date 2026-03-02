"""Live-API component tests for Kalshi, Polymarket, Discord, and Supabase.

These tests call the real external services to verify connectivity and
response shapes.  They are marked with ``integration`` so they are excluded
from the normal CI test run and must be invoked explicitly:

    pytest -m integration -v tests/test_integration.py
    make integrate

Kalshi and Polymarket tests require no credentials (public APIs).
Discord and Supabase tests require DISCORD_WEBHOOK_URL and DATABASE_URL in .env.
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import text

from arbiter.config import settings
from arbiter.db.session import engine
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.polymarket import PolymarketClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Kalshi live-API component tests
# ---------------------------------------------------------------------------


class TestKalshiLiveAPI:
    """Verify the real Kalshi elections API returns contracts we can parse."""

    @pytest.fixture
    async def kalshi_contracts(self):
        async with httpx.AsyncClient(timeout=30) as http:
            client = KalshiClient(
                http,
                base_url="https://api.elections.kalshi.com/trade-api/v2",
                max_markets=100,
                min_volume_24h=0.0,
            )
            return await client.fetch_markets(limit=20)

    async def test_returns_at_least_one_contract(self, kalshi_contracts):
        """The live API must return parseable contracts — not an empty list."""
        assert len(kalshi_contracts) > 0, (
            "Kalshi returned 0 contracts. "
            "Check status filter, price field names (yes_bid_dollars etc.), "
            "and that the API is reachable."
        )

    async def test_all_contracts_have_valid_prices(self, kalshi_contracts):
        """Every returned contract must have yes_price in [0, 1].

        Note: yes_price=0.0 is valid for illiquid markets with an empty order
        book and no last trade — the parser includes them for snapshot collection.
        """
        for c in kalshi_contracts:
            assert 0.0 <= c.yes_price <= 1.0, (
                f"Contract {c.contract_id} has invalid yes_price={c.yes_price}"
            )
            assert 0.0 <= c.no_price <= 1.0, (
                f"Contract {c.contract_id} has invalid no_price={c.no_price}"
            )

    async def test_prices_sum_to_approx_one(self, kalshi_contracts):
        """yes_price + no_price should be very close to 1.0."""
        for c in kalshi_contracts:
            total = c.yes_price + c.no_price
            assert abs(total - 1.0) < 1e-6, (
                f"Contract {c.contract_id}: yes_price + no_price = {total}"
            )

    async def test_all_contracts_have_required_fields(self, kalshi_contracts):
        """Every contract must have source, contract_id, title, and status."""
        for c in kalshi_contracts:
            assert c.source == "kalshi"
            assert c.contract_id, f"Empty contract_id: {c}"
            assert c.title, f"Empty title for {c.contract_id}"
            assert c.status in ("open", "active"), (
                f"Unexpected status '{c.status}' for {c.contract_id}"
            )

    async def test_raw_response_contains_dollars_fields(self):
        """Smoke-test that the raw JSON contains *_dollars price fields.

        This will catch a future API schema change before it silently breaks
        the parser.
        """
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"limit": 5, "status": "open"},
            )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        assert markets, "API returned no markets at all"

        first = markets[0]
        # At least one of the _dollars fields must be present in every market
        dollar_fields = {k for k in first if k.endswith("_dollars") or k.endswith("_fp")}
        assert dollar_fields, (
            f"No *_dollars or *_fp fields found in live response.\n"
            f"Available keys: {sorted(first.keys())}"
        )

    async def test_raw_response_status_is_active_not_open(self):
        """Confirm the 'status' field value is 'active', not 'open'.

        This guards against a regression of the primary bug: the query param
        uses 'open' but the response body field value is 'active'.
        """
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"limit": 10, "status": "open"},
            )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        assert markets, "API returned no markets"

        statuses = {m.get("status") for m in markets}
        # The response body uses "active" for open markets
        assert "active" in statuses, (
            f"Expected 'active' in status values, got: {statuses}. "
            "The status filter in _parse_market may need updating."
        )


# ---------------------------------------------------------------------------
# Polymarket live-API component tests
# ---------------------------------------------------------------------------


class TestPolymarketLiveAPI:
    """Verify the real Polymarket Gamma API returns contracts we can parse."""

    @pytest.fixture
    async def poly_contracts(self):
        async with httpx.AsyncClient(timeout=30) as http:
            client = PolymarketClient(
                http,
                gamma_base_url="https://gamma-api.polymarket.com",
                min_volume_24h=100.0,
                max_markets=200,
                max_empty_pages=3,
            )
            return await client.fetch_markets(limit=20)

    async def test_returns_at_least_one_contract(self, poly_contracts):
        """The live API must return parseable contracts — not an empty list."""
        assert len(poly_contracts) > 0, (
            "Polymarket returned 0 contracts. "
            "Check closed=false param, acceptingOrders field name, "
            "outcomes/outcomePrices parsing, and API reachability."
        )

    async def test_all_contracts_have_valid_prices(self, poly_contracts):
        """Every returned contract must have yes_price in (0, 1)."""
        for c in poly_contracts:
            assert 0.0 < c.yes_price < 1.0, (
                f"Contract {c.contract_id} has invalid yes_price={c.yes_price}"
            )

    async def test_all_contracts_have_nonzero_volume(self, poly_contracts):
        """All returned contracts should have passed the 24h volume filter."""
        for c in poly_contracts:
            assert c.volume_24h >= 100.0, (
                f"Contract {c.contract_id} has volume_24h={c.volume_24h} "
                f"below threshold — volume24hr field may be wrong."
            )

    async def test_all_contracts_have_required_fields(self, poly_contracts):
        """Every contract must have source, contract_id, title, and status."""
        for c in poly_contracts:
            assert c.source == "polymarket"
            assert c.contract_id, f"Empty contract_id: {c}"
            assert c.title, f"Empty title for {c.contract_id}"
            assert c.status == "open"

    async def test_raw_response_contains_camel_case_fields(self):
        """Smoke-test that the raw JSON uses camelCase field names.

        This will catch a future API schema change before it silently breaks
        the parser.
        """
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 5},
            )
        resp.raise_for_status()
        markets: list[dict] = resp.json()
        assert markets, "Gamma API returned no markets"

        first = markets[0]
        # These camelCase keys must be present for parsing to work
        for required_key in ("conditionId", "outcomes", "outcomePrices", "volume24hr"):
            assert required_key in first, (
                f"Required field '{required_key}' missing from live response.\n"
                f"Available keys: {sorted(first.keys())}"
            )

    async def test_raw_response_closed_false_returns_open_markets(self):
        """Confirm closed=false actually excludes closed markets from results."""
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 10},
            )
        resp.raise_for_status()
        markets: list[dict] = resp.json()
        assert markets, "Gamma API returned no markets"

        closed_markets = [m for m in markets if m.get("closed")]
        assert closed_markets == [], (
            f"closed=false param returned {len(closed_markets)} closed markets — "
            "the param may no longer be respected by the API."
        )


# ---------------------------------------------------------------------------
# Discord webhook connectivity tests
# ---------------------------------------------------------------------------


class TestDiscordWebhook:
    """Verify the Discord webhook URL is configured and accepts messages."""

    async def test_webhook_url_is_configured(self):
        """DISCORD_WEBHOOK_URL must be set in .env before alerts will work."""
        assert settings.discord_webhook_url, (
            "DISCORD_WEBHOOK_URL is not set. Add it to .env and retry."
        )

    async def test_webhook_delivers_message(self):
        """A POST to the webhook must return 204 (Discord accepted the message)."""
        if not settings.discord_webhook_url:
            pytest.skip("DISCORD_WEBHOOK_URL not configured")
        payload = {"embeds": [{"title": "[ARBITER] integration test ping", "color": 0x2ECC71}]}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(settings.discord_webhook_url, json=payload)
        assert r.status_code == 204, (
            f"Discord webhook returned {r.status_code}. "
            f"Check that the webhook URL is still valid. Body: {r.text}"
        )


# ---------------------------------------------------------------------------
# Supabase (Postgres) connectivity tests
# ---------------------------------------------------------------------------


class TestSupabaseDB:
    """Verify the database connection string is valid and the DB is reachable."""

    async def test_database_url_is_configured(self):
        """DATABASE_URL must point to Postgres, not the local SQLite default."""
        assert settings.database_url, "DATABASE_URL is not set"
        if settings.database_url.startswith("sqlite"):
            pytest.skip("DATABASE_URL is SQLite — set the Supabase Postgres URL in .env")

    async def test_can_connect_and_query(self):
        """A simple SELECT 1 must succeed — confirms auth and network path."""
        if settings.database_url.startswith("sqlite"):
            pytest.skip("DATABASE_URL is SQLite — set the Supabase Postgres URL in .env")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1, "SELECT 1 did not return 1"
