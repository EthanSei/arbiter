"""Tests for alert channels (Discord webhook)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from arbiter.alerts.discord import DiscordChannel
from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity


def _make_opportunity(
    direction: str = "yes",
    market_price: float = 0.40,
    model_probability: float = 0.55,
    expected_value: float = 0.14,
    kelly_size: float = 0.27,
) -> ScoredOpportunity:
    """Helper to create a test ScoredOpportunity."""
    contract = Contract(
        source="kalshi",
        contract_id="TEST-001",
        title="Will BTC exceed $100k?",
        category="crypto",
        yes_price=0.40,
        no_price=0.60,
        yes_bid=0.39,
        yes_ask=0.41,
        last_price=None,
        volume_24h=50000.0,
        open_interest=25000.0,
        expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        url="https://kalshi.com/markets/test-001",
        status="open",
    )
    return ScoredOpportunity(
        contract=contract,
        direction=direction,
        market_price=market_price,
        model_probability=model_probability,
        expected_value=expected_value,
        kelly_size=kelly_size,
    )


class TestDiscordChannel:
    def test_disabled_when_no_webhook_url(self) -> None:
        """Channel degrades gracefully when webhook URL is empty."""
        channel = DiscordChannel(webhook_url="")
        assert channel._enabled is False

    async def test_send_skips_when_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """Disabled channel warns once at init and send() silently skips."""
        with caplog.at_level("WARNING"):
            channel = DiscordChannel(webhook_url="")
            await channel.send(_make_opportunity())
        assert "Discord not configured" in caplog.text

    async def test_send_posts_to_webhook(self) -> None:
        """send() POSTs a JSON payload to the webhook URL."""
        url = "https://discord.com/api/webhooks/test/token"
        mock_response = httpx.Response(204, request=httpx.Request("POST", url))
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(_make_opportunity())

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["url"] == url
        payload = call_kwargs.kwargs["json"]
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    async def test_embed_contains_opportunity_fields(self) -> None:
        """Discord embed includes key opportunity data."""
        url = "https://discord.com/api/webhooks/test/token"
        mock_response = httpx.Response(204, request=httpx.Request("POST", url))
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(_make_opportunity())

        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        assert "BTC" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Direction" in field_names
        assert "Expected Value" in field_names
        assert "Model Probability" in field_names

    async def test_handles_http_error_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """send() logs a warning on HTTP error instead of raising."""
        url = "https://discord.com/api/webhooks/test/token"
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("POST", url),
                response=httpx.Response(500),
            )
            with caplog.at_level("WARNING"):
                await channel.send(_make_opportunity())

        assert "Discord alert failed" in caplog.text

    async def test_handles_network_error_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """send() logs a warning on connection error instead of raising."""
        url = "https://discord.com/api/webhooks/test/token"
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with caplog.at_level("WARNING"):
                await channel.send(_make_opportunity())

        assert "Discord alert failed" in caplog.text

    async def test_handles_timeout_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """send() logs a warning on request timeout instead of raising."""
        url = "https://discord.com/api/webhooks/test/token"
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")
            with caplog.at_level("WARNING"):
                await channel.send(_make_opportunity())

        assert "Discord alert failed" in caplog.text

    async def test_close_closes_client(self) -> None:
        """close() shuts down the underlying httpx client."""
        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/test/token")
        with patch.object(channel._client, "aclose", new_callable=AsyncMock) as mock_close:
            await channel.close()
        mock_close.assert_called_once()

    async def test_consistency_arb_shows_both_legs(self) -> None:
        """Consistency-arb alerts show both legs when anchor_contract is set."""
        url = "https://discord.com/api/webhooks/test/token"
        mock_response = httpx.Response(204, request=httpx.Request("POST", url))
        channel = DiscordChannel(webhook_url=url)

        contract = Contract(
            source="kalshi",
            contract_id="KXBTCMAXMON-BTC-26MAR31-8000000",
            title="Will the BTC trimmed mean price be above $80,000 by March 31?",
            category="",
            yes_price=0.31,
            no_price=0.69,
            yes_bid=0.30,
            yes_ask=0.32,
            last_price=0.31,
            volume_24h=342.0,
            open_interest=1200.0,
            expires_at=datetime(2026, 3, 31, tzinfo=UTC),
            url="https://kalshi.com/markets/KXBTCMAXMON/KXBTCMAXMON-BTC-26MAR31-8000000",
            status="open",
        )
        anchor = Contract(
            source="kalshi",
            contract_id="KXBTCMAXMON-BTC-26MAR31-8250000",
            title="Will the BTC trimmed mean price be above $82,500 by March 31?",
            category="",
            yes_price=0.50,
            no_price=0.50,
            yes_bid=0.49,
            yes_ask=0.51,
            last_price=0.50,
            volume_24h=500.0,
            open_interest=2000.0,
            expires_at=datetime(2026, 3, 31, tzinfo=UTC),
            url="https://kalshi.com/markets/KXBTCMAXMON/KXBTCMAXMON-BTC-26MAR31-8250000",
            status="open",
        )
        opp = ScoredOpportunity(
            contract=contract,
            direction="yes",
            market_price=0.31,
            model_probability=0.50,
            expected_value=0.18,
            kelly_size=0.265,
            anchor_contract=anchor,
        )

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(opp)

        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        field_map = {f["name"]: f["value"] for f in embed["fields"]}

        # Leg 1: BUY YES on the underpriced contract, with side=yes deep link
        assert "Leg 1" in field_map
        assert "BUY YES" in field_map["Leg 1"]
        assert "31" in field_map["Leg 1"]
        assert "action=buy" in field_map["Leg 1"]
        assert "side=yes" in field_map["Leg 1"]

        # Leg 2: BUY NO on the anchor contract, with side=no deep link
        assert "Leg 2" in field_map
        assert "BUY NO" in field_map["Leg 2"]
        assert "50" in field_map["Leg 2"]
        assert "action=buy" in field_map["Leg 2"]
        assert "side=no" in field_map["Leg 2"]

        # Locked profit per pair
        profit_field = [v for k, v in field_map.items() if "Profit" in k][0]
        assert "19" in profit_field

        # Payoff breakdown with arb math
        assert "Payoffs" in field_map
        payoffs = field_map["Payoffs"]
        assert "81" in payoffs  # total cost: 31 + 50
        assert "19" in payoffs  # worst-case profit
        assert "119" in payoffs  # best-case profit (middle outcome)

    async def test_consistency_arb_without_anchor_falls_back(self) -> None:
        """Consistency-arb without anchor_contract shows single-leg format."""
        url = "https://discord.com/api/webhooks/test/token"
        mock_response = httpx.Response(204, request=httpx.Request("POST", url))
        channel = DiscordChannel(webhook_url=url)

        contract = Contract(
            source="kalshi",
            contract_id="KXBTCMAXMON-BTC-26MAR31-8000000",
            title="Will the BTC trimmed mean price be above $80,000 by March 31?",
            category="",
            yes_price=0.31,
            no_price=0.69,
            yes_bid=0.30,
            yes_ask=0.32,
            last_price=0.31,
            volume_24h=342.0,
            open_interest=1200.0,
            expires_at=datetime(2026, 3, 31, tzinfo=UTC),
            url="https://kalshi.com/markets/KXBTCMAXMON/KXBTCMAXMON-BTC-26MAR31-8000000",
            status="open",
        )
        opp = ScoredOpportunity(
            contract=contract,
            direction="yes",
            market_price=0.31,
            model_probability=0.50,
            expected_value=0.18,
            kelly_size=0.265,
        )

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(opp)

        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        field_map = {f["name"]: f["value"] for f in embed["fields"]}
        # Falls back to single-action format
        assert "Action" in field_map
        assert "BUY YES" in field_map["Action"]

    async def test_non_consistency_embed_uses_standard_format(self) -> None:
        """Non-consistency alerts keep the original format (Direction, Model Probability)."""
        url = "https://discord.com/api/webhooks/test/token"
        mock_response = httpx.Response(204, request=httpx.Request("POST", url))
        channel = DiscordChannel(webhook_url=url)

        with patch.object(channel._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(_make_opportunity())

        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        field_names = {f["name"] for f in embed["fields"]}
        assert "Direction" in field_names
        assert "Model Probability" in field_names
