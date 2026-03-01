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

    async def test_close_closes_client(self) -> None:
        """close() shuts down the underlying httpx client."""
        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/test/token")
        with patch.object(channel._client, "aclose", new_callable=AsyncMock) as mock_close:
            await channel.close()
        mock_close.assert_called_once()
