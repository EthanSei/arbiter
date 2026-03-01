"""Discord webhook alert channel."""

import logging

import httpx

from arbiter.alerts.base import AlertChannel
from arbiter.scoring.ev import ScoredOpportunity

logger = logging.getLogger(__name__)

# Green = positive EV, matches Discord's "green" embed color
_EMBED_COLOR = 0x2ECC71


class DiscordChannel(AlertChannel):
    """Sends alerts via Discord webhook.

    Degrades gracefully if DISCORD_WEBHOOK_URL is not configured.
    """

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url
        self._enabled = bool(webhook_url)
        self._client = httpx.AsyncClient()
        if not self._enabled:
            logger.warning("Discord not configured — alerts will be skipped")

    async def send(self, opportunity: ScoredOpportunity) -> None:
        if not self._enabled:
            return

        payload = self._build_payload(opportunity)
        try:
            response = await self._client.post(url=self._webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning("Discord alert failed: %s", exc)

    async def close(self) -> None:
        await self._client.aclose()

    def _build_payload(self, opp: ScoredOpportunity) -> dict[str, object]:
        """Build a Discord webhook payload with a rich embed."""
        embed = {
            "title": f"[{opp.contract.source.upper()}] {opp.contract.title}",
            "url": opp.contract.url,
            "color": _EMBED_COLOR,
            "fields": [
                {"name": "Direction", "value": opp.direction.upper(), "inline": True},
                {"name": "Market Price", "value": f"{opp.market_price:.1%}", "inline": True},
                {
                    "name": "Model Probability",
                    "value": f"{opp.model_probability:.1%}",
                    "inline": True,
                },
                {
                    "name": "Expected Value",
                    "value": f"{opp.expected_value:+.1%}",
                    "inline": True,
                },
                {"name": "Kelly Size", "value": f"{opp.kelly_size:.1%}", "inline": True},
            ],
        }
        return {"embeds": [embed]}
