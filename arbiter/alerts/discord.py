"""Discord webhook alert channel."""

import logging
import re

import httpx

from arbiter.alerts.base import AlertChannel
from arbiter.scoring.ev import ScoredOpportunity

logger = logging.getLogger(__name__)

# Green = positive EV, matches Discord's "green" embed color
_EMBED_COLOR = 0x2ECC71

# Detect consistency-arb contracts by MAXMON/MINMON in the ticker.
_CONSISTENCY_RE = re.compile(r"MAXMON|MINMON", re.IGNORECASE)


def _is_consistency_arb(opp: ScoredOpportunity) -> bool:
    return bool(_CONSISTENCY_RE.search(opp.contract.contract_id))


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
        if _is_consistency_arb(opp):
            return self._build_consistency_payload(opp)
        return self._build_standard_payload(opp)

    def _build_standard_payload(self, opp: ScoredOpportunity) -> dict[str, object]:
        """Standard embed for model-based EV alerts."""
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

    def _build_consistency_payload(self, opp: ScoredOpportunity) -> dict[str, object]:
        """Action-oriented embed for range-market consistency violations."""
        price_cents = round(opp.market_price * 100)
        floor_cents = round(opp.model_probability * 100)

        if opp.anchor_contract is not None:
            return self._build_two_leg_payload(opp, price_cents, floor_cents)
        return self._build_single_leg_payload(opp, price_cents, floor_cents)

    def _build_single_leg_payload(
        self, opp: ScoredOpportunity, price_cents: int, floor_cents: int
    ) -> dict[str, object]:
        """Fallback single-leg embed when anchor contract is not available."""
        embed = {
            "title": f"\u26a0\ufe0f Consistency Arb: {opp.contract.contract_id}",
            "url": opp.contract.url,
            "color": 0xF39C12,
            "description": opp.contract.title,
            "fields": [
                {
                    "name": "Action",
                    "value": f"**BUY YES @ {price_cents}\u00a2**",
                    "inline": True,
                },
                {
                    "name": "Floor",
                    "value": f"{floor_cents}\u00a2 (sibling price)",
                    "inline": True,
                },
                {
                    "name": "Expected Value",
                    "value": f"{opp.expected_value:+.1%}",
                    "inline": True,
                },
                {
                    "name": "Kelly Size",
                    "value": f"{opp.kelly_size:.1%}",
                    "inline": True,
                },
            ],
            "footer": {"text": "Monotonicity violation \u2014 acts fast, typically <1 min"},
        }
        return {"embeds": [embed]}

    def _build_two_leg_payload(
        self, opp: ScoredOpportunity, price_cents: int, floor_cents: int
    ) -> dict[str, object]:
        """Two-leg embed showing both sides of the consistency arb."""
        assert opp.anchor_contract is not None
        no_price_cents = round((1.0 - opp.anchor_contract.yes_price) * 100)
        profit_cents = floor_cents - price_cents
        cost_cents = price_cents + no_price_cents
        best_profit = 200 - cost_cents  # both legs pay out (middle outcome)

        leg1_url = f"{opp.contract.url}?action=buy&side=yes"
        leg2_url = f"{opp.anchor_contract.url}?action=buy&side=no"

        embed = {
            "title": f"\u26a0\ufe0f Consistency Arb: {opp.contract.contract_id}",
            "url": opp.contract.url,
            "color": 0xF39C12,
            "description": opp.contract.title,
            "fields": [
                {
                    "name": "Leg 1",
                    "value": (
                        f"**BUY YES @ {price_cents}\u00a2**\n"
                        f"[{opp.contract.contract_id}]({leg1_url})"
                    ),
                    "inline": True,
                },
                {
                    "name": "Leg 2",
                    "value": (
                        f"**BUY NO @ {no_price_cents}\u00a2**\n"
                        f"[{opp.anchor_contract.contract_id}]({leg2_url})"
                    ),
                    "inline": True,
                },
                {
                    "name": "Locked Profit",
                    "value": f"**{profit_cents}\u00a2/pair**",
                    "inline": True,
                },
                {
                    "name": "Payoffs",
                    "value": (
                        f"```\n"
                        f"Cost    {cost_cents}\u00a2  "
                        f"({price_cents}\u00a2 + {no_price_cents}\u00a2)\n"
                        f"Worst  +{profit_cents}\u00a2  (guaranteed)\n"
                        f"Best  +{best_profit}\u00a2  (middle outcome)\n"
                        f"```"
                    ),
                    "inline": False,
                },
                {
                    "name": "Kelly Size",
                    "value": f"{opp.kelly_size:.1%}",
                    "inline": True,
                },
            ],
            "footer": {"text": "Monotonicity violation \u2014 acts fast, typically <1 min"},
        }
        return {"embeds": [embed]}
