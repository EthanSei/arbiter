"""Discord webhook alert channel."""

from arbiter.alerts.base import AlertChannel
from arbiter.scoring.ev import ScoredOpportunity


class DiscordChannel(AlertChannel):
    """Sends alerts via Discord webhook.

    Degrades gracefully if DISCORD_WEBHOOK_URL is not configured.
    """

    async def send(self, opportunity: ScoredOpportunity) -> None:
        raise NotImplementedError  # Phase 5
