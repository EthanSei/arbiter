"""SMS alert channel via Twilio REST API."""

from arbiter.alerts.base import AlertChannel
from arbiter.scoring.ev import ScoredOpportunity


class SMSChannel(AlertChannel):
    """Sends alerts via SMS using the Twilio REST API directly (httpx POST).

    No sync Twilio SDK — uses raw HTTP to stay fully async.
    Degrades gracefully if Twilio credentials are not configured.
    """

    async def send(self, opportunity: ScoredOpportunity) -> None:
        raise NotImplementedError  # Phase 5
