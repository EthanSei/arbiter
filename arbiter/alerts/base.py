"""Abstract base for alert delivery channels."""

from abc import ABC, abstractmethod

from arbiter.scoring.ev import ScoredOpportunity


class AlertChannel(ABC):
    """Delivers alerts about mispriced market opportunities.

    Implementations should degrade gracefully when credentials are missing
    (log a warning and skip, rather than raising).
    """

    @abstractmethod
    async def send(self, opportunity: ScoredOpportunity) -> None:
        """Send an alert about a scored opportunity."""
        ...

    def format_message(self, opp: ScoredOpportunity) -> str:
        """Default plain-text formatting for an opportunity alert."""
        return (
            f"[{opp.contract.source.upper()}] {opp.contract.title}\n"
            f"Direction: {opp.direction.upper()}\n"
            f"Market price: {opp.market_price:.1%}\n"
            f"Model probability: {opp.model_probability:.1%}\n"
            f"Expected value: {opp.expected_value:+.1%}\n"
            f"Kelly size: {opp.kelly_size:.1%}\n"
            f"Link: {opp.contract.url}"
        )
