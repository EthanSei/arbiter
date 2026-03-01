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

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (e.g. httpx client)."""
        ...
