"""Base types for market data ingestion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Contract:
    """Normalized representation of a prediction market contract.

    All prices are floats in the range [0, 1] representing probabilities.
    Both Kalshi and Polymarket data normalize into this shape.
    """

    source: str  # "kalshi" or "polymarket"
    contract_id: str  # ticker (Kalshi) or condition_id (Polymarket)
    title: str
    category: str  # event category for feature engineering

    yes_price: float  # market-implied YES probability (midpoint)
    no_price: float  # market-implied NO probability (midpoint)
    yes_bid: float  # best bid for YES
    yes_ask: float  # best ask for YES
    last_price: float | None  # last traded price for YES
    volume_24h: float  # 24h volume in dollars
    open_interest: float  # open interest

    expires_at: datetime | None
    url: str
    status: str  # "open", "closed", etc.


class MarketClient(ABC):
    """Abstract base for prediction market API clients."""

    @abstractmethod
    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        """Fetch active markets and return normalized contracts."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up the underlying HTTP client."""
        ...
