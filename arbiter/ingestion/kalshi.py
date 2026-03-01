"""Kalshi API client for fetching prediction market data."""

from arbiter.ingestion.base import Contract, MarketClient


class KalshiClient(MarketClient):
    """Fetches and normalizes market data from the Kalshi API.

    Uses cursor-based pagination. Prices are normalized from dollar strings
    (or legacy cent integers) to floats in [0, 1].
    """

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        raise NotImplementedError  # Phase 2

    async def close(self) -> None:
        raise NotImplementedError  # Phase 2
