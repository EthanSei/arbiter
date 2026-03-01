"""Polymarket API client for fetching prediction market data."""

from arbiter.ingestion.base import Contract, MarketClient


class PolymarketClient(MarketClient):
    """Fetches and normalizes market data from the Polymarket Gamma API.

    Uses offset-based pagination. Rate limited to 60 requests/minute.
    Prices are normalized from decimal strings to floats in [0, 1].
    """

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        raise NotImplementedError  # Phase 2

    async def close(self) -> None:
        raise NotImplementedError  # Phase 2
