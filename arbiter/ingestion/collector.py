"""Background data collector for order book snapshots."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arbiter.db.models import OrderBookSnapshot, Source
from arbiter.ingestion.base import Contract
from arbiter.ingestion.kalshi import KalshiClient

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects order book snapshots for the highest-volume contracts."""

    def __init__(
        self,
        kalshi_client: KalshiClient,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        top_n: int = 20,
    ) -> None:
        self._kalshi = kalshi_client
        self._session_factory = session_factory
        self._top_n = top_n

    async def collect_orderbooks(self, contracts: list[Contract]) -> int:
        """Fetch and persist order book snapshots for top-N Kalshi contracts by volume.

        Returns the number of snapshots successfully collected.
        """
        kalshi_contracts = [c for c in contracts if c.source == "kalshi"]
        top = sorted(kalshi_contracts, key=lambda c: c.volume_24h, reverse=True)[: self._top_n]

        collected = 0
        snapshots: list[OrderBookSnapshot] = []

        for contract in top:
            try:
                ob = await self._kalshi.fetch_orderbook(contract.contract_id)
            except Exception as exc:
                logger.warning("Order book fetch failed for %s: %s", contract.contract_id, exc)
                continue

            snapshots.append(
                OrderBookSnapshot(
                    source=Source.KALSHI,
                    contract_id=contract.contract_id,
                    series_ticker=contract.series_ticker,
                    event_ticker=contract.event_ticker,
                    bids=ob["bids"],
                    asks=ob["asks"],
                )
            )
            collected += 1

        if snapshots:
            async with self._session_factory() as session:
                session.add_all(snapshots)
                await session.commit()

        return collected
