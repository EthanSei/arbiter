"""Data collection for model training — historical scraping and live snapshots."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.ingestion.base import MarketClient


async def collect_resolved_markets(client: MarketClient) -> list[dict[str, object]]:
    """Fetch historically resolved/settled markets for training data bootstrap.

    Kalshi: status=settled. Polymarket: closed=true.
    """
    raise NotImplementedError  # Phase 4


async def snapshot_live_markets(
    client: MarketClient,
    session: AsyncSession,
    price_change_threshold: float = 0.01,
) -> int:
    """Snapshot current live market data into MarketSnapshot table.

    Only snapshots markets whose price has changed by more than the threshold
    since the last snapshot, to avoid redundant training data.

    Returns the number of snapshots written.
    """
    raise NotImplementedError  # Phase 4


async def backfill_outcomes(session: AsyncSession) -> int:
    """Update MarketSnapshot.outcome for markets that have since resolved.

    Returns the number of snapshots updated.
    """
    raise NotImplementedError  # Phase 4
