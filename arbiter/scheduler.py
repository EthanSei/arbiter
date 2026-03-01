"""Pipeline orchestration — the core scan loop.

Uses a bare asyncio loop (not APScheduler) to avoid overlapping execution
bugs and unnecessary complexity. Supports dynamic cycle intervals.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arbiter.alerts.base import AlertChannel
from arbiter.ingestion.base import MarketClient
from arbiter.models.base import ProbabilityEstimator


class ScanPipeline:
    """Orchestrates the full scan cycle: ingest → estimate → score → dedup → alert → log.

    Each cycle:
    1. Fetch markets from all clients concurrently (asyncio.gather)
    2. Match contracts cross-platform
    3. Extract features + run estimator inference
    4. Compute EV, filter by threshold
    5. State-based deduplication (upsert by contract_id+direction)
    6. Persist Opportunity + send alerts + log AlertLog
    7. Snapshot market state for continuous training data
    """

    def __init__(
        self,
        clients: list[MarketClient],
        estimator: ProbabilityEstimator,
        channels: list[AlertChannel],
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        raise NotImplementedError  # Phase 6

    async def run_cycle(self) -> int:
        """Execute one full scan cycle. Returns the number of alerts sent."""
        raise NotImplementedError  # Phase 6

    async def run_forever(self, poll_interval: int = 300) -> None:
        """Run the scan loop indefinitely with dynamic interval."""
        raise NotImplementedError  # Phase 6
