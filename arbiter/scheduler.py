"""Pipeline orchestration — the core scan loop.

Uses a bare asyncio loop (not APScheduler) to avoid overlapping execution
bugs and unnecessary complexity. Supports dynamic cycle intervals.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arbiter.alerts.base import AlertChannel
from arbiter.db.models import AlertLog, Direction, MarketSnapshot, Opportunity, Source
from arbiter.ingestion.base import Contract, MarketClient
from arbiter.models.base import ProbabilityEstimator
from arbiter.models.features import FEATURE_VERSION, SPEC, extract_features
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.fees import FeeFn
from arbiter.scoring.strategy import ConsistencyStrategy, Strategy, YesOnlyEVStrategy
from arbiter.trading.paper import PaperTrader

logger = logging.getLogger(__name__)


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
        ev_threshold: float = 0.05,
        fee_rate: float = 0.01,
        kelly_fraction: float = 1.0,
        strategies: list[Strategy] | None = None,
        paper_trader: PaperTrader | None = None,
        fee_fn: FeeFn | None = None,
    ) -> None:
        self._clients = clients
        self._estimator = estimator
        self._channels = channels
        self._session_factory = session_factory
        self._ev_threshold = ev_threshold
        self._fee_rate = fee_rate
        self._kelly_fraction = kelly_fraction
        self._paper_trader = paper_trader
        self._fee_fn = fee_fn
        self._strategies = (
            strategies
            if strategies is not None
            else [
                YesOnlyEVStrategy(fee_rate, fee_fn=fee_fn),
                ConsistencyStrategy(fee_rate, fee_fn=fee_fn),
            ]
        )
        self._last_markets_scanned = 0
        self._alerted_keys: set[tuple[str, str]] = set()  # in-memory dedup

    async def _fetch_safe(self, client: MarketClient) -> list[Contract]:
        """Fetch from one client; return [] on any failure."""
        try:
            return await client.fetch_markets()
        except Exception as exc:
            logger.warning("Market fetch failed (%s): %s", type(client).__name__, exc)
            return []

    async def run_cycle(self) -> int:
        """Execute one full scan cycle. Returns the number of alerts sent."""
        t_start = asyncio.get_running_loop().time()

        # 1. Fetch markets from all clients concurrently
        batches: list[list[Contract]] = list(
            await asyncio.gather(*[self._fetch_safe(c) for c in self._clients])
        )
        all_contracts: list[Contract] = [c for batch in batches for c in batch]
        self._last_markets_scanned = len(all_contracts)
        t_fetch = asyncio.get_running_loop().time()
        logger.info(
            "Contracts fetched: %s total=%d (%.1fs)",
            " ".join(
                f"{type(c).__name__}={len(b)}" for c, b in zip(self._clients, batches, strict=True)
            ),
            len(all_contracts),
            t_fetch - t_start,
        )

        # 2. Run all strategies on the full contract batch
        all_scored: list[ScoredOpportunity] = []
        for strategy in self._strategies:
            try:
                scored = await strategy.score(all_contracts, self._estimator)
                all_scored.extend(scored)
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", strategy.name, exc)

        # Apply fractional Kelly scaling (e.g., 0.25 for quarter-Kelly)
        if self._kelly_fraction != 1.0:
            for opp in all_scored:
                opp.kelly_size *= self._kelly_fraction

        # 3. Group by (contract_id, direction), take max EV per key
        best_by_key: dict[tuple[str, str], ScoredOpportunity] = {}
        for opp in all_scored:
            key = (opp.contract.contract_id, opp.direction)
            if key not in best_by_key or opp.expected_value > best_by_key[key].expected_value:
                best_by_key[key] = opp

        # 4–6. Threshold filter, dedup, alert, deactivate, snapshot
        alerts_sent = 0
        async with self._session_factory() as session:
            # Pre-fetch active opportunity keys in one query so _deactivate can
            # be skipped for the vast majority of contracts with no prior opportunity.
            active_result = await session.execute(
                select(Opportunity.contract_id, Opportunity.direction).where(
                    Opportunity.active.is_(True)
                )
            )
            active_opps: set[tuple[str, str]] = {
                (row.contract_id, row.direction) for row in active_result
            }

            # Collect which (contract_id, direction) keys are above threshold
            above_keys: set[tuple[str, str]] = set()
            for key, opp in best_by_key.items():
                if opp.expected_value >= self._ev_threshold:
                    alerted = await self._upsert_and_alert(session, opp)
                    alerts_sent += alerted
                    active_opps.add(key)
                    above_keys.add(key)

                    if self._paper_trader is not None:
                        try:
                            await self._paper_trader.execute(opp, session=session)
                        except Exception as exc:
                            logger.warning("Paper trade failed: %s", exc)

            # Deactivate per-contract: for each fetched contract, any direction
            # not above threshold should be deactivated if previously active.
            seen_contracts = {c.contract_id for c in all_contracts}
            for contract_id in seen_contracts:
                for direction in ("yes", "no"):
                    key = (contract_id, direction)
                    if key not in above_keys and key in active_opps:
                        await self._deactivate(session, contract_id, direction)
                        active_opps.discard(key)

            # Snapshot for continuous training data
            for contract in all_contracts:
                await self._snapshot(session, contract)

            await session.commit()

        t_end = asyncio.get_running_loop().time()
        logger.info(
            "Cycle complete: contracts=%d alerts=%d fetch=%.1fs score+db=%.1fs total=%.1fs",
            len(all_contracts),
            alerts_sent,
            t_fetch - t_start,
            t_end - t_fetch,
            t_end - t_start,
        )
        return alerts_sent

    async def _upsert_and_alert(self, session: AsyncSession, opp: ScoredOpportunity) -> int:
        """Upsert Opportunity and alert if new or reactivated. Returns 1 if alerted."""
        stmt = select(Opportunity).where(
            Opportunity.contract_id == opp.contract.contract_id,
            Opportunity.direction == opp.direction,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        now = datetime.now(UTC)
        should_alert = False

        if existing is None:
            db_opp = Opportunity(
                source=Source(opp.contract.source),
                contract_id=opp.contract.contract_id,
                title=opp.contract.title,
                direction=Direction(opp.direction),
                strategy_name=opp.strategy_name,
                market_price=opp.market_price,
                model_probability=opp.model_probability,
                expected_value=opp.expected_value,
                kelly_size=opp.kelly_size,
                expires_at=opp.contract.expires_at,
                active=True,
                last_seen_at=now,
            )
            session.add(db_opp)
            should_alert = True
        else:
            was_inactive = not existing.active
            existing.strategy_name = opp.strategy_name
            existing.market_price = opp.market_price
            existing.model_probability = opp.model_probability
            existing.expected_value = opp.expected_value
            existing.kelly_size = opp.kelly_size
            existing.last_seen_at = now
            existing.active = True
            db_opp = existing
            should_alert = was_inactive  # Re-alert only on reactivation

        alert_key = (opp.contract.contract_id, opp.direction)
        if should_alert and alert_key not in self._alerted_keys:
            # Flush so db_opp.id is stable before writing AlertLog rows
            await session.flush()
            await self._send_alerts(session, db_opp, opp, now)
            self._alerted_keys.add(alert_key)
            return 1
        return 0

    async def _deactivate(self, session: AsyncSession, contract_id: str, direction: str) -> None:
        stmt = select(Opportunity).where(
            Opportunity.contract_id == contract_id,
            Opportunity.direction == Direction(direction),
            Opportunity.active.is_(True),
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.active = False
            self._alerted_keys.discard((contract_id, direction))

    async def _send_alerts(
        self,
        session: AsyncSession,
        db_opp: Opportunity,
        opp: ScoredOpportunity,
        now: datetime,
    ) -> None:
        """Send to each channel independently; log every attempt."""
        any_succeeded = False
        for channel in self._channels:
            success = True
            error_msg: str | None = None
            try:
                await channel.send(opp)
                any_succeeded = True
            except Exception as exc:
                success = False
                error_msg = str(exc)
                logger.warning("Alert send failed (%s): %s", type(channel).__name__, exc)

            log = AlertLog(
                opportunity_id=db_opp.id,
                channel=type(channel).__name__,
                success=success,
                error_message=error_msg,
            )
            session.add(log)

        # Only stamp last_alerted_at when at least one channel delivered the alert.
        # If all channels fail (or none are configured), leaving it unset allows
        # retry once channels are available.
        if any_succeeded:
            db_opp.last_alerted_at = now

    async def _snapshot(self, session: AsyncSession, contract: Contract) -> None:
        """Persist a MarketSnapshot for continuous training data collection."""
        features_arr = extract_features(contract)
        features_json: dict[str, object] = {
            name: (None if isinstance(v, float) and math.isnan(v) else float(v))
            for name, v in zip(SPEC.names, features_arr, strict=True)
        }
        snapshot = MarketSnapshot(
            source=Source(contract.source),
            contract_id=contract.contract_id,
            title=contract.title,
            category=contract.category,
            features=features_json,
            feature_version=FEATURE_VERSION,
        )
        session.add(snapshot)

    async def run_forever(self, poll_interval: int = 300) -> None:
        """Run the scan loop indefinitely with dynamic interval."""
        from arbiter.health import update_health

        logger.info("ScanPipeline starting (poll_interval=%ds)", poll_interval)
        while True:
            errors: list[str] = []
            t0 = asyncio.get_running_loop().time()
            try:
                alerts = await self.run_cycle()
                elapsed = asyncio.get_running_loop().time() - t0
                logger.info(
                    "Cycle complete: alerts=%d markets=%d elapsed=%.1fs sleeping=%ds",
                    alerts,
                    self._last_markets_scanned,
                    elapsed,
                    poll_interval,
                )
                update_health(
                    status="healthy",
                    last_cycle_completed_at=datetime.now(UTC),
                    last_cycle_duration_seconds=elapsed,
                    markets_scanned=self._last_markets_scanned,
                    opportunities_found=alerts,
                    errors=errors,
                )
            except Exception as exc:
                logger.exception("Scan cycle error: %s", exc)
                errors.append(str(exc))
                update_health(status="degraded", errors=errors)

            elapsed = asyncio.get_running_loop().time() - t0
            await asyncio.sleep(max(0.0, poll_interval - elapsed))
