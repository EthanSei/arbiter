"""Backtesting engine for evaluating strategies on historical market data."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from arbiter.db.models import MarketSnapshot
from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
from arbiter.scoring.kelly import kelly_criterion
from arbiter.scoring.strategy import Strategy


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    trades: list[dict[str, Any]] = field(default_factory=list)
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)


class BacktestEngine:
    """Runs strategies against historical MarketSnapshot data to evaluate performance.

    The engine replays resolved market snapshots chronologically, runs strategies
    on each batch, simulates trades for above-threshold opportunities, and computes
    performance metrics.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        estimator: ProbabilityEstimator,
        ev_threshold: float = 0.05,
        fee_rate: float = 0.01,
        initial_bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,
    ) -> None:
        self._strategies = strategies
        self._estimator = estimator
        self._ev_threshold = ev_threshold
        self._fee_rate = fee_rate
        self._initial_bankroll = initial_bankroll
        self._kelly_fraction = kelly_fraction

    async def run(
        self,
        session: AsyncSession,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        category: str | None = None,
    ) -> BacktestResult:
        """Run the backtest on resolved MarketSnapshot data.

        1. Query MarketSnapshot rows WHERE outcome IS NOT NULL, ordered by snapshot_at
        2. Group snapshots by snapshot_at timestamp (each group = one "cycle")
        3. For each cycle:
           a. Reconstruct Contract objects from snapshot fields
           b. Run strategies on the contract batch
           c. Filter by ev_threshold
           d. For above-threshold: simulate entry at snapshot's yes_price
           e. Track open positions
        4. For resolved contracts (outcome != None): settle positions, compute P&L
        5. Compute metrics: Sharpe, drawdown, win rate
        6. Return BacktestResult
        """
        # Step 1: Query resolved snapshots
        stmt = (
            select(MarketSnapshot)
            .where(MarketSnapshot.outcome.isnot(None))
            .order_by(MarketSnapshot.snapshot_at)
        )
        if start_date is not None:
            stmt = stmt.where(MarketSnapshot.snapshot_at >= start_date)
        if end_date is not None:
            stmt = stmt.where(MarketSnapshot.snapshot_at <= end_date)
        if category is not None:
            stmt = stmt.where(MarketSnapshot.category == category)

        result = await session.execute(stmt)
        snapshots = list(result.scalars().all())

        if not snapshots:
            return BacktestResult()

        # Step 2: Group snapshots by snapshot_at
        cycles: dict[datetime, list[MarketSnapshot]] = defaultdict(list)
        for snap in snapshots:
            cycles[snap.snapshot_at].append(snap)

        # Step 3-4: Process each cycle
        bankroll = self._initial_bankroll
        trades: list[dict[str, Any]] = []
        equity_curve: list[tuple[datetime, float]] = [
            (snapshots[0].snapshot_at, bankroll),
        ]

        for cycle_time in sorted(cycles.keys()):
            cycle_snapshots = cycles[cycle_time]

            # Reconstruct contracts
            contracts = [self._snapshot_to_contract(s) for s in cycle_snapshots]

            # Build outcome lookup: contract_id -> outcome
            outcome_map: dict[str, float] = {}
            for snap in cycle_snapshots:
                if snap.outcome is not None:
                    outcome_map[snap.contract_id] = snap.outcome

            # Run all strategies
            for strategy in self._strategies:
                scored = await strategy.score(contracts, self._estimator)

                # Filter by EV threshold
                above_threshold = [s for s in scored if s.expected_value >= self._ev_threshold]

                for opp in above_threshold:
                    cid = opp.contract.contract_id
                    outcome = outcome_map.get(cid)
                    if outcome is None:
                        continue  # cannot settle, skip

                    # Position sizing via fractional Kelly
                    entry_price = opp.market_price
                    if entry_price <= 0 or entry_price >= 1:
                        continue

                    payout_ratio = (1.0 / entry_price) - 1.0
                    full_kelly = kelly_criterion(opp.model_probability, payout_ratio)
                    bet_fraction = full_kelly * self._kelly_fraction
                    bet_size = bankroll * bet_fraction

                    if bet_size <= 0:
                        continue

                    # Settle: compute P&L based on direction and outcome
                    if opp.direction == "yes":
                        # Bought YES at entry_price, pays $1 if YES
                        pnl = bet_size * ((1.0 / entry_price - 1.0) if outcome >= 0.5 else -1.0)
                    else:
                        # Bought NO at entry_price (= 1 - yes_price), pays $1 if NO
                        pnl = bet_size * ((1.0 / entry_price - 1.0) if outcome < 0.5 else -1.0)

                    # Subtract fee from pnl
                    pnl -= bet_size * self._fee_rate

                    bankroll += pnl

                    trade = {
                        "contract_id": cid,
                        "direction": opp.direction,
                        "entry_price": entry_price,
                        "model_probability": opp.model_probability,
                        "expected_value": opp.expected_value,
                        "bet_size": bet_size,
                        "pnl": pnl,
                        "outcome": outcome,
                        "strategy": strategy.name,
                        "timestamp": cycle_time,
                    }
                    trades.append(trade)

            equity_curve.append((cycle_time, bankroll))

        # Step 5: Compute metrics
        pnls = [t["pnl"] for t in trades]
        returns = [t["pnl"] / t["bet_size"] for t in trades if t["bet_size"] > 0]
        equity_values = [e[1] for e in equity_curve]

        return BacktestResult(
            trades=trades,
            total_pnl=sum(pnls) if pnls else 0.0,
            sharpe_ratio=sharpe_ratio(returns),
            max_drawdown=max_drawdown(equity_values),
            win_rate=win_rate(pnls),
            num_trades=len(trades),
            equity_curve=equity_curve,
        )

    def _snapshot_to_contract(self, snapshot: MarketSnapshot) -> Contract:
        """Reconstruct a Contract from a MarketSnapshot.

        Uses snapshot.features dict for price/volume fields.
        Features dict has keys: yes_price, no_price, bid_ask_spread, last_price,
        log_volume_24h, log_open_interest, time_to_expiry_hours, etc.
        """
        features: dict[str, Any] = snapshot.features or {}

        yes_price = float(features.get("yes_price", 0.5))
        no_price = float(features.get("no_price", 0.5))
        bid_ask_spread = float(features.get("bid_ask_spread", 0.02))
        last_price = features.get("last_price")
        if last_price is not None:
            last_price = float(last_price)

        # Reconstruct bid/ask from midpoint and spread
        half_spread = bid_ask_spread / 2.0
        yes_bid = max(0.0, yes_price - half_spread)
        yes_ask = min(1.0, yes_price + half_spread)

        # Recover volume from log transform: log_volume = log1p(volume) -> volume = expm1(log_vol)
        log_volume = features.get("log_volume_24h")
        volume_24h = math.expm1(float(log_volume)) if log_volume is not None else 0.0

        log_oi = features.get("log_open_interest")
        open_interest = math.expm1(float(log_oi)) if log_oi is not None else 0.0

        # Reconstruct expiry from time_to_expiry_hours and snapshot_at
        time_to_expiry = features.get("time_to_expiry_hours")
        expires_at = None
        if time_to_expiry is not None and not math.isnan(float(time_to_expiry)):
            from datetime import timedelta

            expires_at = snapshot.snapshot_at + timedelta(hours=float(time_to_expiry))

        return Contract(
            source=snapshot.source.value if hasattr(snapshot.source, "value") else snapshot.source,
            contract_id=snapshot.contract_id,
            title=snapshot.title,
            category=snapshot.category,
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            last_price=last_price,
            volume_24h=volume_24h,
            open_interest=open_interest,
            expires_at=expires_at,
            url="",
            status="resolved",
        )
