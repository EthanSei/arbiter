"""Paper trader — simulated execution for strategy evaluation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arbiter.db.models import Direction, PaperTrade, Source
from arbiter.scoring.ev import ScoredOpportunity

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulated trader that records paper trades for strategy performance tracking.

    Position size is computed via fractional Kelly:
        quantity = kelly_size * kelly_fraction * bankroll / entry_price

    Trades are persisted to the ``paper_trades`` table.  Settlement computes
    P&L when a market resolves.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        initial_bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,
    ) -> None:
        self._session_factory = session_factory
        self._bankroll = initial_bankroll
        self._kelly_fraction = kelly_fraction

    async def execute(
        self, opportunity: ScoredOpportunity, session: AsyncSession | None = None
    ) -> None:
        """Record a paper trade.

        Position size = kelly_size * kelly_fraction * bankroll / entry_price.

        When *session* is provided the trade is added to that session (caller
        owns the commit).  Otherwise a standalone session is created and
        committed immediately.
        """
        entry_price = opportunity.market_price
        if entry_price <= 0:
            logger.warning("Skipping paper trade: entry_price=%s <= 0", entry_price)
            return

        quantity = opportunity.kelly_size * self._kelly_fraction * self._bankroll / entry_price

        trade = PaperTrade(
            source=Source(opportunity.contract.source),
            contract_id=opportunity.contract.contract_id,
            direction=Direction(opportunity.direction),
            strategy_name=opportunity.strategy_name,
            entry_price=entry_price,
            quantity=quantity,
            model_probability=opportunity.model_probability,
            expected_value=opportunity.expected_value,
        )

        if session is not None:
            session.add(trade)
        else:
            async with self._session_factory() as own_session:
                own_session.add(trade)
                await own_session.commit()

        logger.info(
            "Paper trade: %s %s %s qty=%.2f @ %.4f (EV=%.4f, kelly=%.4f)",
            opportunity.contract.contract_id,
            opportunity.direction,
            opportunity.strategy_name,
            quantity,
            entry_price,
            opportunity.expected_value,
            opportunity.kelly_size,
        )

    async def settle(self, contract_id: str, outcome: float) -> None:
        """Close open positions when a market resolves. Compute P&L.

        For YES direction: pnl = (outcome - entry_price) * quantity
        For NO direction:  pnl = ((1.0 - outcome) - entry_price) * quantity
        """
        async with self._session_factory() as session:
            stmt = select(PaperTrade).where(
                PaperTrade.contract_id == contract_id,
                PaperTrade.exited_at.is_(None),
            )
            result = await session.execute(stmt)
            trades = result.scalars().all()

            now = datetime.now(UTC)
            for trade in trades:
                if trade.direction == Direction.YES:
                    pnl = (outcome - trade.entry_price) * trade.quantity
                else:  # NO
                    pnl = ((1.0 - outcome) - trade.entry_price) * trade.quantity

                # Contract pays 1.0 on the correct side at settlement
                trade.exit_price = 1.0
                trade.exited_at = now
                trade.pnl = pnl
                trade.outcome = outcome

            await session.commit()

        logger.info(
            "Settled %d paper trade(s) for %s (outcome=%.2f)",
            len(trades),
            contract_id,
            outcome,
        )

    async def get_portfolio(self) -> dict[str, object]:
        """Return portfolio summary.

        Returns:
            dict with keys: bankroll, open_positions, realized_pnl, unrealized_pnl
        """
        async with self._session_factory() as session:
            # Open positions (not yet exited)
            open_stmt = select(PaperTrade).where(PaperTrade.exited_at.is_(None))
            open_result = await session.execute(open_stmt)
            open_trades = open_result.scalars().all()

            # Settled positions
            settled_stmt = select(PaperTrade).where(PaperTrade.exited_at.is_not(None))
            settled_result = await session.execute(settled_stmt)
            settled_trades = settled_result.scalars().all()

        realized_pnl = sum(t.pnl for t in settled_trades if t.pnl is not None)

        # Unrealized P&L: use model_probability as current estimate
        unrealized_pnl = 0.0
        for trade in open_trades:
            if trade.direction == Direction.YES:
                unrealized_pnl += (trade.model_probability - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl += (
                    (1.0 - trade.model_probability) - trade.entry_price
                ) * trade.quantity

        return {
            "bankroll": self._bankroll,
            "open_positions": len(open_trades),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
        }
