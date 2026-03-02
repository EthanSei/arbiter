"""Tests for PaperTrader — paper trading execution and settlement."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.db.models import Base, PaperTrade
from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _contract(
    source: str = "kalshi",
    contract_id: str = "TEST-001",
    yes_price: float = 0.40,
    no_price: float = 0.60,
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title="Will X happen?",
        category="test",
        yes_price=yes_price,
        no_price=no_price,
        yes_bid=yes_price - 0.01,
        yes_ask=yes_price + 0.01,
        last_price=yes_price,
        volume_24h=1000.0,
        open_interest=500.0,
        expires_at=None,
        url="https://example.com",
        status="open",
    )


def _scored_opportunity(
    contract: Contract | None = None,
    direction: str = "yes",
    market_price: float = 0.40,
    model_probability: float = 0.70,
    expected_value: float = 0.29,
    kelly_size: float = 0.30,
    strategy_name: str = "ev_threshold",
) -> ScoredOpportunity:
    """Build a ScoredOpportunity and attach strategy_name for paper trading."""
    if contract is None:
        contract = _contract(yes_price=market_price, no_price=round(1.0 - market_price, 6))
    return ScoredOpportunity(
        contract=contract,
        direction=direction,
        market_price=market_price,
        model_probability=model_probability,
        expected_value=expected_value,
        kelly_size=kelly_size,
        strategy_name=strategy_name,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_factory():
    """In-memory SQLite session factory for tests."""
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# test_paper_trade_recorded — execute creates a PaperTrade row
# ---------------------------------------------------------------------------


async def test_paper_trade_recorded(db_factory) -> None:
    """Executing a paper trade creates a PaperTrade row in the database."""
    from arbiter.trading.paper import PaperTrader

    trader = PaperTrader(session_factory=db_factory)
    opp = _scored_opportunity()
    await trader.execute(opp)

    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 1
    assert rows[0].contract_id == "TEST-001"
    assert rows[0].source.value == "kalshi"
    assert rows[0].direction.value == "yes"
    assert rows[0].entry_price == opp.market_price
    assert rows[0].model_probability == opp.model_probability
    assert rows[0].expected_value == opp.expected_value
    assert rows[0].exit_price is None
    assert rows[0].pnl is None


# ---------------------------------------------------------------------------
# test_position_sizing_uses_kelly
# ---------------------------------------------------------------------------


async def test_position_sizing_uses_kelly(db_factory) -> None:
    """Quantity = kelly_size * kelly_fraction * bankroll / entry_price."""
    from arbiter.trading.paper import PaperTrader

    bankroll = 10_000.0
    kelly_fraction = 0.25
    trader = PaperTrader(
        session_factory=db_factory,
        initial_bankroll=bankroll,
        kelly_fraction=kelly_fraction,
    )

    kelly_size = 0.30
    entry_price = 0.40
    opp = _scored_opportunity(market_price=entry_price, kelly_size=kelly_size)
    await trader.execute(opp)

    expected_quantity = kelly_size * kelly_fraction * bankroll / entry_price

    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 1
    assert rows[0].quantity == pytest.approx(expected_quantity)


# ---------------------------------------------------------------------------
# test_settle_computes_pnl_yes_outcome — YES direction, outcome=1.0
# ---------------------------------------------------------------------------


async def test_settle_computes_pnl_yes_outcome(db_factory) -> None:
    """Settling a YES trade with outcome=1.0 computes positive P&L.

    For YES direction: pnl = (outcome * 1.0 - entry_price) * quantity
    """
    from arbiter.trading.paper import PaperTrader

    trader = PaperTrader(session_factory=db_factory)
    opp = _scored_opportunity(direction="yes", market_price=0.40)
    await trader.execute(opp)

    await trader.settle("TEST-001", outcome=1.0)

    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 1
    trade = rows[0]
    assert trade.outcome == 1.0
    assert trade.exit_price == 1.0
    assert trade.exited_at is not None
    # pnl = (1.0 - 0.40) * quantity
    expected_pnl = (1.0 - trade.entry_price) * trade.quantity
    assert trade.pnl == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# test_settle_computes_pnl_no_outcome — NO direction, outcome=0.0
# ---------------------------------------------------------------------------


async def test_settle_computes_pnl_no_outcome(db_factory) -> None:
    """Settling a NO trade with outcome=0.0 computes positive P&L.

    For NO direction: pnl = ((1.0 - outcome) * 1.0 - entry_price) * quantity
    When outcome=0.0: pnl = (1.0 - entry_price) * quantity
    """
    from arbiter.trading.paper import PaperTrader

    trader = PaperTrader(session_factory=db_factory)
    opp = _scored_opportunity(direction="no", market_price=0.60)
    await trader.execute(opp)

    await trader.settle("TEST-001", outcome=0.0)

    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 1
    trade = rows[0]
    assert trade.outcome == 0.0
    assert trade.exit_price == 1.0  # NO pays 1.0 when outcome is 0
    assert trade.exited_at is not None
    # pnl = ((1.0 - 0.0) - 0.60) * quantity = 0.40 * quantity
    expected_pnl = (1.0 - trade.entry_price) * trade.quantity
    assert trade.pnl == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# test_portfolio_summary
# ---------------------------------------------------------------------------


async def test_portfolio_summary(db_factory) -> None:
    """get_portfolio returns bankroll, open positions count, realized P&L."""
    from arbiter.trading.paper import PaperTrader

    trader = PaperTrader(session_factory=db_factory, initial_bankroll=10_000.0)

    # Create two trades
    opp1 = _scored_opportunity(
        contract=_contract(contract_id="P-001", yes_price=0.40, no_price=0.60),
        direction="yes",
        market_price=0.40,
    )
    opp2 = _scored_opportunity(
        contract=_contract(contract_id="P-002", yes_price=0.30, no_price=0.70),
        direction="yes",
        market_price=0.30,
    )
    await trader.execute(opp1)
    await trader.execute(opp2)

    # Settle one trade
    await trader.settle("P-001", outcome=1.0)

    portfolio = await trader.get_portfolio()
    assert portfolio["bankroll"] == 10_000.0
    assert portfolio["open_positions"] == 1  # P-002 still open
    assert portfolio["realized_pnl"] > 0  # P-001 settled profitably
    assert "unrealized_pnl" in portfolio


# ---------------------------------------------------------------------------
# test_no_paper_trader_skips — pipeline with paper_trader=None (backward compat)
# ---------------------------------------------------------------------------


async def test_no_paper_trader_skips(db_factory) -> None:
    """Pipeline with paper_trader=None works normally — no paper trades created."""
    from arbiter.alerts.base import AlertChannel
    from arbiter.ingestion.base import MarketClient
    from arbiter.models.base import ProbabilityEstimator
    from arbiter.scheduler import ScanPipeline

    class _OkClient(MarketClient):
        async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
            return [_contract(yes_price=0.30, no_price=0.70)]

        async def close(self) -> None:
            pass

    class _FixedEstimator(ProbabilityEstimator):
        async def estimate(self, contract: Contract) -> float:
            return 0.70

    class _RecordingChannel(AlertChannel):
        def __init__(self) -> None:
            self.sent: list[ScoredOpportunity] = []

        async def send(self, opp: ScoredOpportunity) -> None:
            self.sent.append(opp)

        async def close(self) -> None:
            pass

    channel = _RecordingChannel()
    pipeline = ScanPipeline(
        clients=[_OkClient()],
        estimator=_FixedEstimator(),
        channels=[channel],
        session_factory=db_factory,
        ev_threshold=0.05,
        fee_rate=0.01,
    )
    alerts = await pipeline.run_cycle()
    assert alerts >= 1  # pipeline works normally

    # No PaperTrade rows created
    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# test_strategy_name_recorded
# ---------------------------------------------------------------------------


async def test_strategy_name_recorded(db_factory) -> None:
    """strategy_name from ScoredOpportunity is stored on PaperTrade."""
    from arbiter.trading.paper import PaperTrader

    trader = PaperTrader(session_factory=db_factory)
    opp = _scored_opportunity(strategy_name="consistency_arb")
    await trader.execute(opp)

    async with db_factory() as session:
        rows = (await session.execute(select(PaperTrade))).scalars().all()
    assert len(rows) == 1
    assert rows[0].strategy_name == "consistency_arb"
