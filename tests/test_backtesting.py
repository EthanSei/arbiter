"""Tests for the backtesting framework — metrics and engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.backtesting.engine import BacktestEngine, BacktestResult
from arbiter.backtesting.metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate
from arbiter.db.models import Base, MarketSnapshot, Source
from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
from arbiter.scoring.strategy import EVStrategy, Strategy

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FixedEstimator(ProbabilityEstimator):
    """Returns a fixed probability for every contract."""

    def __init__(self, prob: float = 0.80) -> None:
        self._prob = prob

    async def estimate(self, contract: Contract) -> float:
        return self._prob


class _RecordingStrategy(Strategy):
    """Test double that records calls and delegates to EVStrategy."""

    def __init__(self, fee_rate: float = 0.0) -> None:
        self._inner = EVStrategy(fee_rate=fee_rate)
        self.called = 0

    @property
    def name(self) -> str:
        return "recording"

    async def score(self, contracts, estimator):
        self.called += 1
        return await self._inner.score(contracts, estimator)


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


def _make_snapshot(
    contract_id: str = "TEST-001",
    yes_price: float = 0.30,
    outcome: float | None = 1.0,
    snapshot_at: datetime | None = None,
    volume_24h: float = 1000.0,
) -> MarketSnapshot:
    """Create a MarketSnapshot with sensible defaults for testing."""
    if snapshot_at is None:
        snapshot_at = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    no_price = round(1.0 - yes_price, 6)
    return MarketSnapshot(
        source=Source.KALSHI,
        contract_id=contract_id,
        title=f"Test contract {contract_id}",
        category="test",
        features={
            "yes_price": yes_price,
            "no_price": no_price,
            "bid_ask_spread": 0.02,
            "last_price": yes_price,
            "log_volume_24h": 6.9,  # log1p(1000) ~ 6.9
            "log_open_interest": 6.2,
            "time_to_expiry_hours": 720.0,
            "day_of_week": 3.0,
            "hour_of_day": 12.0,
            "price_delta_1h": float("nan"),
            "price_delta_24h": float("nan"),
            "volume_ratio_24h": float("nan"),
            "price_volatility_24h": float("nan"),
        },
        feature_version="0.1.0",
        outcome=outcome,
        snapshot_at=snapshot_at,
    )


# ===========================================================================
# Metrics — pure function tests
# ===========================================================================


class TestSharpeRatio:
    def test_positive_returns(self) -> None:
        """Positive, non-zero returns should yield a positive Sharpe ratio."""
        result = sharpe_ratio([0.01, 0.02, 0.01, 0.03])
        assert result > 0

    def test_zero_returns(self) -> None:
        """Empty or all-zero returns should yield 0.0."""
        assert sharpe_ratio([]) == 0.0
        assert sharpe_ratio([0.0, 0.0, 0.0]) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        """Equity curve [100, 110, 90, 95, 80, 120] has drawdown 110->80 ~ 0.2727."""
        result = max_drawdown([100, 110, 90, 95, 80, 120])
        assert abs(result - 30 / 110) < 1e-6  # 30/110 ~ 0.2727

    def test_empty(self) -> None:
        """Empty equity curve should return 0.0."""
        assert max_drawdown([]) == 0.0


class TestWinRate:
    def test_mixed_pnls(self) -> None:
        """[10, -5, 20, -3] has 2 wins out of 4 = 0.5."""
        assert win_rate([10, -5, 20, -3]) == 0.5

    def test_empty(self) -> None:
        """Empty P&L list should return 0.0."""
        assert win_rate([]) == 0.0


class TestProfitFactor:
    def test_mixed_pnls(self) -> None:
        """[10, -5, 20, -3]: profits=30, losses=8 -> 3.75."""
        result = profit_factor([10, -5, 20, -3])
        assert abs(result - 3.75) < 1e-6

    def test_no_losses(self) -> None:
        """All-positive P&L should return inf."""
        result = profit_factor([10, 20, 5])
        assert result == float("inf")

    def test_no_profits(self) -> None:
        """All-negative P&L should return 0.0."""
        result = profit_factor([-10, -20])
        assert result == 0.0

    def test_empty(self) -> None:
        """Empty P&L should return 0.0."""
        assert profit_factor([]) == 0.0


# ===========================================================================
# BacktestEngine
# ===========================================================================


class TestBacktestEngine:
    async def test_empty_snapshots_zero_trades(self, db_factory) -> None:
        """No snapshots in DB should produce a BacktestResult with num_trades=0."""
        engine = BacktestEngine(
            strategies=[EVStrategy(fee_rate=0.0)],
            estimator=_FixedEstimator(0.80),
            ev_threshold=0.05,
            fee_rate=0.0,
        )
        async with db_factory() as session:
            result = await engine.run(session)
        assert isinstance(result, BacktestResult)
        assert result.num_trades == 0
        assert result.total_pnl == 0.0

    async def test_single_resolved_opportunity(self, db_factory) -> None:
        """One snapshot with outcome=1.0 and a high estimator probability should
        produce at least one trade with positive P&L.

        Setup:
        - Snapshot: yes_price=0.30, outcome=1.0 (YES resolved)
        - Estimator returns 0.80
        - EVStrategy fee_rate=0.0
        - EV = 0.80 - 0.30 = 0.50 >> threshold 0.05
        - Trade: buy YES at 0.30, resolves YES -> P&L = 1.0 - 0.30 = 0.70 per unit
        """
        snapshot = _make_snapshot(
            contract_id="WINNER-001",
            yes_price=0.30,
            outcome=1.0,
            snapshot_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        async with db_factory() as session:
            session.add(snapshot)
            await session.commit()

        engine = BacktestEngine(
            strategies=[EVStrategy(fee_rate=0.0)],
            estimator=_FixedEstimator(0.80),
            ev_threshold=0.05,
            fee_rate=0.0,
        )
        async with db_factory() as session:
            result = await engine.run(session)

        assert result.num_trades >= 1
        assert result.total_pnl > 0

    async def test_uses_strategy_interface(self, db_factory) -> None:
        """A custom Strategy test double should be called during backtesting."""
        snapshot = _make_snapshot(
            contract_id="STRAT-001",
            yes_price=0.30,
            outcome=1.0,
            snapshot_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        async with db_factory() as session:
            session.add(snapshot)
            await session.commit()

        recording = _RecordingStrategy(fee_rate=0.0)
        engine = BacktestEngine(
            strategies=[recording],
            estimator=_FixedEstimator(0.80),
            ev_threshold=0.05,
            fee_rate=0.0,
        )
        async with db_factory() as session:
            await engine.run(session)

        assert recording.called >= 1
