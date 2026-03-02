"""Tests for Strategy ABC, EVStrategy, ConsistencyStrategy, and pipeline integration."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.alerts.base import AlertChannel
from arbiter.db.models import Base
from arbiter.ingestion.base import Contract, MarketClient
from arbiter.models.base import ProbabilityEstimator
from arbiter.scheduler import ScanPipeline
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.strategy import (
    CategoryRouter,
    ConsistencyStrategy,
    EVStrategy,
    Strategy,
    YesOnlyEVStrategy,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _contract(
    source: str = "kalshi",
    contract_id: str = "TEST-001",
    yes_price: float = 0.40,
    no_price: float = 0.60,
    category: str = "test",
    volume_24h: float = 1000.0,
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title="Will X happen?",
        category=category,
        yes_price=yes_price,
        no_price=no_price,
        yes_bid=yes_price - 0.01,
        yes_ask=yes_price + 0.01,
        last_price=yes_price,
        volume_24h=volume_24h,
        open_interest=500.0,
        expires_at=None,
        url="https://example.com",
        status="open",
    )


class _FixedEstimator(ProbabilityEstimator):
    def __init__(self, prob: float = 0.70) -> None:
        self._prob = prob

    async def estimate(self, contract: Contract) -> float:
        return self._prob


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class TestStrategyABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_name_defaults_to_class_name(self) -> None:
        strategy = EVStrategy(fee_rate=0.01)
        assert strategy.name == "EVStrategy"


# ---------------------------------------------------------------------------
# EVStrategy
# ---------------------------------------------------------------------------


class TestEVStrategy:
    async def test_returns_both_directions(self) -> None:
        strategy = EVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.70))
        directions = {r.direction for r in results}
        assert directions == {"yes", "no"}

    async def test_uses_estimator_probability(self) -> None:
        strategy = EVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.80))
        yes_opp = next(r for r in results if r.direction == "yes")
        assert yes_opp.model_probability == 0.80

    async def test_fee_rate_applied(self) -> None:
        no_fee = EVStrategy(fee_rate=0.0)
        with_fee = EVStrategy(fee_rate=0.05)
        r_no_fee = await no_fee.score([_contract()], _FixedEstimator(0.70))
        r_with_fee = await with_fee.score([_contract()], _FixedEstimator(0.70))
        yes_no_fee = next(r for r in r_no_fee if r.direction == "yes")
        yes_with_fee = next(r for r in r_with_fee if r.direction == "yes")
        assert yes_with_fee.expected_value < yes_no_fee.expected_value

    async def test_multiple_contracts(self) -> None:
        strategy = EVStrategy(fee_rate=0.0)
        contracts = [_contract(contract_id=f"C-{i}") for i in range(3)]
        results = await strategy.score(contracts, _FixedEstimator(0.70))
        # 3 contracts × 2 directions = 6
        assert len(results) == 6

    async def test_empty_contracts(self) -> None:
        strategy = EVStrategy(fee_rate=0.0)
        results = await strategy.score([], _FixedEstimator(0.70))
        assert results == []

    async def test_strategy_name_set(self) -> None:
        strategy = EVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.70))
        assert all(r.strategy_name == "EVStrategy" for r in results)


# ---------------------------------------------------------------------------
# ConsistencyStrategy
# ---------------------------------------------------------------------------


def _maxmon_group() -> list[Contract]:
    """Two MAXMON contracts where the higher threshold has a higher price (violation)."""
    return [
        _contract(
            contract_id="KXBTCMAXMON-BTC-26MAR31-80000",
            yes_price=0.30,
            volume_24h=100.0,
        ),
        _contract(
            contract_id="KXBTCMAXMON-BTC-26MAR31-90000",
            yes_price=0.50,
            volume_24h=100.0,
        ),
    ]


class TestConsistencyStrategy:
    async def test_detects_violation(self) -> None:
        strategy = ConsistencyStrategy(fee_rate=0.0)
        results = await strategy.score(_maxmon_group(), _FixedEstimator(0.50))
        assert len(results) > 0
        assert all(r.direction == "yes" for r in results)

    async def test_no_violation(self) -> None:
        # Prices monotonically decreasing with higher threshold — no violation
        contracts = [
            _contract(
                contract_id="KXBTCMAXMON-BTC-26MAR31-80000",
                yes_price=0.60,
                volume_24h=100.0,
            ),
            _contract(
                contract_id="KXBTCMAXMON-BTC-26MAR31-90000",
                yes_price=0.30,
                volume_24h=100.0,
            ),
        ]
        strategy = ConsistencyStrategy(fee_rate=0.0)
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 0

    async def test_ignores_estimator(self) -> None:
        strategy = ConsistencyStrategy(fee_rate=0.0)
        r1 = await strategy.score(_maxmon_group(), _FixedEstimator(0.01))
        r2 = await strategy.score(_maxmon_group(), _FixedEstimator(0.99))
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2, strict=True):
            assert a.expected_value == b.expected_value

    async def test_strategy_name_set(self) -> None:
        strategy = ConsistencyStrategy(fee_rate=0.0)
        results = await strategy.score(_maxmon_group(), _FixedEstimator(0.50))
        assert all(r.strategy_name == "ConsistencyStrategy" for r in results)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class _OkClient(MarketClient):
    def __init__(self, contracts: list[Contract]) -> None:
        self._contracts = contracts

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        return self._contracts

    async def close(self) -> None:
        pass


class _RecordingChannel(AlertChannel):
    def __init__(self) -> None:
        self.sent: list[ScoredOpportunity] = []

    async def send(self, opp: ScoredOpportunity) -> None:
        self.sent.append(opp)

    async def close(self) -> None:
        pass


class _AlwaysAlertStrategy(Strategy):
    """Test double: returns a fixed positive-EV opportunity for every contract."""

    async def score(self, contracts, estimator):
        return [
            ScoredOpportunity(
                contract=c,
                direction="yes",
                market_price=c.yes_price,
                model_probability=0.99,
                expected_value=0.50,
                kelly_size=0.30,
                strategy_name=self.name,
            )
            for c in contracts
        ]


@pytest.fixture
async def db_factory():
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


class TestPipelineWithStrategies:
    async def test_explicit_ev_only_skips_consistency(self, db_factory) -> None:
        """When only EVStrategy is passed, consistency violations are not detected."""
        channel = _RecordingChannel()
        pipeline = ScanPipeline(
            clients=[_OkClient(_maxmon_group())],
            estimator=_FixedEstimator(0.50),
            channels=[channel],
            session_factory=db_factory,
            ev_threshold=0.05,
            fee_rate=0.0,
            strategies=[EVStrategy(fee_rate=0.0)],
        )
        await pipeline.run_cycle()
        # EVStrategy at prob=0.50 with yes_price=0.30 → EV=0.20 for the first contract
        # But no consistency violations should be detected
        strategy_names = {opp.strategy_name for opp in channel.sent}
        assert "ConsistencyStrategy" not in strategy_names

    async def test_custom_strategy_flows_through(self, db_factory) -> None:
        """A custom strategy's opportunities flow through the pipeline."""
        channel = _RecordingChannel()
        pipeline = ScanPipeline(
            clients=[_OkClient([_contract()])],
            estimator=_FixedEstimator(0.50),
            channels=[channel],
            session_factory=db_factory,
            ev_threshold=0.05,
            fee_rate=0.0,
            strategies=[_AlwaysAlertStrategy()],
        )
        alerts = await pipeline.run_cycle()
        assert alerts == 1
        assert channel.sent[0].strategy_name == "_AlwaysAlertStrategy"

    async def test_default_strategies_backward_compat(self, db_factory) -> None:
        """Pipeline with no strategies param uses EVStrategy + ConsistencyStrategy."""
        channel = _RecordingChannel()
        pipeline = ScanPipeline(
            clients=[_OkClient([_contract(yes_price=0.30)])],
            estimator=_FixedEstimator(0.70),
            channels=[channel],
            session_factory=db_factory,
            ev_threshold=0.05,
            fee_rate=0.0,
        )
        alerts = await pipeline.run_cycle()
        assert alerts >= 1


# ---------------------------------------------------------------------------
# CategoryRouter
# ---------------------------------------------------------------------------


class _RecordingStrategy(Strategy):
    def __init__(self) -> None:
        self.received: list[Contract] = []

    async def score(self, contracts, estimator):
        self.received.extend(contracts)
        return []


class TestBuildDefaultStrategies:
    def test_without_target_series_returns_flat_strategies(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(fee_rate=0.01)
        assert len(strategies) == 2
        assert isinstance(strategies[0], YesOnlyEVStrategy)
        assert isinstance(strategies[1], ConsistencyStrategy)

    def test_with_target_series_returns_category_router(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.01,
            target_categories=["Economics", "Financials"],
        )
        assert len(strategies) == 1
        assert isinstance(strategies[0], CategoryRouter)

    async def test_router_routes_economics_to_both_strategies(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.0,
            target_categories=["Economics"],
        )
        router = strategies[0]
        contracts = [_contract(yes_price=0.30, category="Economics")]
        results = await router.score(contracts, _FixedEstimator(0.70))
        # YesOnlyEVStrategy produces yes direction only
        assert len(results) >= 1
        assert any(r.strategy_name == "YesOnlyEVStrategy" for r in results)

    async def test_router_uses_ev_only_as_default(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.0,
            target_categories=["Economics"],
        )
        router = strategies[0]
        contracts = [_contract(yes_price=0.30, category="Other")]
        results = await router.score(contracts, _FixedEstimator(0.70))
        strategy_names = {r.strategy_name for r in results}
        assert "YesOnlyEVStrategy" in strategy_names
        assert "ConsistencyStrategy" not in strategy_names


# ---------------------------------------------------------------------------
# YesOnlyEVStrategy
# ---------------------------------------------------------------------------


class TestYesOnlyEVStrategy:
    async def test_returns_only_yes_direction(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        strategy = YesOnlyEVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.70))
        assert len(results) == 1
        assert all(r.direction == "yes" for r in results)

    async def test_strategy_name(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        strategy = YesOnlyEVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.70))
        assert all(r.strategy_name == "YesOnlyEVStrategy" for r in results)

    async def test_uses_estimator_probability(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        strategy = YesOnlyEVStrategy(fee_rate=0.0)
        results = await strategy.score([_contract()], _FixedEstimator(0.80))
        assert results[0].model_probability == 0.80

    async def test_fee_rate_applied(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        no_fee = YesOnlyEVStrategy(fee_rate=0.0)
        with_fee = YesOnlyEVStrategy(fee_rate=0.05)
        r_no_fee = await no_fee.score([_contract()], _FixedEstimator(0.70))
        r_with_fee = await with_fee.score([_contract()], _FixedEstimator(0.70))
        assert r_with_fee[0].expected_value < r_no_fee[0].expected_value

    async def test_multiple_contracts(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        strategy = YesOnlyEVStrategy(fee_rate=0.0)
        contracts = [_contract(contract_id=f"C-{i}") for i in range(3)]
        results = await strategy.score(contracts, _FixedEstimator(0.70))
        # 3 contracts × 1 direction (YES only) = 3
        assert len(results) == 3
        assert all(r.direction == "yes" for r in results)

    async def test_empty_contracts(self) -> None:
        from arbiter.scoring.strategy import YesOnlyEVStrategy

        strategy = YesOnlyEVStrategy(fee_rate=0.0)
        results = await strategy.score([], _FixedEstimator(0.70))
        assert results == []


# ---------------------------------------------------------------------------
# CategoryRouter
# ---------------------------------------------------------------------------


class TestCategoryRouter:
    async def test_routes_by_category(self) -> None:
        crypto_strategy = _RecordingStrategy()
        sports_strategy = _RecordingStrategy()
        router = CategoryRouter(
            routes={"crypto": [crypto_strategy], "sports": [sports_strategy]},
        )
        crypto_c = _contract(contract_id="BTC-001", category="crypto")
        sports_c = _contract(contract_id="NBA-001", category="sports")
        await router.score([crypto_c, sports_c], _FixedEstimator())
        assert crypto_c in crypto_strategy.received
        assert sports_c not in crypto_strategy.received
        assert sports_c in sports_strategy.received
        assert crypto_c not in sports_strategy.received

    async def test_default_fallback(self) -> None:
        default_strategy = _RecordingStrategy()
        crypto_strategy = _RecordingStrategy()
        router = CategoryRouter(
            routes={"crypto": [crypto_strategy]},
            default=[default_strategy],
        )
        unknown_c = _contract(contract_id="MYSTERY-001", category="unknown")
        crypto_c = _contract(contract_id="BTC-001", category="crypto")
        await router.score([unknown_c, crypto_c], _FixedEstimator())
        assert unknown_c in default_strategy.received
        assert crypto_c not in default_strategy.received
        assert crypto_c in crypto_strategy.received

    async def test_case_insensitive(self) -> None:
        crypto_strategy = _RecordingStrategy()
        router = CategoryRouter(routes={"crypto": [crypto_strategy]})
        upper_c = _contract(contract_id="BTC-001", category="Crypto")
        lower_c = _contract(contract_id="ETH-001", category="crypto")
        await router.score([upper_c, lower_c], _FixedEstimator())
        assert upper_c in crypto_strategy.received
        assert lower_c in crypto_strategy.received
        assert len(crypto_strategy.received) == 2

    async def test_empty_contracts(self) -> None:
        router = CategoryRouter(
            routes={"crypto": [_RecordingStrategy()]},
            default=[_RecordingStrategy()],
        )
        results = await router.score([], _FixedEstimator())
        assert results == []

    async def test_composable_with_pipeline(self, db_factory) -> None:
        router = CategoryRouter(routes={"test": [EVStrategy(fee_rate=0.01)]})
        assert isinstance(router, Strategy)
        results = await router.score(
            [_contract(yes_price=0.30, category="test")], _FixedEstimator(0.70)
        )
        assert len(results) > 0
        assert any(r.expected_value > 0 for r in results)
