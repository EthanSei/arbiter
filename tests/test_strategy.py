"""Tests for scoring strategies and pipeline integration."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.alerts.base import AlertChannel
from arbiter.data.providers.base import FeatureSet
from arbiter.db.models import Base
from arbiter.ingestion.base import Contract, MarketClient
from arbiter.models.base import ProbabilityEstimator
from arbiter.scheduler import ScanPipeline
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.strategy import (
    AnchorStrategy,
    CategoryRouter,
    ConsistencyStrategy,
    EVStrategy,
    IndicatorRouter,
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
    category: str = "",
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
        """Pipeline with no strategies param fires at least one alert."""
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

    def test_default_strategies_use_yes_only_ev(self) -> None:
        """Pipeline default is YesOnlyEVStrategy + ConsistencyStrategy (YES-side only)."""
        pipeline = ScanPipeline(
            clients=[],
            estimator=_FixedEstimator(),
            channels=[],
            session_factory=None,  # type: ignore[arg-type]
            ev_threshold=0.05,
            fee_rate=0.01,
        )
        assert isinstance(pipeline._strategies[0], YesOnlyEVStrategy)
        assert isinstance(pipeline._strategies[1], ConsistencyStrategy)

    async def test_kelly_fraction_scales_kelly_size(self, db_factory) -> None:
        """kelly_fraction param multiplies kelly_size on all scored opportunities."""
        channel = _RecordingChannel()
        pipeline = ScanPipeline(
            clients=[_OkClient([_contract(yes_price=0.30)])],
            estimator=_FixedEstimator(0.70),
            channels=[channel],
            session_factory=db_factory,
            ev_threshold=0.05,
            fee_rate=0.0,
            kelly_fraction=0.25,
            strategies=[_AlwaysAlertStrategy()],  # kelly_size=0.30 fixed
        )
        await pipeline.run_cycle()
        assert len(channel.sent) == 1
        # Full Kelly from _AlwaysAlertStrategy is 0.30; quarter-Kelly → 0.075
        assert channel.sent[0].kelly_size == pytest.approx(0.075)


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
    def test_without_anchor_returns_flat_strategies(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(fee_rate=0.01)
        assert len(strategies) == 2
        assert isinstance(strategies[0], YesOnlyEVStrategy)
        assert isinstance(strategies[1], ConsistencyStrategy)

    def test_with_anchor_returns_indicator_router(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.01,
            anchor_providers=[_MockProvider()],
        )
        assert len(strategies) == 1
        assert isinstance(strategies[0], IndicatorRouter)

    def test_include_ev_false_returns_only_consistency(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(fee_rate=0.01, include_ev=False)
        assert len(strategies) == 1
        assert isinstance(strategies[0], ConsistencyStrategy)

    def test_include_ev_false_with_anchor_returns_indicator_router(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.01,
            include_ev=False,
            anchor_providers=[_MockProvider()],
        )
        assert len(strategies) == 1
        assert isinstance(strategies[0], IndicatorRouter)

    async def test_router_excludes_ev_from_indicator_contracts(self) -> None:
        """Indicator contracts (KXCPI) go through Anchor, not EV."""
        from arbiter.scoring.strategy import build_default_strategies

        mock_provider = _MockProvider()
        strategies = build_default_strategies(
            fee_rate=0.0,
            anchor_providers=[mock_provider],
        )
        router = strategies[0]
        # KXCPI contract with empty category (matching production)
        contracts = [_contract(contract_id="KXCPI-26JAN-T0.3", yes_price=0.30)]
        results = await router.score(contracts, _FixedEstimator(0.70))
        strategy_names = {r.strategy_name for r in results}
        assert "YesOnlyEVStrategy" not in strategy_names

    async def test_router_uses_ev_for_non_indicator_contracts(self) -> None:
        """Non-indicator contracts go through EV, not Anchor."""
        from arbiter.scoring.strategy import build_default_strategies

        mock_provider = _MockProvider()
        strategies = build_default_strategies(
            fee_rate=0.0,
            anchor_providers=[mock_provider],
        )
        router = strategies[0]
        # Non-indicator contract (empty category, matching production)
        contracts = [_contract(contract_id="SOMEOTHER-26JAN", yes_price=0.30)]
        results = await router.score(contracts, _FixedEstimator(0.70))
        strategy_names = {r.strategy_name for r in results}
        assert "YesOnlyEVStrategy" in strategy_names
        assert "AnchorStrategy" not in strategy_names

    async def test_include_ev_false_drops_non_indicator_contracts(self) -> None:
        """When model is uncalibrated, non-indicator contracts get no strategy."""
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.0,
            include_ev=False,
            anchor_providers=[_MockProvider()],
        )
        router = strategies[0]
        contracts = [_contract(contract_id="SOMEOTHER-26JAN", yes_price=0.30)]
        results = await router.score(contracts, _FixedEstimator(0.70))
        assert results == []

    async def test_empty_category_contracts_still_route_by_ticker(self) -> None:
        """Regression: contracts with category='' must still route correctly."""
        from arbiter.scoring.strategy import build_default_strategies

        fred = _MockProvider(
            name="fred",
            feature_sets={
                "KXCPI": FeatureSet(
                    provider="fred",
                    indicator_id="KXCPI",
                    anchor_mu=0.003,
                    anchor_sigma=0.0015,
                )
            },
        )
        strategies = build_default_strategies(
            fee_rate=0.0,
            anchor_providers=[fred],
        )
        router = strategies[0]
        # Empty category, like real Kalshi data
        contracts = _anchor_contracts(prefix="KXCPI-26JAN", prices=[0.60, 0.40, 0.08])
        results = await router.score(contracts, _FixedEstimator(0.50))
        assert any(r.strategy_name == "AnchorStrategy" for r in results)


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


class TestIndicatorRouter:
    async def test_routes_indicator_contracts_to_indicator_strategies(self) -> None:
        indicator_strategy = _RecordingStrategy()
        default_strategy = _RecordingStrategy()
        router = IndicatorRouter(
            indicator_strategies=[indicator_strategy],
            default=[default_strategy],
        )
        cpi_c = _contract(contract_id="KXCPI-26JAN-T0.3", yes_price=0.30)
        other_c = _contract(contract_id="SOMEOTHER-26JAN", yes_price=0.30)
        await router.score([cpi_c, other_c], _FixedEstimator())
        assert cpi_c in indicator_strategy.received
        assert other_c not in indicator_strategy.received
        assert other_c in default_strategy.received
        assert cpi_c not in default_strategy.received

    async def test_routes_all_known_indicators(self) -> None:
        indicator_strategy = _RecordingStrategy()
        router = IndicatorRouter(indicator_strategies=[indicator_strategy])
        contracts = [
            _contract(contract_id="KXCPI-26JAN-T0.3"),
            _contract(contract_id="KXPAYROLLS-26MAR-T100000"),
            _contract(contract_id="KXJOBLESSCLAIMS-26MAR06-T220"),
            _contract(contract_id="KXCPIYOY-26FEB-T2.5"),
            _contract(contract_id="KXCPICOREYOY-26FEB-T3.0"),
        ]
        await router.score(contracts, _FixedEstimator())
        assert len(indicator_strategy.received) == 5

    async def test_empty_category_does_not_break_routing(self) -> None:
        """Regression: contracts with category='' (Kalshi default) route by ticker."""
        indicator_strategy = _RecordingStrategy()
        default_strategy = _RecordingStrategy()
        router = IndicatorRouter(
            indicator_strategies=[indicator_strategy],
            default=[default_strategy],
        )
        # category="" matches production Kalshi data
        cpi_c = _contract(contract_id="KXCPI-26JAN-T0.3", category="")
        other_c = _contract(contract_id="GENERIC-001", category="")
        await router.score([cpi_c, other_c], _FixedEstimator())
        assert cpi_c in indicator_strategy.received
        assert other_c in default_strategy.received

    async def test_empty_contracts(self) -> None:
        router = IndicatorRouter(
            indicator_strategies=[_RecordingStrategy()],
            default=[_RecordingStrategy()],
        )
        results = await router.score([], _FixedEstimator())
        assert results == []

    async def test_no_default_drops_non_indicator(self) -> None:
        indicator_strategy = _RecordingStrategy()
        router = IndicatorRouter(indicator_strategies=[indicator_strategy])
        other_c = _contract(contract_id="GENERIC-001")
        results = await router.score([other_c], _FixedEstimator())
        assert results == []
        assert len(indicator_strategy.received) == 0

    async def test_is_strategy_subclass(self) -> None:
        router = IndicatorRouter(indicator_strategies=[])
        assert isinstance(router, Strategy)


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
        router = CategoryRouter(routes={"economics": [EVStrategy(fee_rate=0.01)]})
        assert isinstance(router, Strategy)
        results = await router.score(
            [_contract(yes_price=0.30, category="economics")], _FixedEstimator(0.70)
        )
        assert len(results) > 0
        assert any(r.expected_value > 0 for r in results)


# ---------------------------------------------------------------------------
# AnchorStrategy
# ---------------------------------------------------------------------------


class _DummyCalibrator:
    """Test double calibrator that returns inputs unchanged."""

    def predict(self, values: list[float]) -> list[float]:
        return values


class _MockProvider:
    """Test double FeatureProvider that returns a fixed FeatureSet."""

    def __init__(
        self,
        name: str = "mock",
        feature_sets: dict[str, FeatureSet] | None = None,
    ) -> None:
        self._name = name
        self._feature_sets = feature_sets or {}

    @property
    def name(self) -> str:
        return self._name

    def load(self, indicator_id: str) -> FeatureSet | None:
        return self._feature_sets.get(indicator_id)


def _anchor_contracts(
    prefix: str = "KXCPI-26JAN",
    thresholds: list[float] | None = None,
    prices: list[float] | None = None,
) -> list[Contract]:
    """Create a group of T-suffix contracts for anchor testing."""
    thresholds = thresholds or [0.002, 0.003, 0.004]
    prices = prices or [0.60, 0.40, 0.08]
    return [
        _contract(
            contract_id=f"{prefix}-T{t}",
            yes_price=p,
            volume_24h=100.0,
        )
        for t, p in zip(thresholds, prices, strict=True)
    ]


class TestAnchorStrategy:
    def _fred_provider(
        self, indicator_id: str = "KXCPI", mu: float = 0.003, sigma: float = 0.0015
    ) -> _MockProvider:
        return _MockProvider(
            name="fred",
            feature_sets={
                indicator_id: FeatureSet(
                    provider="fred",
                    indicator_id=indicator_id,
                    anchor_mu=mu,
                    anchor_sigma=sigma,
                )
            },
        )

    def _bls_provider(self, indicator_id: str = "KXCPI") -> _MockProvider:
        return _MockProvider(
            name="bls",
            feature_sets={
                indicator_id: FeatureSet(
                    provider="bls",
                    indicator_id=indicator_id,
                    features={"shelter_cpi_mom": 0.40, "energy_cpi_mom": -1.2},
                )
            },
        )

    async def test_returns_opportunities_for_underpriced_contracts(self) -> None:
        # mu=0.003, sigma=0.0015 → P(X > 0.004) ≈ 25%, market at 8¢ → EV > 0
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        contracts = _anchor_contracts()
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) > 0
        assert all(r.expected_value > 0 for r in results)
        assert all(r.direction == "yes" for r in results)

    async def test_returns_empty_for_fairly_priced_markets(self) -> None:
        # KXCPI uses threshold_scale=0.01: Kalshi T0.2/0.3/0.4 (pct points)
        # → FRED 0.002/0.003/0.004. With mu=0.003, sigma=0.0015:
        # P(X>0.002)≈75%, P(X>0.003)≈50%, P(X>0.004)≈25%
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        contracts = _anchor_contracts(thresholds=[0.2, 0.3, 0.4], prices=[0.75, 0.50, 0.25])
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 0

    async def test_ignores_contracts_without_matching_indicator(self) -> None:
        # Provider only has KXCPI data, but contracts are KXFOO
        strategy = AnchorStrategy(
            providers=[self._fred_provider(indicator_id="KXCPI")], fee_rate=0.01
        )
        contracts = _anchor_contracts(prefix="KXFOO-26JAN")
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 0

    async def test_ignores_non_t_suffix_contracts(self) -> None:
        # MAXMON-style contracts should pass through without error
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        contracts = [
            _contract(contract_id="KXBTCMAXMON-BTC-26MAR31-80000", yes_price=0.30),
        ]
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 0

    async def test_aggregates_from_multiple_providers(self) -> None:
        # FRED provides mu/sigma, BLS provides only features (no anchor params)
        # Should still work — uses FRED's mu/sigma
        strategy = AnchorStrategy(
            providers=[self._fred_provider(), self._bls_provider()], fee_rate=0.01
        )
        contracts = _anchor_contracts()
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) > 0

    async def test_skips_provider_returning_none(self) -> None:
        # Second provider returns None for this indicator — should still work
        empty_provider = _MockProvider(name="empty")
        strategy = AnchorStrategy(providers=[self._fred_provider(), empty_provider], fee_rate=0.01)
        contracts = _anchor_contracts()
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) > 0

    async def test_strategy_name_is_anchor_strategy(self) -> None:
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        contracts = _anchor_contracts()
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert all(r.strategy_name == "AnchorStrategy" for r in results)

    async def test_is_strategy_subclass(self) -> None:
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        assert isinstance(strategy, Strategy)

    async def test_empty_contracts(self) -> None:
        strategy = AnchorStrategy(providers=[self._fred_provider()], fee_rate=0.01)
        results = await strategy.score([], _FixedEstimator(0.50))
        assert results == []

    async def test_no_providers(self) -> None:
        strategy = AnchorStrategy(providers=[], fee_rate=0.01)
        contracts = _anchor_contracts()
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert results == []

    async def test_multiple_indicator_groups(self) -> None:
        # Mix of KXCPI and KXJOBLESSCLAIMS contracts
        fred = _MockProvider(
            name="fred",
            feature_sets={
                "KXCPI": FeatureSet(
                    provider="fred",
                    indicator_id="KXCPI",
                    anchor_mu=0.003,
                    anchor_sigma=0.0015,
                ),
                "KXJOBLESSCLAIMS": FeatureSet(
                    provider="fred",
                    indicator_id="KXJOBLESSCLAIMS",
                    anchor_mu=220.0,
                    anchor_sigma=10.0,
                ),
            },
        )
        cpi_contracts = _anchor_contracts(prefix="KXCPI-26JAN", prices=[0.60, 0.40, 0.08])
        claims_contracts = [
            _contract(contract_id="KXJOBLESSCLAIMS-26MAR06-T210", yes_price=0.05, volume_24h=50),
        ]
        strategy = AnchorStrategy(providers=[fred], fee_rate=0.01)
        results = await strategy.score(cpi_contracts + claims_contracts, _FixedEstimator(0.50))
        # Should have results from both groups
        contract_ids = {r.contract.contract_id for r in results}
        assert any("KXCPI" in cid for cid in contract_ids)
        assert any("KXJOBLESSCLAIMS" in cid for cid in contract_ids)

    async def test_strategy_name_uses_subclass_name(self) -> None:
        """AnchorStrategy sets strategy_name via self.name, enabling subclassing."""

        class _CustomAnchor(AnchorStrategy):
            pass

        strategy = _CustomAnchor(providers=[self._fred_provider()], fee_rate=0.0)
        contracts = _anchor_contracts(prices=[0.60, 0.40, 0.08])
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) > 0
        assert all(r.strategy_name == "_CustomAnchor" for r in results)

    async def test_skips_group_when_provider_returns_zero_sigma(self) -> None:
        """AnchorStrategy returns empty when sigma=0 (avoids ValueError crash)."""
        zero_sigma_provider = _MockProvider(
            name="zero_sigma",
            feature_sets={
                "KXCPI": FeatureSet(
                    provider="zero_sigma",
                    indicator_id="KXCPI",
                    anchor_mu=0.003,
                    anchor_sigma=0.0,
                )
            },
        )
        strategy = AnchorStrategy(providers=[zero_sigma_provider], fee_rate=0.01)
        contracts = _anchor_contracts(prices=[0.60, 0.40, 0.08])
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert results == []

    async def test_threshold_scale_applied_from_indicator_registry(self) -> None:
        # KXCPIYOY has threshold_scale=0.01 in the registry.
        # Contracts use percentage-point thresholds (e.g. T2.5 = 2.5%).
        # Provider supplies FRED decimal data: mu=0.030, sigma=0.0035.
        # With scale: 2.5 * 0.01 = 0.025 → well below mu=0.030 → anchor_prob ≈ 0.92.
        # Without scale: threshold=2.5 >> mu=0.030 → anchor_prob ≈ 0 → no opportunity.
        fred = _MockProvider(
            name="fred",
            feature_sets={
                "KXCPIYOY": FeatureSet(
                    provider="fred",
                    indicator_id="KXCPIYOY",
                    anchor_mu=0.030,
                    anchor_sigma=0.0035,
                )
            },
        )
        contracts = _anchor_contracts(
            prefix="KXCPIYOY-26FEB",
            thresholds=[2.5],
            prices=[0.09],  # well underpriced vs expected ~92%
        )
        strategy = AnchorStrategy(providers=[fred], fee_rate=0.01)
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 1
        assert results[0].model_probability == pytest.approx(0.924, abs=0.01)

    def test_build_default_strategies_with_calibrators_path(self, tmp_path) -> None:
        """build_default_strategies loads calibrators from a pickle file."""
        import pickle

        from arbiter.scoring.strategy import build_default_strategies

        cal_path = tmp_path / "cal.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"KXCPI": _DummyCalibrator()}, f)

        strategies = build_default_strategies(
            anchor_providers=[self._fred_provider()],
            calibrators_path=str(cal_path),
        )
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    async def test_build_default_strategies_with_anchor(self) -> None:
        from arbiter.scoring.strategy import build_default_strategies

        strategies = build_default_strategies(
            fee_rate=0.01,
            anchor_providers=[self._fred_provider()],
        )
        assert len(strategies) == 1
        assert isinstance(strategies[0], IndicatorRouter)

    async def test_calibrators_dict_adjusts_model_probability(self) -> None:
        """AnchorStrategy applies the per-indicator calibrator to anchor_prob."""

        class _HalfCalibrator:
            def predict(self, vals: list[float]) -> list[float]:
                return [v * 0.5 for v in vals]

        strategy = AnchorStrategy(
            providers=[self._fred_provider()],
            fee_rate=0.01,
            calibrators={"KXCPI": _HalfCalibrator()},
        )
        # KXCPI has threshold_scale=0.01: T0.4 → effective 0.004 → raw_prob ≈ 0.252
        # calibrated = 0.126, market=0.08, ev = 0.036 > 0 → opportunity
        contracts = _anchor_contracts(thresholds=[0.4], prices=[0.08])
        results = await strategy.score(contracts, _FixedEstimator(0.50))
        assert len(results) == 1
        assert results[0].model_probability == pytest.approx(0.126, abs=0.01)

    async def test_calibrators_dict_missing_indicator_uses_raw_prob(self) -> None:
        """Indicator not in calibrators dict falls back to raw anchor_prob."""

        class _HalfCalibrator:
            def predict(self, vals: list[float]) -> list[float]:
                return [v * 0.5 for v in vals]

        strategy_with = AnchorStrategy(
            providers=[self._fred_provider()],
            fee_rate=0.01,
            calibrators={"KXCPI": _HalfCalibrator()},
        )
        strategy_without = AnchorStrategy(
            providers=[self._fred_provider()],
            fee_rate=0.01,
        )
        # KXCPI has threshold_scale=0.01: T0.4 → effective 0.004 → raw_prob ≈ 0.252
        contracts = _anchor_contracts(thresholds=[0.4], prices=[0.08])
        results_with = await strategy_with.score(contracts, _FixedEstimator(0.50))
        results_without = await strategy_without.score(contracts, _FixedEstimator(0.50))
        # calibrated (≈0.126) < raw (≈0.252)
        assert results_with[0].model_probability < results_without[0].model_probability

    async def test_calibrators_none_leaves_behavior_unchanged(self) -> None:
        """calibrators=None is identical to not passing calibrators."""
        strategy_explicit_none = AnchorStrategy(
            providers=[self._fred_provider()],
            fee_rate=0.01,
            calibrators=None,
        )
        strategy_default = AnchorStrategy(
            providers=[self._fred_provider()],
            fee_rate=0.01,
        )
        contracts = _anchor_contracts()
        results_none = await strategy_explicit_none.score(contracts, _FixedEstimator(0.50))
        results_default = await strategy_default.score(contracts, _FixedEstimator(0.50))
        assert len(results_none) == len(results_default)
        for r_n, r_d in zip(results_none, results_default, strict=True):
            assert r_n.model_probability == pytest.approx(r_d.model_probability)
