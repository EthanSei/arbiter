"""Abstract base for scoring strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from arbiter.data.indicators import INDICATORS
from arbiter.data.providers.base import FeatureProvider
from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
from arbiter.scoring.anchor import Calibrator, find_anchor_mispricings, group_anchor_contracts
from arbiter.scoring.consistency import find_consistency_violations
from arbiter.scoring.ev import ScoredOpportunity, compute_ev
from arbiter.scoring.fees import FeeFn

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Scores contracts and returns opportunities.

    Strategies receive the full contract batch and return scored opportunities.
    The pipeline handles threshold filtering, dedup, alerting, and persistence.
    """

    @abstractmethod
    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        """Score contracts and return all opportunities (no threshold filtering).

        Args:
            contracts: All contracts from the current fetch cycle.
            estimator: The probability estimator for model-based strategies.

        Returns:
            All ScoredOpportunity results, including negative-EV ones.
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


class EVStrategy(Strategy):
    """Per-contract expected value scoring using model probability estimates."""

    def __init__(
        self,
        fee_rate: float = 0.01,
        fee_fn: FeeFn | None = None,
    ) -> None:
        self._fee_rate = fee_rate
        self._fee_fn = fee_fn

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        results: list[ScoredOpportunity] = []
        for contract in contracts:
            model_prob = await estimator.estimate(contract)
            scored = compute_ev(contract, model_prob, fee_rate=self._fee_rate, fee_fn=self._fee_fn)
            for opp in scored:
                opp.strategy_name = self.name
            results.extend(scored)
        return results


class YesOnlyEVStrategy(EVStrategy):
    """EVStrategy that only trades the YES side.

    Backtest analysis showed the model has genuine edge on YES-side trades
    (48.2% win rate vs 41.1% breakeven) but is anti-calibrated on NO-side
    trades (46.8% win rate vs 59.7% breakeven). This strategy filters to
    YES-direction opportunities only.
    """

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        results = await super().score(contracts, estimator)
        yes_only = [r for r in results if r.direction == "yes"]
        for opp in yes_only:
            opp.strategy_name = self.name
        return yes_only


class ConsistencyStrategy(Strategy):
    """Kalshi range-market internal consistency scoring.

    Detects stochastic dominance violations across MAXMON/MINMON contract families.
    Ignores the estimator — uses sibling prices as the probability floor.
    """

    def __init__(
        self,
        fee_rate: float = 0.01,
        fee_fn: FeeFn | None = None,
    ) -> None:
        self._fee_rate = fee_rate
        self._fee_fn = fee_fn

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        results = find_consistency_violations(
            contracts, fee_rate=self._fee_rate, fee_fn=self._fee_fn
        )
        for opp in results:
            opp.strategy_name = self.name
        return results


class AnchorStrategy(Strategy):
    """Prices contracts using external probability anchors from feature providers.

    For each economic release series:
    1. Load FeatureSets from all registered providers for that indicator
    2. Aggregate μ (consensus) and σ (surprise volatility) from providers
    3. Compute P(X > K) for each threshold
    4. Flag contracts where anchor_prob > market_price + fee
    """

    def __init__(
        self,
        providers: list[FeatureProvider],
        fee_rate: float = 0.01,
        calibrators: dict[str, Calibrator] | None = None,
        fee_fn: FeeFn | None = None,
    ) -> None:
        self._providers = providers
        self._fee_rate = fee_rate
        self._calibrators = calibrators or {}
        self._fee_fn = fee_fn

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        groups = group_anchor_contracts(contracts)
        if not groups:
            return []

        results: list[ScoredOpportunity] = []
        for group_key, group in groups.items():
            indicator_id = _extract_indicator_id(group_key)
            if indicator_id is None:
                continue

            mu, sigma = self._aggregate_anchor_params(indicator_id)
            if mu is None or sigma is None or sigma <= 0:
                continue

            config = INDICATORS.get(indicator_id)
            threshold_scale = config.threshold_scale if config is not None else 1.0
            calibrator = self._calibrators.get(indicator_id)
            opps = find_anchor_mispricings(
                group,
                mu,
                sigma,
                self._fee_rate,
                threshold_scale,
                calibrator,
                fee_fn=self._fee_fn,
            )
            for opp in opps:
                opp.strategy_name = self.name
            results.extend(opps)

        return results

    def _aggregate_anchor_params(self, indicator_id: str) -> tuple[float | None, float | None]:
        """Load FeatureSets from all providers and return the first (μ, σ) pair.

        Uses the first provider that supplies both anchor_mu and anchor_sigma,
        ensuring μ and σ come from the same data source.
        """
        for provider in self._providers:
            fs = provider.load(indicator_id)
            if fs is None:
                continue
            if fs.anchor_mu is not None and fs.anchor_sigma is not None:
                return fs.anchor_mu, fs.anchor_sigma

        return None, None


def _extract_indicator_id(group_key: str) -> str | None:
    """Extract the Kalshi indicator series from a group key.

    'KXCPI-26JAN' → 'KXCPI'
    'KXJOBLESSCLAIMS-26MAR06' → 'KXJOBLESSCLAIMS'
    """
    parts = group_key.split("-")
    if not parts:
        return None
    return parts[0]


def build_default_strategies(
    fee_rate: float = 0.01,
    anchor_providers: list[FeatureProvider] | None = None,
    include_ev: bool = True,
    calibrators_path: str | None = None,
    fee_fn: FeeFn | None = None,
) -> list[Strategy]:
    """Build the default strategy list.

    Uses YesOnlyEVStrategy (YES-side only) as the EV strategy because
    backtest analysis showed the model is anti-calibrated on NO-side trades.
    Pass include_ev=False to omit it when the model is not calibrated —
    raw uncalibrated probabilities produce too many false signals.

    When anchor_providers are given, returns an IndicatorRouter that routes
    contracts matching the INDICATORS registry to [Consistency + Anchor] and
    all others to [EV] (or nothing when uncalibrated).

    Without anchor_providers, returns the flat list [EV? + Consistency].

    Args:
        fee_rate: Legacy flat fee rate. Ignored when fee_fn is provided.
        fee_fn: Callable(price, is_taker) → fee. Overrides fee_rate when set.
    """
    ev: list[Strategy] = [YesOnlyEVStrategy(fee_rate, fee_fn=fee_fn)] if include_ev else []
    calibrators: dict[str, Calibrator] | None = None
    if calibrators_path is not None:
        import pickle

        with open(calibrators_path, "rb") as _f:
            calibrators = pickle.load(_f)  # noqa: S301
    anchor: list[Strategy] = (
        [AnchorStrategy(anchor_providers, fee_rate, calibrators, fee_fn=fee_fn)]
        if anchor_providers
        else []
    )

    if not anchor:
        return ev + [ConsistencyStrategy(fee_rate, fee_fn=fee_fn)]

    indicator_strategies = [ConsistencyStrategy(fee_rate, fee_fn=fee_fn)] + anchor
    default = ev
    router = IndicatorRouter(indicator_strategies=indicator_strategies, default=default)
    return [router]


class IndicatorRouter(Strategy):
    """Routes contracts by ticker prefix: known indicators → anchor strategies, rest → default.

    Splits contracts based on whether their ticker prefix (e.g. ``KXCPI`` from
    ``KXCPI-26JAN-T0.4``) matches a key in the ``INDICATORS`` registry. This
    replaces the category-based routing that failed because Kalshi's API does
    not return a ``category`` field.
    """

    def __init__(
        self,
        indicator_strategies: list[Strategy],
        default: list[Strategy] | None = None,
    ) -> None:
        self._indicator_strategies = indicator_strategies
        self._default = default or []

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        indicator_contracts: list[Contract] = []
        other_contracts: list[Contract] = []

        for c in contracts:
            prefix = c.contract_id.split("-")[0] if c.contract_id else ""
            if prefix in INDICATORS:
                indicator_contracts.append(c)
            else:
                other_contracts.append(c)

        results: list[ScoredOpportunity] = []
        if indicator_contracts:
            for strategy in self._indicator_strategies:
                results.extend(await strategy.score(indicator_contracts, estimator))
        if other_contracts:
            for strategy in self._default:
                results.extend(await strategy.score(other_contracts, estimator))
        return results


class CategoryRouter(Strategy):
    """Routes contracts to category-specific strategy pipelines.

    Composite pattern: CategoryRouter is itself a Strategy, so the pipeline
    doesn't need to know whether it holds flat strategies or a router.

    .. note:: Kalshi's API does not populate the ``category`` field, so this
       router is only useful with sources that provide categories. For
       indicator-based routing, use :class:`IndicatorRouter` instead.
    """

    def __init__(
        self,
        routes: dict[str, list[Strategy]],
        default: list[Strategy] | None = None,
    ) -> None:
        self._routes = {k.lower(): v for k, v in routes.items()}
        self._default = default or []

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        by_category: dict[str, list[Contract]] = defaultdict(list)
        for c in contracts:
            by_category[c.category.lower()].append(c)

        results: list[ScoredOpportunity] = []
        for category, group in by_category.items():
            strategies = self._routes.get(category, self._default)
            if not strategies:
                logger.debug(
                    "CategoryRouter: no strategies for category=%r (%d contracts dropped)",
                    category,
                    len(group),
                )
                continue
            for strategy in strategies:
                results.extend(await strategy.score(group, estimator))
        return results
