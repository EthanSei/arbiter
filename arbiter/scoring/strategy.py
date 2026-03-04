"""Abstract base for scoring strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from arbiter.data.providers.base import FeatureProvider
from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
from arbiter.scoring.anchor import find_anchor_mispricings, group_anchor_contracts
from arbiter.scoring.consistency import find_consistency_violations
from arbiter.scoring.ev import ScoredOpportunity, compute_ev

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

    def __init__(self, fee_rate: float = 0.01) -> None:
        self._fee_rate = fee_rate

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        results: list[ScoredOpportunity] = []
        for contract in contracts:
            model_prob = await estimator.estimate(contract)
            scored = compute_ev(contract, model_prob, self._fee_rate)
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

    def __init__(self, fee_rate: float = 0.01) -> None:
        self._fee_rate = fee_rate

    async def score(
        self,
        contracts: list[Contract],
        estimator: ProbabilityEstimator,
    ) -> list[ScoredOpportunity]:
        results = find_consistency_violations(contracts, self._fee_rate)
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
    ) -> None:
        self._providers = providers
        self._fee_rate = fee_rate

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

            opps = find_anchor_mispricings(group, mu, sigma, self._fee_rate)
            for opp in opps:
                opp.strategy_name = self.name
            results.extend(opps)

        return results

    def _aggregate_anchor_params(
        self, indicator_id: str
    ) -> tuple[float | None, float | None]:
        """Load FeatureSets from all providers and aggregate μ and σ."""
        mu: float | None = None
        sigma: float | None = None

        for provider in self._providers:
            fs = provider.load(indicator_id)
            if fs is None:
                continue
            if mu is None and fs.anchor_mu is not None:
                mu = fs.anchor_mu
            if sigma is None and fs.anchor_sigma is not None:
                sigma = fs.anchor_sigma

        return mu, sigma


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
    target_categories: list[str] | None = None,
    anchor_providers: list[FeatureProvider] | None = None,
    include_ev: bool = True,
) -> list[Strategy]:
    """Build the default strategy list.

    Uses YesOnlyEVStrategy (YES-side only) as the EV strategy because
    backtest analysis showed the model is anti-calibrated on NO-side trades.
    Pass include_ev=False to omit it when the model is not calibrated —
    raw uncalibrated probabilities produce too many false signals.

    When target_categories is provided, returns a CategoryRouter that routes
    those categories to [EV? + Consistency + Anchor?] and others to [EV?] only.

    When target_categories is None, returns the flat list.
    """
    ev: list[Strategy] = [YesOnlyEVStrategy(fee_rate)] if include_ev else []
    anchor: list[Strategy] = (
        [AnchorStrategy(anchor_providers, fee_rate)] if anchor_providers else []
    )

    if not target_categories:
        return ev + [ConsistencyStrategy(fee_rate)] + anchor

    # For targeted categories (Economics/Financials), exclude LightGBM EV when anchor
    # is available: the model has no current-cycle knowledge for economic releases and
    # generates unreliable signals. AnchorStrategy is the correct pricer for these.
    # Fall back to EV only when no anchor data exists.
    targeted = (
        [ConsistencyStrategy(fee_rate)] + anchor
        if anchor
        else ev + [ConsistencyStrategy(fee_rate)]
    )
    default = ev  # non-economics categories: EV only (or nothing when uncalibrated)
    routes = {cat.lower(): targeted for cat in target_categories}
    router = CategoryRouter(routes=routes, default=default)
    return [router]


class CategoryRouter(Strategy):
    """Routes contracts to category-specific strategy pipelines.

    Composite pattern: CategoryRouter is itself a Strategy, so the pipeline
    doesn't need to know whether it holds flat strategies or a router.
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
