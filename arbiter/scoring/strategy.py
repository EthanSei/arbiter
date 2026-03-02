"""Abstract base for scoring strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
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


def build_default_strategies(
    fee_rate: float = 0.01,
    target_categories: list[str] | None = None,
) -> list[Strategy]:
    """Build the default strategy list.

    When target_categories is provided, returns a CategoryRouter that routes
    those categories to [EVStrategy + ConsistencyStrategy] and everything
    else to [EVStrategy] only.

    When target_categories is None, returns the flat default:
    [EVStrategy, ConsistencyStrategy].
    """
    if not target_categories:
        return [EVStrategy(fee_rate), ConsistencyStrategy(fee_rate)]

    targeted = [EVStrategy(fee_rate), ConsistencyStrategy(fee_rate)]
    routes = {cat.lower(): targeted for cat in target_categories}
    router = CategoryRouter(routes=routes, default=[EVStrategy(fee_rate)])
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
