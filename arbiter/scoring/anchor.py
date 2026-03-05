"""Anchor scoring — compare market prices against external probability anchors.

Given an economic indicator's consensus forecast (μ) and historical surprise
volatility (σ), compute the anchor probability for each Kalshi threshold K:

    P(X > K) = 1 - Φ((K - μ) / σ)

Flag contracts where the market price is significantly below the anchor
probability (underpriced YES side).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

from scipy.stats import norm

from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.kelly import kelly_criterion

_T_SUFFIX_RE = re.compile(r"-T([\d.]+)K?$", re.IGNORECASE)


class Calibrator(Protocol):
    def predict(self, x: Sequence[float]) -> Sequence[float]: ...


class PlattCalibrator:
    """Platt scaling calibrator: logistic regression on log-odds.

    Only 2 parameters (slope + intercept), resistant to overfitting on
    small samples unlike isotonic regression.

    Heavy imports (numpy, scipy.special, sklearn) are deferred to method
    calls so that importing this module doesn't add startup latency when
    calibrators are not used.
    """

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, probs: list[float], outcomes: list[float]) -> PlattCalibrator:
        import numpy as np
        from scipy.special import logit
        from sklearn.linear_model import LogisticRegression

        arr = np.clip(probs, 1e-6, 1 - 1e-6)
        features = logit(arr).reshape(-1, 1)
        y = np.array(outcomes)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(features, y)
        self._lr = lr
        return self

    def predict(self, x: Sequence[float]) -> Sequence[float]:
        if self._lr is None or len(x) == 0:
            return list(x)
        import numpy as np
        from scipy.special import logit

        arr = np.clip(x, 1e-6, 1 - 1e-6)
        features = logit(np.asarray(arr, dtype=float)).reshape(-1, 1)
        result: list[float] = self._lr.predict_proba(features)[:, 1].tolist()
        return result

    @property
    def coef(self) -> tuple[float, float]:
        """Return (slope, intercept) for diagnostics."""
        if self._lr is None:
            return (1.0, 0.0)
        return (float(self._lr.coef_[0, 0]), float(self._lr.intercept_[0]))


def extract_threshold(contract_id: str) -> tuple[str, float] | None:
    """Parse a Kalshi T-suffix ticker into (group_key, threshold).

    Examples:
        'KXCPI-26JAN-T0.003' → ('KXCPI-26JAN', 0.003)
        'KXJOBLESSCLAIMS-26MAR06-T220' → ('KXJOBLESSCLAIMS-26MAR06', 220.0)

    Returns None for tickers without a T-suffix (e.g. MAXMON/MINMON numeric suffixes).
    """
    m = _T_SUFFIX_RE.search(contract_id)
    if m is None:
        return None
    group_key = contract_id[: m.start()]
    threshold = float(m.group(1))
    return group_key, threshold


def group_anchor_contracts(
    contracts: list[Contract],
) -> dict[str, list[tuple[float, Contract]]]:
    """Group Kalshi T-suffix contracts by event prefix.

    Returns dict mapping group_key → list of (threshold, contract) sorted by
    threshold ascending. Filters to Kalshi source, skips zero-volume.
    """
    groups: dict[str, list[tuple[float, Contract]]] = defaultdict(list)

    for c in contracts:
        if c.source != "kalshi":
            continue
        if c.volume_24h <= 0:
            continue
        parsed = extract_threshold(c.contract_id)
        if parsed is None:
            continue
        group_key, threshold = parsed
        groups[group_key].append((threshold, c))

    return {k: sorted(v, key=lambda x: x[0]) for k, v in groups.items()}


def compute_anchor_prob(threshold: float, mu: float, sigma: float) -> float:
    """Compute P(X > threshold) = 1 - Φ((threshold - μ) / σ).

    Uses the normal survival function. Raises ValueError if σ ≤ 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return float(norm.sf(threshold, loc=mu, scale=sigma))


def find_anchor_mispricings(
    group: list[tuple[float, Contract]],
    mu: float,
    sigma: float,
    fee_rate: float = 0.01,
    threshold_scale: float = 1.0,
    calibrator: Calibrator | None = None,
) -> list[ScoredOpportunity]:
    """Compare anchor P(X>K) vs market YES price for each contract in a group.

    Flags contracts where anchor_prob > market_price + fee_rate (underpriced YES).
    Only produces YES-direction opportunities.

    When ``calibrator`` is provided, applies ``calibrator.predict([anchor_prob])[0]``
    to recalibrate the probability before scoring (e.g. isotonic regression).
    """
    results: list[ScoredOpportunity] = []

    for threshold, contract in group:
        anchor_prob = compute_anchor_prob(threshold * threshold_scale, mu, sigma)
        if calibrator is not None:
            anchor_prob = float(calibrator.predict([anchor_prob])[0])
        market_price = contract.yes_price
        ev = anchor_prob - market_price - fee_rate

        if ev <= 0:
            continue

        yes_cost = market_price + fee_rate
        payout_ratio = (1.0 / yes_cost) - 1.0 if yes_cost < 1.0 else 0.0
        kelly = kelly_criterion(anchor_prob, payout_ratio)

        results.append(
            ScoredOpportunity(
                contract=contract,
                direction="yes",
                market_price=market_price,
                model_probability=anchor_prob,
                expected_value=ev,
                kelly_size=kelly,
                strategy_name="AnchorStrategy",
            )
        )

    return results
