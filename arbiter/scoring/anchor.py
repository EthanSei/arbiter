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

from scipy.stats import norm

from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.kelly import kelly_criterion

_T_SUFFIX_RE = re.compile(r"-T([\d.]+)K?$", re.IGNORECASE)


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
) -> list[ScoredOpportunity]:
    """Compare anchor P(X>K) vs market YES price for each contract in a group.

    Flags contracts where anchor_prob > market_price + fee_rate (underpriced YES).
    Only produces YES-direction opportunities.
    """
    results: list[ScoredOpportunity] = []

    for threshold, contract in group:
        anchor_prob = compute_anchor_prob(threshold, mu, sigma)
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
