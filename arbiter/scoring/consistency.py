"""Range-market internal consistency scoring for Kalshi price bracket contracts.

Kalshi publishes families of "above $X" and "below $X" contracts on the same
underlying (e.g. KXBTCMAXMON-BTC-26MAR31-8000000).  By stochastic dominance:

  P(above X₁) ≥ P(above X₂)  whenever X₁ ≤ X₂
  P(below X₁) ≤ P(below X₂)  whenever X₁ ≤ X₂

A contract that violates these bounds is underpriced relative to its siblings.
We set model_prob = max(sibling probs that imply a higher floor) and score the
edge as model_prob - market_price - fee_rate.
"""

from __future__ import annotations

import re
from collections import defaultdict

from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity
from arbiter.scoring.kelly import kelly_criterion

_SUFFIX_RE = re.compile(r"-(\d+)$")
_MIN_GROUP_SIZE = 2
_EPSILON = 1e-4  # ignore floating-point noise below this level

# Kalshi names price-range series as KX<ASSET>MAXMON (above) or KX<ASSET>MINMON (below).
# "MAXMON" / "MINMON" are the canonical tokens; checking for bare "MAX"/"MIN" would
# match team codes like "MINDEN" (Minnesota/Denver) and produce false positives.
_IS_ABOVE_RE = re.compile(r"MAXMON", re.IGNORECASE)
_IS_BELOW_RE = re.compile(r"MINMON", re.IGNORECASE)


def find_consistency_violations(
    contracts: list[Contract],
    fee_rate: float = 0.0,
) -> list[ScoredOpportunity]:
    """Return ScoredOpportunity for each Kalshi range contract underpriced vs its siblings.

    Only processes Kalshi contracts whose ticker ends with a numeric threshold
    suffix (e.g. KXBTCMAXMON-BTC-26MAR31-8250000).  Groups are inferred from
    the base ticker (everything before the last -DIGITS segment).  Direction
    (above/below) is inferred from MAX/MIN in the base ticker.

    Args:
        contracts: All contracts from the current scan cycle.
        fee_rate: Execution cost applied to EV; same value used in compute_ev.

    Returns:
        ScoredOpportunity list, direction always "yes" (buy the underpriced side).
    """
    groups: dict[str, list[tuple[int, Contract]]] = defaultdict(list)

    for c in contracts:
        if c.source != "kalshi":
            continue
        m = _SUFFIX_RE.search(c.contract_id)
        if m is None:
            continue
        base = c.contract_id[: m.start()]
        if not _IS_ABOVE_RE.search(base) and not _IS_BELOW_RE.search(base):
            continue
        groups[base].append((int(m.group(1)), c))

    violations: list[ScoredOpportunity] = []

    for base, entries in groups.items():
        if len(entries) < _MIN_GROUP_SIZE:
            continue

        is_above = bool(_IS_ABOVE_RE.search(base))
        sorted_entries = sorted(entries, key=lambda x: x[0])
        # Only use prices from contracts with real trading activity as anchors.
        # volume_24h == 0 indicates a default midpoint (0.50) with no real market signal.
        traded = [
            (c.yes_price if c.volume_24h > 0 else 0.0, c) for _, c in sorted_entries
        ]

        for i, (_, contract) in enumerate(sorted_entries):
            # P(above X) ≥ P(above Y) for Y > X; P(below X) ≥ P(below Y) for Y < X
            # Floor derived only from contracts with real volume (traded anchors).
            candidates = traded[i + 1 :] if is_above else traded[:i]
            if not candidates:
                continue
            anchor_price, anchor_contract = max(candidates, key=lambda x: x[0])

            if anchor_price <= contract.yes_price + _EPSILON:
                continue

            model_prob = anchor_price
            ev = model_prob - contract.yes_price - fee_rate
            if ev <= 0:
                continue

            yes_cost = contract.yes_price + fee_rate
            payout_ratio = (1.0 / yes_cost) - 1.0 if yes_cost < 1.0 else 0.0
            kelly = kelly_criterion(model_prob, payout_ratio)

            violations.append(
                ScoredOpportunity(
                    contract=contract,
                    direction="yes",
                    market_price=contract.yes_price,
                    model_probability=model_prob,
                    expected_value=ev,
                    kelly_size=kelly,
                    anchor_contract=anchor_contract,
                )
            )

    return violations
