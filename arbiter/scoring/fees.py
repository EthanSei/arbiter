"""Fee calculation for prediction market platforms.

Kalshi uses a parabolic fee formula that peaks at P=0.50:
    taker_fee = ceil(0.07 * contracts * P * (1-P))  per contract
    maker_fee = ceil(0.0175 * contracts * P * (1-P)) per contract

For single-contract EV calculations we express fees as a fraction of $1 payout.
"""

from __future__ import annotations

import math
from collections.abc import Callable

# Type alias: fee function takes (price, is_taker) → fee as fraction of $1
FeeFn = Callable[[float, bool], float]


def kalshi_fee(price: float, is_taker: bool = True) -> float:
    """Kalshi's parabolic fee for a single contract.

    Args:
        price: Contract price in [0, 1].
        is_taker: True for taker (crossing the spread), False for maker (posting limit).

    Returns:
        Fee as a fraction of the $1 payout, rounded up to the nearest cent.
    """
    coeff = 0.07 if is_taker else 0.0175
    raw = coeff * price * (1.0 - price)
    # Kalshi rounds up to nearest cent ($0.01)
    return math.ceil(raw * 100) / 100


def flat_fee_rate(rate: float) -> FeeFn:
    """Return a FeeFn that always returns a fixed rate regardless of price."""

    def _fee(price: float, is_taker: bool = True) -> float:
        return rate

    return _fee
