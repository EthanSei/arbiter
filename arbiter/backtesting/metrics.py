"""Pure metric functions for backtesting evaluation."""

from __future__ import annotations

import math


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    Uses the standard formula: (mean_return - risk_free_rate) / std_return * sqrt(252).
    Returns 0.0 if no returns or zero standard deviation.
    """
    if len(returns) == 0:
        return 0.0

    n = len(returns)
    mean_ret = sum(returns) / n
    excess = mean_ret - risk_free_rate

    variance = sum((r - mean_ret) ** 2 for r in returns) / n
    std = math.sqrt(variance)

    if std == 0.0:
        return 0.0

    return (excess / std) * math.sqrt(252)


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough decline as a fraction.

    Returns 0.0 if no data or if the equity curve never declines.
    """
    if len(equity_curve) == 0:
        return 0.0

    peak = equity_curve[0]
    worst_drawdown = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > worst_drawdown:
            worst_drawdown = drawdown

    return worst_drawdown


def win_rate(pnls: list[float]) -> float:
    """Fraction of trades with positive P&L.

    Returns 0.0 if no trades.
    """
    if len(pnls) == 0:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: list[float]) -> float:
    """Sum of profits divided by sum of losses.

    Returns float('inf') if no losses, 0.0 if no profits or no trades.
    """
    if len(pnls) == 0:
        return 0.0

    total_profit = sum(p for p in pnls if p > 0)
    total_loss = sum(abs(p) for p in pnls if p < 0)

    if total_profit == 0.0:
        return 0.0
    if total_loss == 0.0:
        return float("inf")

    return total_profit / total_loss
