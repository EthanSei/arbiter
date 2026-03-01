"""Kelly criterion for optimal position sizing."""


def kelly_criterion(win_prob: float, payout_ratio: float) -> float:
    """Compute the full Kelly fraction for a binary bet.

    f* = (b * p - q) / b

    where:
        b = payout_ratio (net profit per dollar wagered on a win)
        p = probability of winning
        q = 1 - p (probability of losing)

    Returns 0 if edge is non-positive (no bet).
    The caller should multiply by a fractional Kelly factor (e.g., 0.25)
    for practical position sizing.

    Args:
        win_prob: Probability of winning, in (0, 1).
        payout_ratio: Net profit per dollar wagered on win.
            For a binary contract at price c, payout_ratio = (1/c) - 1.

    Returns:
        Optimal fraction of bankroll to wager, or 0 if no edge.
    """
    if payout_ratio <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    q = 1.0 - win_prob
    f_star = (payout_ratio * win_prob - q) / payout_ratio
    return max(f_star, 0.0)
