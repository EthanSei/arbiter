"""Tests for the fee calculation module."""

import pytest

from arbiter.scoring.fees import flat_fee_rate, kalshi_fee


class TestKalshiFee:
    @pytest.mark.parametrize(
        ("price", "is_taker", "expected"),
        [
            # At P=0.50 (max fee): 0.07 * 0.5 * 0.5 = 0.0175 → ceil(1.75) = 0.02
            (0.50, True, 0.02),
            # At P=0.50 maker: 0.0175 * 0.25 = 0.004375 → ceil(0.4375) = 0.01
            (0.50, False, 0.01),
            # At P=0.0: fee = 0
            (0.0, True, 0.0),
            # At P=1.0: fee = 0
            (1.0, True, 0.0),
            # At P=0.01: 0.07 * 0.01 * 0.99 = 0.000693 → ceil(0.0693) = 0.01
            (0.01, True, 0.01),
            # At P=0.99: symmetric with 0.01
            (0.99, True, 0.01),
            # At P=0.60: 0.07 * 0.6 * 0.4 = 0.0168 → ceil(1.68) = 0.02
            (0.60, True, 0.02),
            # At P=0.05: 0.07 * 0.05 * 0.95 = 0.003325 → ceil(0.3325) = 0.01
            (0.05, True, 0.01),
            # At P=0.95: symmetric with 0.05
            (0.95, True, 0.01),
        ],
    )
    def test_kalshi_fee_values(self, price: float, is_taker: bool, expected: float) -> None:
        assert kalshi_fee(price, is_taker) == expected

    def test_taker_always_gte_maker(self) -> None:
        for p in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            assert kalshi_fee(p, True) >= kalshi_fee(p, False)

    def test_symmetric(self) -> None:
        """Fee at P is the same as at (1-P)."""
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert kalshi_fee(p, True) == kalshi_fee(1.0 - p, True)

    def test_max_at_midpoint(self) -> None:
        """Fee peaks at P=0.50."""
        max_fee = kalshi_fee(0.50, True)
        for p in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            assert kalshi_fee(p, True) <= max_fee

    def test_default_is_taker(self) -> None:
        assert kalshi_fee(0.50) == kalshi_fee(0.50, True)


class TestFlatFeeRate:
    def test_returns_fixed_rate(self) -> None:
        fee_fn = flat_fee_rate(0.01)
        assert fee_fn(0.50, True) == 0.01
        assert fee_fn(0.10, False) == 0.01
        assert fee_fn(0.99, True) == 0.01

    def test_zero_rate(self) -> None:
        fee_fn = flat_fee_rate(0.0)
        assert fee_fn(0.50, True) == 0.0
