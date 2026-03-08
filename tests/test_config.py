"""Tests for configuration validation."""

import pytest
from pydantic import ValidationError

from arbiter.config import Settings


class TestConfigValidation:
    def test_poll_interval_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(poll_interval_seconds=0)

    def test_ev_threshold_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(ev_threshold=-0.01)
        with pytest.raises(ValidationError):
            Settings(ev_threshold=0.0)

    def test_kelly_fraction_must_be_in_unit_interval(self):
        with pytest.raises(ValidationError):
            Settings(kelly_fraction=0.0)
        with pytest.raises(ValidationError):
            Settings(kelly_fraction=1.1)

    def test_fee_rate_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Settings(fee_rate=-0.01)

    def test_valid_defaults(self):
        s = Settings()
        assert s.poll_interval_seconds == 300
        assert s.ev_threshold == 0.05
        assert 0 < s.kelly_fraction < 1
        assert s.fee_rate >= 0

    def test_kalshi_target_series_defaults_to_economics(self):
        s = Settings()
        tickers = [t.strip() for t in s.kalshi_target_series.split(",")]
        assert "KXCPI" in tickers
        assert "KXPAYROLLS" in tickers
        assert "KXCPIYOY" in tickers

    def test_kalshi_target_series_accepts_comma_separated(self):
        s = Settings(kalshi_target_series="KXCPI,KXPAYROLLS,KXCPIYOY")
        assert s.kalshi_target_series == "KXCPI,KXPAYROLLS,KXCPIYOY"

    def test_polymarket_enabled_default_false(self):
        s = Settings()
        assert s.polymarket_enabled is False

    def test_polymarket_enabled_accepts_true(self):
        s = Settings(polymarket_enabled=True)
        assert s.polymarket_enabled is True

    def test_paper_trade_only_defaults_true(self):
        s = Settings()
        assert s.paper_trade_only is True

    def test_paper_trade_only_accepts_false(self):
        s = Settings(paper_trade_only=False)
        assert s.paper_trade_only is False
