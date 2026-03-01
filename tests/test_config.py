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
