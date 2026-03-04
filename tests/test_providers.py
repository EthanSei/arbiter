"""Tests for the FeatureProvider system and concrete providers."""

from __future__ import annotations

import json
import os

import pytest

from arbiter.data.indicators import INDICATORS, get_indicator
from arbiter.data.providers.base import FeatureSet
from arbiter.data.providers.bls import BLSComponentProvider
from arbiter.data.providers.fred import FREDSurpriseProvider

# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------


class TestFeatureSet:
    def test_construct_with_all_fields(self):
        fs = FeatureSet(
            provider="fred",
            indicator_id="KXCPI",
            anchor_mu=0.003,
            anchor_sigma=0.0015,
            features={"extra": 1.0},
        )
        assert fs.provider == "fred"
        assert fs.indicator_id == "KXCPI"
        assert fs.anchor_mu == 0.003
        assert fs.anchor_sigma == 0.0015
        assert fs.features == {"extra": 1.0}

    def test_none_for_mu_and_sigma(self):
        fs = FeatureSet(provider="bls", indicator_id="KXCPI")
        assert fs.anchor_mu is None
        assert fs.anchor_sigma is None

    def test_empty_features_by_default(self):
        fs = FeatureSet(provider="fred", indicator_id="KXCPI")
        assert fs.features == {}

    def test_frozen(self):
        fs = FeatureSet(provider="fred", indicator_id="KXCPI")
        with pytest.raises(AttributeError):
            fs.provider = "bls"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FREDSurpriseProvider
# ---------------------------------------------------------------------------


def _make_fred_cache(
    tmpdir: str,
    indicator_id: str,
    observations: list[dict],
    current_consensus: float,
) -> str:
    """Write a FRED cache JSON file and return the directory path."""
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, f"{indicator_id}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "indicator_id": indicator_id,
                "observations": observations,
                "current_consensus": current_consensus,
            },
            f,
        )
    return tmpdir


class TestFREDSurpriseProvider:
    def test_loads_cache_and_returns_feature_set(self, tmp_path):
        obs = [
            {"date": "2025-01-01", "actual": 220.0, "consensus": 215.0, "surprise": 5.0},
            {"date": "2025-01-08", "actual": 210.0, "consensus": 218.0, "surprise": -8.0},
            {"date": "2025-01-15", "actual": 215.0, "consensus": 212.0, "surprise": 3.0},
        ]
        data_dir = _make_fred_cache(str(tmp_path), "KXJOBLESSCLAIMS", obs, 215.0)

        provider = FREDSurpriseProvider(data_dir=data_dir)
        result = provider.load("KXJOBLESSCLAIMS")

        assert result is not None
        assert result.provider == "fred"
        assert result.indicator_id == "KXJOBLESSCLAIMS"
        assert result.anchor_mu == pytest.approx(215.0)
        assert result.anchor_sigma is not None
        assert result.anchor_sigma > 0

    def test_anchor_mu_matches_current_consensus(self, tmp_path):
        obs = [
            {"date": "2025-01-01", "actual": 0.003, "consensus": 0.002, "surprise": 0.001},
            {"date": "2025-02-01", "actual": 0.004, "consensus": 0.003, "surprise": 0.001},
        ]
        data_dir = _make_fred_cache(str(tmp_path), "KXCPI", obs, 0.0028)

        provider = FREDSurpriseProvider(data_dir=data_dir)
        result = provider.load("KXCPI")

        assert result is not None
        assert result.anchor_mu == pytest.approx(0.0028)

    def test_simple_sigma_matches_known_values(self, tmp_path):
        # Surprises: [10, -10, 10, -10] → std = 10.0
        obs = [
            {"date": f"2025-01-{i:02d}", "actual": 0, "consensus": 0, "surprise": s}
            for i, s in enumerate([10.0, -10.0, 10.0, -10.0], start=1)
        ]
        data_dir = _make_fred_cache(str(tmp_path), "KXTEST", obs, 0.0)

        provider = FREDSurpriseProvider(data_dir=data_dir, halflife=None)
        result = provider.load("KXTEST")

        assert result is not None
        assert result.anchor_sigma == pytest.approx(10.0)

    def test_exponential_weighting_recent_data_weighted_more(self, tmp_path):
        # Old surprises are large (100), recent surprises are small (1)
        obs = [
            {"date": "2020-01-01", "actual": 0, "consensus": 0, "surprise": 100.0},
            {"date": "2020-02-01", "actual": 0, "consensus": 0, "surprise": -100.0},
            {"date": "2020-03-01", "actual": 0, "consensus": 0, "surprise": 100.0},
            {"date": "2025-10-01", "actual": 0, "consensus": 0, "surprise": 1.0},
            {"date": "2025-11-01", "actual": 0, "consensus": 0, "surprise": -1.0},
            {"date": "2025-12-01", "actual": 0, "consensus": 0, "surprise": 1.0},
            {"date": "2026-01-01", "actual": 0, "consensus": 0, "surprise": -1.0},
        ]
        data_dir = _make_fred_cache(str(tmp_path), "KXTEST", obs, 0.0)

        # With short halflife, recent small surprises should dominate
        provider_weighted = FREDSurpriseProvider(data_dir=data_dir, halflife=3)
        result_weighted = provider_weighted.load("KXTEST")

        # Without weighting, old large surprises inflate sigma
        provider_simple = FREDSurpriseProvider(data_dir=data_dir, halflife=None)
        result_simple = provider_simple.load("KXTEST")

        assert result_weighted is not None
        assert result_simple is not None
        assert result_weighted.anchor_sigma < result_simple.anchor_sigma

    def test_returns_none_for_unknown_indicator(self, tmp_path):
        provider = FREDSurpriseProvider(data_dir=str(tmp_path))
        result = provider.load("KXNONEXISTENT")
        assert result is None

    def test_uses_indicator_config_halflife_when_registered(self, tmp_path):
        # KXJOBLESSCLAIMS has recency_halflife=52; KXTEST (unregistered) has none.
        # Both get the same old-large / recent-small surprise pattern.
        # KXJOBLESSCLAIMS should produce a smaller sigma due to its configured halflife.
        obs = [
            {"date": "2010-01-01", "actual": 0, "consensus": 0, "surprise": 5000.0},
            {"date": "2010-02-01", "actual": 0, "consensus": 0, "surprise": -5000.0},
            {"date": "2025-01-01", "actual": 0, "consensus": 0, "surprise": 5.0},
            {"date": "2025-02-01", "actual": 0, "consensus": 0, "surprise": -5.0},
        ]
        data_dir = str(tmp_path)
        for indicator in ("KXJOBLESSCLAIMS", "KXTEST"):
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, f"{indicator}.json")
            with open(path, "w") as f:
                json.dump({"observations": obs, "current_consensus": 0.0}, f)

        provider = FREDSurpriseProvider(data_dir=data_dir, halflife=None)
        result_configured = provider.load("KXJOBLESSCLAIMS")  # halflife=52 from registry
        result_unconfigured = provider.load("KXTEST")  # no registry entry → no halflife

        assert result_configured is not None
        assert result_unconfigured is not None
        # Registered indicator uses halflife, downweights the old large surprises
        assert result_configured.anchor_sigma < result_unconfigured.anchor_sigma

    def test_returns_none_for_missing_current_consensus(self, tmp_path):
        """Returns None gracefully when cache lacks the current_consensus key."""
        path = os.path.join(str(tmp_path), "KXCPI.json")
        obs = [
            {"date": "2025-01-01", "actual": 0.003, "consensus": 0.002, "surprise": 0.001},
            {"date": "2025-02-01", "actual": 0.004, "consensus": 0.003, "surprise": 0.001},
        ]
        with open(path, "w") as f:
            json.dump({"observations": obs}, f)  # No "current_consensus" key
        provider = FREDSurpriseProvider(data_dir=str(tmp_path))
        result = provider.load("KXCPI")
        assert result is None

    def test_returns_none_for_malformed_observation(self, tmp_path):
        """Returns None gracefully when an observation is missing the 'surprise' key."""
        path = os.path.join(str(tmp_path), "KXCPI.json")
        obs = [
            {"date": "2025-01-01", "actual": 0.003, "consensus": 0.002},  # no "surprise"
            {"date": "2025-02-01", "actual": 0.004, "consensus": 0.003, "surprise": 0.001},
        ]
        with open(path, "w") as f:
            json.dump({"observations": obs, "current_consensus": 0.003}, f)
        provider = FREDSurpriseProvider(data_dir=str(tmp_path))
        result = provider.load("KXCPI")
        assert result is None

    def test_requires_at_least_two_observations_for_sigma(self, tmp_path):
        obs = [
            {"date": "2025-01-01", "actual": 0.003, "consensus": 0.002, "surprise": 0.001},
        ]
        data_dir = _make_fred_cache(str(tmp_path), "KXCPI", obs, 0.003)

        provider = FREDSurpriseProvider(data_dir=data_dir)
        result = provider.load("KXCPI")

        # With only 1 observation, can't compute sigma → return None
        assert result is None


# ---------------------------------------------------------------------------
# BLSComponentProvider
# ---------------------------------------------------------------------------


def _make_bls_cache(
    tmpdir: str,
    indicator_id: str,
    components: dict[str, float],
) -> str:
    """Write a BLS cache JSON file and return the directory path."""
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, f"{indicator_id}.json")
    with open(path, "w") as f:
        json.dump({"indicator_id": indicator_id, "components": components}, f)
    return tmpdir


class TestBLSComponentProvider:
    def test_loads_cache_returns_feature_set(self, tmp_path):
        components = {
            "shelter_cpi_mom": 0.40,
            "energy_cpi_mom": -1.2,
            "food_cpi_mom": 0.25,
            "core_cpi_mom": 0.28,
        }
        data_dir = _make_bls_cache(str(tmp_path), "KXCPI", components)

        provider = BLSComponentProvider(data_dir=data_dir)
        result = provider.load("KXCPI")

        assert result is not None
        assert result.provider == "bls"
        assert result.indicator_id == "KXCPI"

    def test_anchor_mu_and_sigma_are_none(self, tmp_path):
        """BLS provides features, not anchor parameters."""
        components = {"shelter_cpi_mom": 0.40}
        data_dir = _make_bls_cache(str(tmp_path), "KXCPI", components)

        provider = BLSComponentProvider(data_dir=data_dir)
        result = provider.load("KXCPI")

        assert result is not None
        assert result.anchor_mu is None
        assert result.anchor_sigma is None

    def test_features_contain_component_data(self, tmp_path):
        components = {
            "shelter_cpi_mom": 0.40,
            "energy_cpi_mom": -1.2,
            "food_cpi_mom": 0.25,
            "core_cpi_mom": 0.28,
        }
        data_dir = _make_bls_cache(str(tmp_path), "KXCPI", components)

        provider = BLSComponentProvider(data_dir=data_dir)
        result = provider.load("KXCPI")

        assert result is not None
        assert result.features["shelter_cpi_mom"] == pytest.approx(0.40)
        assert result.features["energy_cpi_mom"] == pytest.approx(-1.2)
        assert result.features["food_cpi_mom"] == pytest.approx(0.25)
        assert result.features["core_cpi_mom"] == pytest.approx(0.28)

    def test_returns_none_for_unknown_indicator(self, tmp_path):
        provider = BLSComponentProvider(data_dir=str(tmp_path))
        result = provider.load("KXJOBLESSCLAIMS")
        assert result is None

    def test_returns_none_for_non_numeric_component(self, tmp_path):
        """Returns None when a component value cannot be converted to float."""
        path = os.path.join(str(tmp_path), "KXCPI.json")
        with open(path, "w") as f:
            json.dump({"components": {"shelter_cpi_mom": "not-a-number"}}, f)
        provider = BLSComponentProvider(data_dir=str(tmp_path))
        result = provider.load("KXCPI")
        assert result is None


# ---------------------------------------------------------------------------
# IndicatorConfig registry
# ---------------------------------------------------------------------------


class TestIndicatorConfig:
    def test_jobless_claims_config(self):
        config = get_indicator("KXJOBLESSCLAIMS")
        assert config is not None
        assert config.kalshi_series == "KXJOBLESSCLAIMS"
        assert config.fred_series == "ICSA"
        assert config.transform == "level"
        assert config.consensus_method == "moving_average_4w"
        assert "fred" in config.providers

    def test_cpi_config_has_both_providers(self):
        config = get_indicator("KXCPI")
        assert config is not None
        assert config.kalshi_series == "KXCPI"
        assert config.fred_series == "CPIAUCSL"
        assert config.transform == "mom_pct"
        assert config.consensus_method == "prior_value"
        assert "fred" in config.providers
        assert "bls" in config.providers

    def test_unknown_series_returns_none(self):
        assert get_indicator("KXNONEXISTENT") is None

    def test_all_indicators_have_required_fields(self):
        for series, config in INDICATORS.items():
            assert config.kalshi_series == series
            assert len(config.providers) > 0
            assert config.fred_series
            assert config.transform in ("level", "mom_pct", "mom_change")
            assert config.recency_halflife > 0
