"""Indicator registry — maps Kalshi series to configs and providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndicatorConfig:
    """Configuration for an economic indicator tracked by arbiter."""

    kalshi_series: str
    providers: list[str] = field(default_factory=list)
    fred_series: str = ""
    transform: str = "level"  # "level", "mom_pct", "mom_change", or "yoy_pct"
    consensus_method: str = "prior_value"  # "prior_value" or "moving_average_4w"
    recency_halflife: int = 24
    threshold_scale: float = 1.0  # multiply Kalshi ticker threshold to get FRED units


INDICATORS: dict[str, IndicatorConfig] = {
    "KXJOBLESSCLAIMS": IndicatorConfig(
        kalshi_series="KXJOBLESSCLAIMS",
        providers=["fred"],
        fred_series="ICSA",
        transform="level",
        consensus_method="moving_average_4w",
        recency_halflife=52,
        threshold_scale=1.0,
    ),
    "KXCPI": IndicatorConfig(
        kalshi_series="KXCPI",
        providers=["fred", "bls"],
        fred_series="CPIAUCSL",
        transform="mom_pct",
        consensus_method="prior_value",
        recency_halflife=24,
        threshold_scale=0.01,
    ),
    "KXCPIYOY": IndicatorConfig(
        kalshi_series="KXCPIYOY",
        providers=["fred"],
        fred_series="CPIAUCSL",
        transform="yoy_pct",
        consensus_method="prior_value",
        recency_halflife=24,
        threshold_scale=0.01,
    ),
    "KXCPICOREYOY": IndicatorConfig(
        kalshi_series="KXCPICOREYOY",
        providers=["fred"],
        fred_series="CPILFESL",
        transform="yoy_pct",
        consensus_method="prior_value",
        recency_halflife=24,
        threshold_scale=0.01,
    ),
    "KXPAYROLLS": IndicatorConfig(
        kalshi_series="KXPAYROLLS",
        providers=["fred"],
        fred_series="PAYEMS",
        transform="mom_change",
        consensus_method="prior_value",
        recency_halflife=24,
        threshold_scale=0.001,
    ),
}


def get_indicator(kalshi_series: str) -> IndicatorConfig | None:
    """Look up an indicator config by Kalshi series ticker."""
    return INDICATORS.get(kalshi_series)
