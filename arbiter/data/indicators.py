"""Indicator registry — maps Kalshi series to configs and providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndicatorConfig:
    """Configuration for an economic indicator tracked by arbiter."""

    kalshi_series: str
    providers: list[str] = field(default_factory=list)
    fred_series: str = ""
    transform: str = "level"  # "level", "mom_pct", or "mom_change"
    consensus_method: str = "prior_value"  # "prior_value" or "moving_average_4w"
    recency_halflife: int = 24


INDICATORS: dict[str, IndicatorConfig] = {
    "KXJOBLESSCLAIMS": IndicatorConfig(
        kalshi_series="KXJOBLESSCLAIMS",
        providers=["fred"],
        fred_series="ICSA",
        transform="level",
        consensus_method="moving_average_4w",
        recency_halflife=52,
    ),
    "KXCPI": IndicatorConfig(
        kalshi_series="KXCPI",
        providers=["fred", "bls"],
        fred_series="CPIAUCSL",
        transform="mom_pct",
        consensus_method="prior_value",
        recency_halflife=24,
    ),
}


def get_indicator(kalshi_series: str) -> IndicatorConfig | None:
    """Look up an indicator config by Kalshi series ticker."""
    return INDICATORS.get(kalshi_series)
