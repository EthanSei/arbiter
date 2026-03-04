"""Base types for the FeatureProvider system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FeatureSet:
    """Standardized output from any feature provider.

    anchor_mu/sigma contribute to the P(X>K) anchor calculation directly.
    features dict holds arbitrary values for the future ML correction layer.
    """

    provider: str
    indicator_id: str
    anchor_mu: float | None = None
    anchor_sigma: float | None = None
    features: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class FeatureProvider(Protocol):
    """Protocol for external data sources.

    Each provider loads cached data for an indicator and returns a FeatureSet.
    The cache is populated by a separate fetch script (not called at runtime).
    """

    @property
    def name(self) -> str: ...

    def load(self, indicator_id: str) -> FeatureSet | None: ...
