"""Stdout alert channel for local review."""

from __future__ import annotations

import json

from arbiter.alerts.base import AlertChannel
from arbiter.scoring.ev import ScoredOpportunity


class StdoutChannel(AlertChannel):
    """Prints alerts as JSON to stdout for easy copy-paste review."""

    async def send(self, opportunity: ScoredOpportunity) -> None:
        data = {
            "strategy": opportunity.strategy_name,
            "contract_id": opportunity.contract.contract_id,
            "title": opportunity.contract.title,
            "direction": opportunity.direction,
            "market_price": round(opportunity.market_price, 4),
            "model_probability": round(opportunity.model_probability, 4),
            "expected_value": round(opportunity.expected_value, 4),
            "kelly_size": round(opportunity.kelly_size, 4),
            "url": opportunity.contract.url,
        }
        print(json.dumps(data))

    async def close(self) -> None:
        pass
