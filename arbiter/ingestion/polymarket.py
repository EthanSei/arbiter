"""Polymarket API client for fetching prediction market data."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx

from arbiter.ingestion.base import Contract, MarketClient

_SPREAD_ESTIMATE = 0.02  # Approximate half-spread for bid/ask estimation
# Server-side pre-filter: markets with < $10k all-time volume are almost certainly
# inactive or illiquid.
_MIN_TOTAL_VOLUME_PREFILTER = 10_000.0


class PolymarketClient(MarketClient):
    """Fetches and normalizes market data from the Polymarket Gamma API.

    Issues a single request sorted by 24h volume descending. The API returns
    only active markets with >= _MIN_TOTAL_VOLUME_PREFILTER all-time volume,
    so the entire liquid corpus fits in one page (~20 markets at the 10k floor).
    Prices are normalized from decimal strings to floats in [0, 1].
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        gamma_base_url: str = "https://gamma-api.polymarket.com",
        min_volume_24h: float = 100.0,
    ) -> None:
        self._http = http
        self._gamma_base = gamma_base_url.rstrip("/")
        self._min_volume_24h = min_volume_24h

    async def fetch_markets(self, *, limit: int = 200) -> list[Contract]:
        params: dict[str, str | int | float] = {
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "volume_num_min": _MIN_TOTAL_VOLUME_PREFILTER,
        }
        resp = await self._http.get(f"{self._gamma_base}/markets", params=params)
        resp.raise_for_status()
        markets: list[dict[str, Any]] = resp.json()

        contracts: list[Contract] = []
        for m in markets:
            if m.get("closed") or not m.get("acceptingOrders", True):
                continue
            if _safe_float(m.get("volume24hr")) < self._min_volume_24h:
                continue
            contract = self._parse_market(m)
            if contract is not None:
                contracts.append(contract)
        return contracts

    async def close(self) -> None:
        pass  # Client is borrowed, not owned

    @staticmethod
    def _parse_market(m: dict[str, Any]) -> Contract | None:
        outcomes = _parse_json_list(m.get("outcomes"))
        prices = _parse_json_list(m.get("outcomePrices"))
        yes_price = _extract_outcome_price(outcomes, prices, "Yes")
        no_price = _extract_outcome_price(outcomes, prices, "No")
        if yes_price is None:
            return None
        if no_price is None:
            no_price = 1.0 - yes_price

        expires_at = None
        end_date = m.get("endDate")
        if end_date:
            expires_at = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))

        slug = str(m.get("slug", ""))
        url = f"https://polymarket.com/event/{slug}" if slug else ""

        return Contract(
            source="polymarket",
            contract_id=str(m.get("conditionId", "")),
            title=str(m.get("question", "")),
            category=str(m.get("category", "")),
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=max(0.0, yes_price - _SPREAD_ESTIMATE),
            yes_ask=min(1.0, yes_price + _SPREAD_ESTIMATE),
            last_price=None,  # Gamma API doesn't provide last trade price
            volume_24h=_safe_float(m.get("volume24hr")),
            open_interest=_safe_float(m.get("liquidity")),
            expires_at=expires_at,
            url=url,
            status="open",
        )


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_json_list(val: Any) -> list:
    """Return val as a list, parsing JSON strings if needed."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return []


def _extract_outcome_price(outcomes: list, prices: list, outcome: str) -> float | None:
    for i, o in enumerate(outcomes):
        if str(o).lower() == outcome.lower():
            try:
                return float(prices[i])
            except (IndexError, ValueError, TypeError):
                return None
    return None
