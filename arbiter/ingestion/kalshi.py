"""Kalshi API client for fetching prediction market data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from arbiter.ingestion.base import Contract, HttpClient, MarketClient


class KalshiClient(MarketClient):
    """Fetches and normalizes market data from the Kalshi API.

    Uses cursor-based pagination with limit=1000 per page (API max). At 20 RPM,
    fetching all ~10k markets takes ~30s (10 requests × 3s). Prices are normalized
    from dollar strings to floats in [0, 1].

    Pagination stops at the first of:
    - Cursor is empty (natural end of results)
    - total_fetched reaches max_markets (safety cap)
    - max_empty_pages consecutive pages yield zero qualifying contracts
    """

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        max_markets: int = 10_000,
        min_volume_24h: float = 5.0,
        max_empty_pages: int = 5,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._max_markets = max_markets
        self._min_volume_24h = min_volume_24h
        self._max_empty_pages = max_empty_pages

    async def fetch_markets(self, *, limit: int = 1000) -> list[Contract]:
        contracts: list[Contract] = []
        total_fetched = 0
        consecutive_empty = 0
        cursor = ""
        while True:
            # NOTE: Kalshi accepts "open" as the query-param value to request open
            # markets, but the response body returns "status": "active" for those
            # records — both values are accepted in the _parse_market filter below.
            params: dict[str, str | int] = {
                "limit": limit,
                "status": "open",
                "mve_filter": "exclude",
            }
            if cursor:
                params["cursor"] = cursor

            resp = await self._http.get(f"{self._base_url}/markets", params=params)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

            page = data.get("markets", [])
            if not page:
                break  # Guard against stale cursor returning empty page

            total_fetched += len(page)
            count_before = len(contracts)
            for m in page:
                if m.get("status") not in ("open", "active"):
                    continue
                contract = self._parse_market(m)
                if contract is not None and contract.volume_24h >= self._min_volume_24h:
                    contracts.append(contract)

            if len(contracts) == count_before:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            cursor = data.get("cursor", "")
            if (
                not cursor
                or total_fetched >= self._max_markets
                or consecutive_empty >= self._max_empty_pages
            ):
                break
        return contracts

    async def close(self) -> None:
        pass  # Client is borrowed, not owned

    @staticmethod
    def _parse_market(m: dict[str, Any]) -> Contract | None:
        yes_bid = _to_float(m.get("yes_bid_dollars"))
        yes_ask = _to_float(m.get("yes_ask_dollars"))
        last_price = _to_float(m.get("last_price_dollars"))

        # Prefer live bid/ask; fall back to last_price midpoint for markets
        # with no active orders (still useful for snapshot/feature collection).
        if yes_bid is None or yes_ask is None:
            if last_price is None:
                return None
            yes_bid = last_price
            yes_ask = last_price
        yes_price = (yes_bid + yes_ask) / 2.0

        expires_at = None
        close_time = m.get("close_time")
        if close_time:
            expires_at = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))

        return Contract(
            source="kalshi",
            contract_id=str(m["ticker"]),
            title=str(m.get("title", "")),
            category=str(m.get("category", "")),
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            last_price=last_price,
            volume_24h=_to_float(m.get("volume_24h_fp")) or 0.0,
            open_interest=_to_float(m.get("open_interest")) or 0.0,
            expires_at=expires_at,
            url=m.get("url") or _build_kalshi_url(str(m["ticker"])),
            status=str(m.get("status", "open")),
        )


def _build_kalshi_url(ticker: str) -> str:
    """Build a two-segment Kalshi URL that works as a mobile deep link.

    Format: https://kalshi.com/markets/{series_prefix}/{ticker}
    Series prefix is the text before the first '-' in the ticker.
    """
    series_prefix = ticker.split("-", 1)[0]
    return f"https://kalshi.com/markets/{series_prefix}/{ticker}"


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
