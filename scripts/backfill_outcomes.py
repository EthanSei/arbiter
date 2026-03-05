"""Backfill MarketSnapshot outcomes from Kalshi settlement data.

For each unresolved T-suffix snapshot in the DB, fetches the current market
status from Kalshi and sets outcome = 1.0 (YES) or 0.0 (NO) when settled.

Usage:
    python scripts/backfill_outcomes.py          # Dry run (show what would change)
    python scripts/backfill_outcomes.py --write  # Commit updates to DB
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from datetime import datetime

import httpx
from dotenv import load_dotenv
from sqlalchemy import or_, select, update

from arbiter.data.indicators import INDICATORS
from arbiter.db.models import MarketSnapshot
from arbiter.db.session import async_session_factory

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi public API rate limit is 20 RPM — sleep 3.1s between requests to stay safe.
_REQUEST_DELAY = 3.1


async def _fetch_market(client: httpx.AsyncClient, ticker: str) -> dict | None:
    """Fetch a single market from Kalshi. Returns the market dict or None on error."""
    try:
        resp = await client.get(f"{KALSHI_BASE}/markets/{ticker}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("market")
    except httpx.HTTPError as exc:
        print(f"  HTTP error for {ticker}: {exc}")
        return None


async def main(write: bool) -> None:
    load_dotenv()

    # Build filters for each registered economic indicator series.
    # This avoids matching sports/crypto T-suffix contracts (e.g. KXATPMATCH-TIR).
    series_filters = [
        MarketSnapshot.contract_id.like(f"{indicator_id}-%") for indicator_id in INDICATORS
    ]

    async with async_session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(MarketSnapshot.contract_id)
                    .where(MarketSnapshot.outcome.is_(None))
                    .where(or_(*series_filters))
                    .distinct()
                )
            )
            .scalars()
            .all()
        )

    contract_ids = sorted(rows)
    print(f"Found {len(contract_ids)} unresolved T-suffix contracts in DB")

    if not contract_ids:
        return

    # (contract_id, outcome, resolved_at_str)
    resolved: list[tuple[str, float, str]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, ticker in enumerate(contract_ids):
            market = await _fetch_market(client, ticker)
            if market is None:
                print(f"  [{i + 1}/{len(contract_ids)}] {ticker}: not found")
            else:
                status = market.get("status", "")
                result = market.get("result", "")
                if status == "settled" and result in ("yes", "no"):
                    outcome = 1.0 if result == "yes" else 0.0
                    close_time = str(market.get("close_time") or "")
                    print(
                        f"  [{i + 1}/{len(contract_ids)}] {ticker}: "
                        f"settled → {result} (outcome={outcome})"
                    )
                    resolved.append((ticker, outcome, close_time))
                else:
                    print(
                        f"  [{i + 1}/{len(contract_ids)}] {ticker}: "
                        f"status={status!r} result={result!r}"
                    )

            if i < len(contract_ids) - 1:
                await asyncio.sleep(_REQUEST_DELAY)

    print(f"\n{len(resolved)} of {len(contract_ids)} contracts are settled")

    if not resolved:
        return

    if not write:
        print("Dry run — pass --write to commit updates to DB")
        return

    async with async_session_factory() as session:
        for ticker, outcome, close_time in resolved:
            resolved_at: datetime | None = None
            if close_time:
                with contextlib.suppress(ValueError):
                    resolved_at = datetime.fromisoformat(close_time.replace("Z", "+00:00"))

            await session.execute(
                update(MarketSnapshot)
                .where(MarketSnapshot.contract_id == ticker)
                .where(MarketSnapshot.outcome.is_(None))
                .values(outcome=outcome, resolved_at=resolved_at)
            )
        await session.commit()
        print(f"Committed {len(resolved)} contract resolutions to DB")


if __name__ == "__main__":
    write = "--write" in sys.argv
    asyncio.run(main(write))
