"""Export MarketSnapshot rows from the database to parquet.

Exports in batches to avoid Supabase statement timeouts. Incrementally appends
to existing parquet file — only fetches rows newer than the latest snapshot_at
already in the file.

Usage:
    python scripts/export_snapshots.py            # Incremental (or full if no parquet exists)
    python scripts/export_snapshots.py --days 90  # Override: fetch last N days
    python scripts/export_snapshots.py --full     # Force re-fetch everything
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from arbiter.db.session import async_session_factory, init_db
from arbiter.export.dataframes import export_snapshots

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "snapshots.parquet"

BATCH_SIZE = 10_000
DEFAULT_LOOKBACK_DAYS = 90


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Export snapshots to parquet")
    parser.add_argument(
        "--days", type=int, default=None, help="Lookback days (overrides incremental)"
    )
    parser.add_argument("--full", action="store_true", help="Force full re-export")
    args = parser.parse_args()

    await init_db()

    # Determine start date: incremental from existing parquet, or explicit lookback
    existing_df = None
    if args.full:
        start = datetime.now(UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        print(f"Full re-export: fetching last {DEFAULT_LOOKBACK_DAYS} days...")
    elif args.days is not None:
        start = datetime.now(UTC) - timedelta(days=args.days)
        print(f"Fetching last {args.days} days...")
    elif PARQUET_PATH.exists():
        existing_df = pd.read_parquet(PARQUET_PATH)
        latest = pd.to_datetime(existing_df["snapshot_at"]).max()
        start = latest.to_pydatetime()
        print(f"Incremental: fetching snapshots after {start} ({len(existing_df)} existing rows)")
    else:
        start = datetime.now(UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        print(f"No existing parquet — fetching last {DEFAULT_LOOKBACK_DAYS} days...")

    all_rows: list[dict] = []
    offset = 0

    while True:
        async with async_session_factory() as session:
            batch = await export_snapshots(
                session, start_date=start, limit=BATCH_SIZE, offset=offset
            )
        if not batch:
            break
        all_rows.extend(batch)
        print(f"  Fetched {len(all_rows)} rows...")
        offset += BATCH_SIZE

    if not all_rows and existing_df is None:
        print("No snapshots found.")
        return

    new_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if existing_df is not None and not new_df.empty:
        new_df = pd.concat([existing_df, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["id"], keep="last")
    elif existing_df is not None:
        print("No new rows since last export.")
        return
    elif PARQUET_PATH.exists() and not args.full:
        existing = pd.read_parquet(PARQUET_PATH)
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["id"], keep="last")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(PARQUET_PATH, index=False)
    print(f"\nDone. {len(new_df)} total rows in {PARQUET_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
