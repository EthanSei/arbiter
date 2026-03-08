"""Data export utilities for Jupyter notebook analysis.

Returns list[dict] -- no pandas dependency. Users: pd.DataFrame(result) in notebooks.
"""

import csv
import io
from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.db.models import (
    CandlestickBar,
    MarketSnapshot,
    Opportunity,
    OrderBookSnapshot,
    PaperTrade,
    Source,
)


def _paginate(
    stmt: Select[Any], limit: int | None, offset: int | None, order_by: Any = None
) -> Select[Any]:
    if order_by is not None:
        stmt = stmt.order_by(order_by)
    if offset is not None:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


async def export_snapshots(
    session: AsyncSession,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    source: str | None = None,
    category: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, object]]:
    """Export MarketSnapshot rows as list of dicts.

    Each dict has: id, source, contract_id, title, category,
    feature_version, outcome, snapshot_at, resolved_at,
    plus all feature keys flattened (yes_price, no_price, etc.)
    """
    stmt = select(MarketSnapshot)

    if start_date is not None:
        stmt = stmt.where(MarketSnapshot.snapshot_at >= start_date)
    if end_date is not None:
        stmt = stmt.where(MarketSnapshot.snapshot_at <= end_date)
    if source is not None:
        stmt = stmt.where(MarketSnapshot.source == Source(source))
    if category is not None:
        stmt = stmt.where(MarketSnapshot.category == category)

    stmt = _paginate(stmt, limit, offset, order_by=MarketSnapshot.snapshot_at)

    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    rows: list[dict[str, object]] = []
    for snap in snapshots:
        row: dict[str, object] = {
            "id": snap.id,
            "source": str(snap.source.value) if snap.source else None,
            "contract_id": snap.contract_id,
            "title": snap.title,
            "category": snap.category,
            "series_ticker": snap.series_ticker,
            "feature_version": snap.feature_version,
            "outcome": snap.outcome,
            "snapshot_at": snap.snapshot_at,
            "resolved_at": snap.resolved_at,
        }
        # Flatten features dict into top-level keys
        if snap.features:
            for key, value in snap.features.items():
                row[key] = value
        rows.append(row)

    return rows


async def export_opportunities(
    session: AsyncSession,
    *,
    active_only: bool = False,
    start_date: datetime | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, object]]:
    """Export Opportunity rows as list of dicts.

    Each dict has all Opportunity columns.
    """
    stmt = select(Opportunity)

    if active_only:
        stmt = stmt.where(Opportunity.active.is_(True))
    if start_date is not None:
        stmt = stmt.where(Opportunity.discovered_at >= start_date)

    stmt = _paginate(stmt, limit, offset, order_by=Opportunity.discovered_at)

    result = await session.execute(stmt)
    opportunities = result.scalars().all()

    rows: list[dict[str, object]] = []
    for opp in opportunities:
        rows.append(
            {
                "id": opp.id,
                "source": str(opp.source.value) if opp.source else None,
                "contract_id": opp.contract_id,
                "title": opp.title,
                "direction": str(opp.direction.value) if opp.direction else None,
                "strategy_name": opp.strategy_name,
                "market_price": opp.market_price,
                "model_probability": opp.model_probability,
                "expected_value": opp.expected_value,
                "kelly_size": opp.kelly_size,
                "expires_at": opp.expires_at,
                "discovered_at": opp.discovered_at,
                "active": opp.active,
                "last_alerted_at": opp.last_alerted_at,
                "last_seen_at": opp.last_seen_at,
            }
        )

    return rows


async def export_paper_trades(
    session: AsyncSession,
    *,
    settled_only: bool = False,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, object]]:
    """Export PaperTrade rows as list of dicts."""
    stmt = select(PaperTrade)

    if settled_only:
        stmt = stmt.where(PaperTrade.exited_at.is_not(None))

    stmt = _paginate(stmt, limit, offset, order_by=PaperTrade.entered_at)

    result = await session.execute(stmt)
    trades = result.scalars().all()

    rows: list[dict[str, object]] = []
    for t in trades:
        rows.append(
            {
                "id": t.id,
                "source": str(t.source.value) if t.source else None,
                "contract_id": t.contract_id,
                "direction": str(t.direction.value) if t.direction else None,
                "strategy_name": t.strategy_name,
                "entry_price": t.entry_price,
                "quantity": t.quantity,
                "model_probability": t.model_probability,
                "expected_value": t.expected_value,
                "entered_at": t.entered_at,
                "exit_price": t.exit_price,
                "exited_at": t.exited_at,
                "pnl": t.pnl,
                "outcome": t.outcome,
            }
        )

    return rows


async def export_order_book_snapshots(
    session: AsyncSession,
    *,
    contract_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, object]]:
    """Export OrderBookSnapshot rows as list of dicts."""
    stmt = select(OrderBookSnapshot)

    if contract_id is not None:
        stmt = stmt.where(OrderBookSnapshot.contract_id == contract_id)
    if start_date is not None:
        stmt = stmt.where(OrderBookSnapshot.snapshot_at >= start_date)
    if end_date is not None:
        stmt = stmt.where(OrderBookSnapshot.snapshot_at <= end_date)

    stmt = _paginate(stmt, limit, offset, order_by=OrderBookSnapshot.snapshot_at)

    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    return [
        {
            "id": s.id,
            "source": str(s.source.value) if s.source else None,
            "contract_id": s.contract_id,
            "series_ticker": s.series_ticker,
            "event_ticker": s.event_ticker,
            "bids": s.bids,
            "asks": s.asks,
            "snapshot_at": s.snapshot_at,
        }
        for s in snapshots
    ]


async def export_candlestick_bars(
    session: AsyncSession,
    *,
    contract_id: str | None = None,
    series_ticker: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, object]]:
    """Export CandlestickBar rows as list of dicts."""
    stmt = select(CandlestickBar)

    if contract_id is not None:
        stmt = stmt.where(CandlestickBar.contract_id == contract_id)
    if series_ticker is not None:
        stmt = stmt.where(CandlestickBar.series_ticker == series_ticker)
    if start_date is not None:
        stmt = stmt.where(CandlestickBar.period_start >= start_date)
    if end_date is not None:
        stmt = stmt.where(CandlestickBar.period_start <= end_date)

    stmt = _paginate(stmt, limit, offset, order_by=CandlestickBar.period_start)

    result = await session.execute(stmt)
    bars = result.scalars().all()

    return [
        {
            "id": b.id,
            "source": str(b.source.value) if b.source else None,
            "contract_id": b.contract_id,
            "series_ticker": b.series_ticker,
            "period_start": b.period_start,
            "period_interval": b.period_interval,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]


def to_csv(rows: list[dict[str, object]]) -> str:
    """Convert list of dicts to CSV string. Returns empty string for empty input."""
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
