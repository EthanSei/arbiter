"""Data export utilities for Jupyter notebook analysis.

Returns list[dict] -- no pandas dependency. Users: pd.DataFrame(result) in notebooks.
"""

import csv
import io
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.db.models import MarketSnapshot, Opportunity, PaperTrade, Source


async def export_snapshots(
    session: AsyncSession,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    source: str | None = None,
    category: str | None = None,
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
) -> list[dict[str, object]]:
    """Export Opportunity rows as list of dicts.

    Each dict has all Opportunity columns.
    """
    stmt = select(Opportunity)

    if active_only:
        stmt = stmt.where(Opportunity.active.is_(True))
    if start_date is not None:
        stmt = stmt.where(Opportunity.discovered_at >= start_date)

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
) -> list[dict[str, object]]:
    """Export PaperTrade rows as list of dicts."""
    stmt = select(PaperTrade)

    if settled_only:
        stmt = stmt.where(PaperTrade.exited_at.is_not(None))

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


def to_csv(rows: list[dict[str, object]]) -> str:
    """Convert list of dicts to CSV string. Returns empty string for empty input."""
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
