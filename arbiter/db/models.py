"""SQLAlchemy ORM models for opportunity tracking, alert logging, and training data."""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Source(enum.StrEnum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class Direction(enum.StrEnum):
    YES = "yes"
    NO = "no"


class Opportunity(Base):
    """A scored opportunity discovered by the scanner.

    Uses state-based deduplication: `active` flag tracks whether this opportunity
    is still live. `last_alerted_at` and `last_seen_at` control re-alert logic.
    """

    __tablename__ = "opportunities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[Source] = mapped_column(Enum(Source))
    contract_id: Mapped[str] = mapped_column(String(256))
    title: Mapped[str] = mapped_column(Text)
    direction: Mapped[Direction] = mapped_column(Enum(Direction))

    strategy_name: Mapped[str] = mapped_column(String(64), default="")

    market_price: Mapped[float] = mapped_column(Float)
    model_probability: Mapped[float] = mapped_column(Float)
    expected_value: Mapped[float] = mapped_column(Float)
    kelly_size: Mapped[float] = mapped_column(Float, default=0.0)

    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # State-based deduplication fields
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_dedup", "contract_id", "direction"),)


class AlertLog(Base):
    """Record of an alert sent for an opportunity."""

    __tablename__ = "alert_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    opportunity_id: Mapped[str] = mapped_column(String(36), index=True)
    channel: Mapped[str] = mapped_column(String(32))
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class MarketSnapshot(Base):
    """Raw market data snapshot for training the probability model.

    Snapshots are taken each poll cycle (when price changes > threshold).
    The `outcome` field is backfilled when the market resolves.
    `feature_version` tracks which feature extraction code produced the features JSON.
    """

    __tablename__ = "market_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[Source] = mapped_column(Enum(Source))
    contract_id: Mapped[str] = mapped_column(String(256))
    title: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(128), default="unknown")

    features: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    feature_version: Mapped[str] = mapped_column(String(32), default="0.0.0")

    # Outcome label — backfilled when market resolves (1.0 = YES, 0.0 = NO)
    outcome: Mapped[float | None] = mapped_column(Float, nullable=True)

    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_snapshot_lookup", "contract_id", "snapshot_at"),)


class PaperTrade(Base):
    """A simulated trade recorded by the paper trader.

    Tracks entry, exit, and P&L for backtesting strategy performance
    without risking real capital.
    """

    __tablename__ = "paper_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[Source] = mapped_column(Enum(Source))
    contract_id: Mapped[str] = mapped_column(String(256), index=True)
    direction: Mapped[Direction] = mapped_column(Enum(Direction))
    strategy_name: Mapped[str] = mapped_column(String(64))
    entry_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    model_probability: Mapped[float] = mapped_column(Float)
    expected_value: Mapped[float] = mapped_column(Float)
    entered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[float | None] = mapped_column(Float, nullable=True)
