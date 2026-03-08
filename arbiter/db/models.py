"""SQLAlchemy ORM models for opportunity tracking, alert logging, and training data."""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
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
    series_ticker: Mapped[str] = mapped_column(String(64), default="")

    features: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    feature_version: Mapped[str] = mapped_column(String(32), default="0.0.0")

    # Outcome label — backfilled when market resolves (1.0 = YES, 0.0 = NO)
    outcome: Mapped[float | None] = mapped_column(Float, nullable=True)

    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_snapshot_lookup", "contract_id", "snapshot_at"),
        Index("ix_snapshot_at", "snapshot_at"),
    )


class OrderBookSnapshot(Base):
    """Point-in-time snapshot of a contract's order book.

    Stores full bid/ask depth as JSON arrays for maker strategy analysis.
    Each entry: {"price": float, "quantity": int}.
    """

    __tablename__ = "order_book_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[Source] = mapped_column(Enum(Source))
    contract_id: Mapped[str] = mapped_column(String(256))
    series_ticker: Mapped[str] = mapped_column(String(64), default="")
    event_ticker: Mapped[str] = mapped_column(String(128), default="")
    bids: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    asks: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_ob_lookup", "contract_id", "snapshot_at"),
        Index("ix_ob_snapshot_at", "snapshot_at"),
    )


class CandlestickBar(Base):
    """OHLCV candlestick bar for a contract over a fixed time period.

    Prices are YES prices in [0, 1]. period_interval is in minutes
    (60 = 1h, 1440 = 1 day).
    """

    __tablename__ = "candlestick_bars"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[Source] = mapped_column(Enum(Source))
    contract_id: Mapped[str] = mapped_column(String(256))
    series_ticker: Mapped[str] = mapped_column(String(64), default="")
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_interval: Mapped[int] = mapped_column(Integer)  # minutes (60=1h, 1440=1d)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_candle_lookup", "contract_id", "period_start"),
        Index("ix_candle_period_start", "period_start"),
        UniqueConstraint("contract_id", "period_start", "period_interval", name="uq_candle_dedup"),
    )


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
