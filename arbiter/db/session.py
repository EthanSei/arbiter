"""Async database engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")
engine = create_async_engine(
    settings.database_url,
    echo=False,
    # SQLite doesn't support connection pool tuning — skip those args for local dev.
    **({} if _is_sqlite else {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}),
)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    from arbiter.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
