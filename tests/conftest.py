"""Shared test fixtures."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.db.models import Base


@pytest.fixture
async def db_session():
    """Provide an isolated in-memory SQLite session per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = factory()
    yield session
    await session.rollback()
    await session.close()
    await engine.dispose()
