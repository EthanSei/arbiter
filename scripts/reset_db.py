"""Drop and recreate all tables.

Usage:
    python scripts/reset_db.py          # uses DATABASE_URL from .env
    DATABASE_URL=sqlite+aiosqlite:///local.db python scripts/reset_db.py
"""

import asyncio
import os
import sys

from sqlalchemy.ext.asyncio import create_async_engine

from arbiter.db.models import Base


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:
            from arbiter.config import Settings

            url = Settings().database_url  # type: ignore[attr-defined]
        except Exception:
            pass

    if not url:
        print("ERROR: DATABASE_URL not set and not found in .env", file=sys.stderr)
        sys.exit(1)

    engine = create_async_engine(url)

    async with engine.begin() as conn:
        print(f"Dropping all tables on {engine.url} ...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating all tables ...")
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
