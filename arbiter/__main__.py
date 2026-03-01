"""Allow running as `python -m arbiter`."""

import asyncio
import sys


def main() -> None:
    try:
        from arbiter.main import main as async_main  # noqa: PLC0415

        asyncio.run(async_main())
    except ImportError:
        print("arbiter: main entrypoint not yet implemented")
        sys.exit(1)


main()
