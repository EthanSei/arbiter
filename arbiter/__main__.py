"""Allow running as `python -m arbiter`."""

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="arbiter")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print alerts as JSON to stdout in addition to configured channels.",
    )
    args = parser.parse_args()

    try:
        from arbiter.main import main as async_main  # noqa: PLC0415

        asyncio.run(async_main(stdout_alerts=args.stdout))
    except ImportError:
        print("arbiter: main entrypoint not yet implemented")
        sys.exit(1)


main()
