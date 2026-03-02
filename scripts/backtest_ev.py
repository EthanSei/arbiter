"""Backtest EVStrategy on historical training data.

Loads a trained model, replays the test split of a training CSV,
simulates Kelly-sized trades, and reports P&L metrics.

Run with:
    python scripts/backtest_ev.py
    python scripts/backtest_ev.py --model models/arbiter_lgbm.pkl --data data/training.csv
    python scripts/backtest_ev.py --fee-rate 0.05 --kelly 0.10
"""

from __future__ import annotations

import argparse

from arbiter.training.historical import backtest_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest EVStrategy on training data")
    parser.add_argument(
        "--model", type=str, default="models/arbiter_lgbm.pkl", help="Trained model path"
    )
    parser.add_argument("--data", type=str, default="data/training.csv", help="Training CSV path")
    parser.add_argument(
        "--fee-rate", type=float, default=0.03, help="Execution cost (default: 0.03)"
    )
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Test split fraction (default: 0.15)"
    )
    parser.add_argument(
        "--ev-threshold", type=float, default=0.0, help="Min EV to trade (default: 0.0)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  EVStrategy Backtest")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Data:           {args.data}")
    print(f"  Fee rate:       {args.fee_rate:.2%}")
    print(f"  Kelly fraction: {args.kelly:.2f}")
    print(f"  Test fraction:  {args.test_fraction:.2%}")
    print(f"  EV threshold:   {args.ev_threshold:.4f}")
    print()

    metrics = backtest_from_csv(
        args.model,
        args.data,
        fee_rate=args.fee_rate,
        kelly_fraction=args.kelly,
        test_fraction=args.test_fraction,
        ev_threshold=args.ev_threshold,
    )

    print("  Results:")
    print(f"  {'─' * 40}")
    print(f"  Test samples:     {metrics['test_samples']:.0f}")
    print(f"  Trades:           {metrics['num_trades']:.0f}")
    print(f"  Win rate:         {metrics['win_rate']:.1%}")
    print(f"  Total P&L:        {metrics['total_pnl']:+.4f}")
    print(f"  Final bankroll:   {metrics['final_bankroll']:.4f}")
    print(f"  Max drawdown:     {metrics['max_drawdown']:.1%}")
    print(f"  Sharpe ratio:     {metrics['sharpe']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
