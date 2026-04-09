#!/usr/bin/env python3
"""Run ETF mean-reversion strategy with momentum filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from tradingbot.backtest import (
    backtest_portfolio,
    compute_performance_metrics,
    print_summary,
    save_backtest_outputs,
)
from tradingbot.data import download_adjusted_close
from tradingbot.strategy import StrategyConfig


DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "XLF",
    "XLK",
    "XLE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETF daily mean-reversion model with momentum filter."
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cost-bps", type=float, default=5.0)

    parser.add_argument("--reversion-lookback", type=int, default=20)
    parser.add_argument("--momentum-lookback", type=int, default=126)
    parser.add_argument("--trend-lookback", type=int, default=200)
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(
        reversion_lookback=args.reversion_lookback,
        momentum_lookback=args.momentum_lookback,
        trend_lookback=args.trend_lookback,
        vol_lookback=args.vol_lookback,
        max_positions=args.max_positions,
        min_alpha=args.min_alpha,
        max_weight=args.max_weight,
    )

    prices = download_adjusted_close(args.symbols, args.start, args.end)
    bt = backtest_portfolio(prices, cfg=cfg, cost_bps=args.cost_bps)
    summary = compute_performance_metrics(
        net_returns=bt["net_returns"],  # type: ignore[arg-type]
        drawdown=bt["drawdown"],  # type: ignore[arg-type]
    )

    save_backtest_outputs(
        output_dir=args.output_dir,
        cfg=cfg,
        backtest=bt,
        summary=summary,
    )

    out = Path(args.output_dir)
    eq = bt["equity_curve"]  # type: ignore[assignment]
    plt.figure(figsize=(10, 5))
    eq.plot(title="Equity Curve: Mean Reversion + Momentum Filter")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(out / "equity_curve.png", dpi=150)
    plt.close()

    print_summary(summary)
    print(f"Saved artifacts to: {out.resolve()}")


if __name__ == "__main__":
    main()
