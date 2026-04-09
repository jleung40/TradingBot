"""Backtest engine and performance analytics."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradingbot.strategy import StrategyConfig, generate_target_weights


def backtest_portfolio(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    cost_bps: float = 5.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run a daily ETF backtest with 1-day execution lag and turnover costs."""
    if prices.empty:
        raise ValueError("Price data is empty.")

    weights_target = generate_target_weights(prices, cfg)
    returns = prices.pct_change().fillna(0.0)

    # Target generated at t is traded for t+1 return.
    weights_live = weights_target.shift(1).fillna(0.0)
    gross_returns = (weights_live * returns).sum(axis=1)

    turnover = weights_target.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = weights_target.iloc[0].abs().sum()
    costs = turnover * (cost_bps / 10_000.0)

    net_returns = gross_returns - costs
    equity_curve = (1.0 + net_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    return {
        "weights": weights_target,
        "weights_live": weights_live,
        "asset_returns": returns,
        "gross_returns": gross_returns,
        "costs": costs,
        "net_returns": net_returns,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "turnover": turnover,
    }


def compute_performance_metrics(net_returns: pd.Series, drawdown: pd.Series) -> dict[str, float]:
    """Compute portfolio-level performance statistics."""
    clean_ret = net_returns.dropna()
    if clean_ret.empty:
        raise ValueError("No valid returns to evaluate.")

    ann_factor = 252.0
    avg_daily = clean_ret.mean()
    vol_daily = clean_ret.std(ddof=0)
    ann_return = (1.0 + clean_ret).prod() ** (ann_factor / len(clean_ret)) - 1.0
    ann_vol = vol_daily * np.sqrt(ann_factor)
    sharpe = (avg_daily / vol_daily) * np.sqrt(ann_factor) if vol_daily > 0 else 0.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0
    win_rate = float((clean_ret > 0).mean())

    return {
        "cagr": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": win_rate,
        "avg_daily_return": float(avg_daily),
    }


def save_backtest_outputs(
    output_dir: str | Path,
    cfg: StrategyConfig,
    backtest: dict[str, pd.DataFrame | pd.Series],
    summary: dict[str, float],
) -> None:
    """Persist backtest artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.Series(summary).to_json(out / "summary.json", indent=2)
    pd.Series(asdict(cfg)).to_json(out / "config.json", indent=2)

    cast_series: dict[str, pd.Series] = {
        "portfolio_returns": backtest["net_returns"],  # type: ignore[index]
        "equity_curve": backtest["equity_curve"],  # type: ignore[index]
        "drawdown": backtest["drawdown"],  # type: ignore[index]
        "turnover": backtest["turnover"],  # type: ignore[index]
    }

    for name, series in cast_series.items():
        series.to_csv(out / f"{name}.csv", header=[name])

    cast_df: pd.DataFrame = backtest["weights"]  # type: ignore[assignment]
    cast_df.to_csv(out / "weights.csv")


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print core performance statistics."""
    print(json.dumps(summary, indent=2))
