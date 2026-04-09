from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest import backtest_portfolio
from tradingbot.strategy import StrategyConfig, generate_target_weights


def _make_synthetic_prices(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    base = np.linspace(100.0, 130.0, rows)
    wave = 1.5 * np.sin(np.linspace(0, 20, rows))
    shock = np.zeros(rows)
    shock[150:155] = -4.0
    shock[200:205] = -3.0

    a = base + wave + shock
    b = base * 0.9 + 0.8 * np.cos(np.linspace(0, 18, rows))
    c = base * 1.1 + 0.5 * np.sin(np.linspace(0, 25, rows))

    return pd.DataFrame({"ETF_A": a, "ETF_B": b, "ETF_C": c}, index=idx)


def test_weights_are_bounded_and_normalized() -> None:
    prices = _make_synthetic_prices()
    cfg = StrategyConfig(
        reversion_lookback=20,
        momentum_lookback=40,
        trend_lookback=80,
        vol_lookback=20,
        max_positions=2,
        min_alpha=0.0,
        max_weight=0.7,
    )

    weights = generate_target_weights(prices, cfg)
    row_sums = weights.sum(axis=1)

    assert (weights >= 0).all().all()
    assert (weights <= cfg.max_weight + 1e-9).all().all()
    assert (row_sums <= 1.0 + 1e-9).all()


def test_backtest_outputs_have_expected_shape() -> None:
    prices = _make_synthetic_prices()
    cfg = StrategyConfig()
    bt = backtest_portfolio(prices, cfg, cost_bps=5.0)

    assert len(bt["net_returns"]) == len(prices)
    assert len(bt["equity_curve"]) == len(prices)
    assert list(bt["weights"].columns) == list(prices.columns)
