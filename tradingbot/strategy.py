"""Signal generation and portfolio construction logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for mean-reversion + momentum-filter strategy."""

    reversion_lookback: int = 20
    momentum_lookback: int = 126
    trend_lookback: int = 200
    vol_lookback: int = 20
    max_positions: int = 3
    min_alpha: float = 0.0
    max_weight: float = 0.4


def _zscore(series: pd.DataFrame, lookback: int) -> pd.DataFrame:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std().replace(0.0, np.nan)
    return (series - mean) / std


def _momentum_filter(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    trailing_return = prices.pct_change(cfg.momentum_lookback)
    trend_ma = prices.rolling(cfg.trend_lookback).mean()
    return (trailing_return > 0.0) & (prices > trend_ma)


def _inverse_vol_weights(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    vol = returns.rolling(lookback).std() * np.sqrt(252.0)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    return inv_vol.replace([np.inf, -np.inf], np.nan)


def _normalize_with_cap(row: pd.Series, max_weight: float) -> pd.Series:
    if row.notna().sum() == 0:
        return row.fillna(0.0)

    w = row.fillna(0.0).clip(lower=0.0)
    total = w.sum()
    if total <= 0.0:
        return row.fillna(0.0)
    w = w / total

    # Iteratively cap and redistribute until stable.
    for _ in range(20):
        capped = w.clip(upper=max_weight)
        excess = (w - capped).clip(lower=0.0).sum()
        w = capped

        if excess <= 1e-12:
            break

        eligible = w < max_weight - 1e-12
        eligible_sum = w[eligible].sum()
        if eligible_sum <= 0.0:
            break
        w.loc[eligible] += excess * (w[eligible] / eligible_sum)

    final_sum = w.sum()
    if final_sum > 0.0:
        w = w / final_sum
    return w


def generate_target_weights(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Build daily target weights.

    Mean reversion provides alpha; momentum only gates tradability.
    """
    if prices.empty:
        raise ValueError("Price data is empty.")

    rets = prices.pct_change()
    alpha = -_zscore(prices, cfg.reversion_lookback)
    tradable = _momentum_filter(prices, cfg)
    inv_vol = _inverse_vol_weights(rets, cfg.vol_lookback)

    eligible_alpha = alpha.where(tradable)
    eligible_alpha = eligible_alpha.where(eligible_alpha >= cfg.min_alpha)

    # Cross-sectional rank by mean-reversion alpha.
    ranks = eligible_alpha.rank(axis=1, ascending=False, method="first")
    selected = ranks <= cfg.max_positions

    raw = inv_vol.where(selected)
    weights = raw.apply(lambda row: _normalize_with_cap(row, cfg.max_weight), axis=1)
    weights = weights.fillna(0.0)
    return weights
