# TradingBot

Daily ETF quantitative trading model that combines:

- **Mean reversion** as the alpha source.
- **Momentum** as a **trade filter only** (not an alpha signal).

The default implementation is long-only, rebalanced daily, and includes:

- Historical data download (Yahoo Finance).
- Signal generation with momentum gating.
- Portfolio construction with risk controls.
- Backtesting with transaction costs.
- Performance metrics and equity curve output.

## Strategy Overview

### 1) Alpha: Mean Reversion
For each ETF:

1. Compute a short lookback z-score of price vs. rolling mean:
   - `z_t = (price_t - mean(price, L)) / std(price, L)`
2. Mean-reversion alpha is:
   - `alpha_t = -z_t`

So oversold ETFs (negative z-score) get positive alpha.

### 2) Filter: Momentum Regime (non-alpha)
An ETF is tradable only when both conditions are true:

- `momentum_lookback` day return is positive.
- Price is above `trend_lookback` day moving average.

This filter gates entries but does not rank assets by momentum.

### 3) Portfolio Construction

- Rank ETFs by mean-reversion alpha among momentum-eligible names.
- Select top `max_positions`.
- Weight selected ETFs by inverse recent volatility.
- Normalize and apply a per-asset weight cap.
- Rebalance daily.

### 4) Backtest Conventions

- Uses adjusted close daily data.
- Signals at day `t` are executed for returns from `t+1` onward (1-day lag).
- Includes linear transaction costs in bps based on daily turnover.

## Repository Layout

```text
tradingbot/
  __init__.py
  data.py
  strategy.py
  backtest.py
run_backtest.py
requirements.txt
tests/
  test_strategy.py
```

## Quick Start

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Run a backtest:

```bash
python run_backtest.py \
  --symbols SPY QQQ IWM EFA EEM TLT GLD XLF XLK XLE \
  --start 2012-01-01 \
  --end 2025-12-31 \
  --output-dir outputs
```

3) Inspect outputs in `outputs/`:

- `portfolio_returns.csv`
- `weights.csv`
- `equity_curve.csv`
- `summary.json`
- `equity_curve.png`

## Configurable Parameters

Key flags in `run_backtest.py`:

- `--reversion-lookback` (default: `20`)
- `--momentum-lookback` (default: `126`)
- `--trend-lookback` (default: `200`)
- `--vol-lookback` (default: `20`)
- `--max-positions` (default: `3`)
- `--min-alpha` (default: `0.0`)
- `--max-weight` (default: `0.4`)
- `--cost-bps` (default: `5.0`)

## Notes

- This is a research model, not investment advice.
- Real execution requires robust slippage modeling, data quality checks,
  and portfolio/risk controls beyond this baseline.
