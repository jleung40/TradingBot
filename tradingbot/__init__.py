"""TradingBot quantitative strategy package."""

from tradingbot.backtest import backtest_portfolio, compute_performance_metrics
from tradingbot.data import download_adjusted_close
from tradingbot.strategy import StrategyConfig, generate_target_weights

__all__ = [
    "StrategyConfig",
    "download_adjusted_close",
    "generate_target_weights",
    "backtest_portfolio",
    "compute_performance_metrics",
]
