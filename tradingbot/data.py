"""Data access helpers for ETF backtests."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


def download_adjusted_close(
    symbols: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices for a symbol list from Yahoo Finance."""
    symbol_list = list(symbols)
    if not symbol_list:
        raise ValueError("At least one symbol is required.")

    data = yf.download(
        tickers=symbol_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if "Adj Close" in data:
        prices = data["Adj Close"].copy()
    elif "Close" in data:
        prices = data["Close"].copy()
    else:
        raise ValueError("Expected Close or Adj Close columns in downloaded data.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbol_list[0])

    prices = prices.sort_index().dropna(how="all")
    prices = prices.ffill().dropna(how="all")
    return prices
