"""Download and load close-price data for a pair of tickers.

Author: Udit Chauhan
"""

import os
import pandas as pd
import yfinance as yf
from . import config


def download_pair(ticker_x: str, ticker_y: str, start: str = config.DEFAULT_START,
                   end: str = config.DEFAULT_END, save: bool = True) -> pd.DataFrame:
    """Download close prices for two tickers and return a 2-column DataFrame.

    Columns are named exactly `ticker_x` and `ticker_y`.
    """
    raw = yf.download([ticker_x, ticker_y], start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][[ticker_x, ticker_y]].copy()
    else:
        # Single-ticker edge case shouldn't happen here since we pass two tickers,
        # but guard anyway.
        close = raw[["Close"]].copy()
        close.columns = [ticker_x]

    close.columns = [ticker_x, ticker_y]
    close = close.dropna()

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        path = os.path.join(config.DATA_DIR, f"{ticker_x}_{ticker_y}.csv")
        close.to_csv(path)
        print(f"Saved {len(close)} rows to {path}")

    return close


def load_pair_csv(path: str) -> pd.DataFrame:
    """Load a previously-saved 2-column close-price CSV (see download_pair)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.shape[1] != 2:
        raise ValueError(
            f"Expected exactly 2 price columns in {path}, found {df.shape[1]}: {list(df.columns)}"
        )
    return df.dropna().astype(float)
