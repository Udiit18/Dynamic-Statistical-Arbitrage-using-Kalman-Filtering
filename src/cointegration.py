"""
Static (constant hedge ratio) pairs-trading tools.

Author: Udit Chauhan
"""

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def test_cointegration(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Engle-Granger cointegration test. Returns (p_value, test_statistic)."""
    score, pvalue, _ = coint(x, y)
    return pvalue, score


def static_hedge_ratio(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """OLS hedge ratio: y = beta * x + alpha. Returns (beta, alpha)."""
    beta, alpha = np.polyfit(x.astype(float), y.astype(float), 1)
    return beta, alpha


def compute_spread(x: pd.Series, y: pd.Series, hedge_ratio: float, intercept: float = 0.0) -> pd.Series:
    return y - (hedge_ratio * x + intercept)


def rolling_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of the spread (avoids full-sample look-ahead bias)."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std
