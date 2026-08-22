"""PnL simulation and performance metrics.

Author: Udit Chauhan
"""

from typing import Dict

import numpy as np
import pandas as pd

from . import config


def compute_pnl(
    x: pd.Series,
    y: pd.Series,
    hedge_ratio,
    position: pd.Series,
    intercept=0.0,
    slippage_pct: float = config.SLIPPAGE_PCT,
) -> pd.DataFrame:
    """Simulate PnL of holding `position` units of the spread `y - hedge_ratio*x - intercept`.

    `hedge_ratio` / `intercept` may be a scalar (static strategy) or a
    time-indexed Series (Kalman strategy).
    """
    spread = y - (hedge_ratio * x + intercept)
    spread_change = spread.diff()

    gross_pnl = position.shift(1) * spread_change
    trade_size = position.diff().abs().fillna(0)
    costs = slippage_pct * spread.abs() * trade_size

    pnl = gross_pnl - costs
    cum_pnl = pnl.cumsum()

    return pd.DataFrame({
        "spread": spread,
        "position": position,
        "pnl": pnl,
        "cum_pnl": cum_pnl,
    })


def performance_metrics(pnl_df: pd.DataFrame) -> Dict[str, float]:
    returns = pnl_df["pnl"].dropna()
    std = returns.std()

    sharpe = float(np.sqrt(252) * returns.mean() / std) if std and not np.isnan(std) else float("nan")
    running_max = pnl_df["cum_pnl"].cummax()
    max_drawdown = float((pnl_df["cum_pnl"] - running_max).min())
    total_pnl = float(pnl_df["cum_pnl"].iloc[-1]) if len(pnl_df) else float("nan")
    num_trades = int((pnl_df["position"].diff().abs() > 0).sum())

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
    }
