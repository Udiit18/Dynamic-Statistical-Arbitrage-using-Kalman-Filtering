"""CLI entry point: download data, run static and/or Kalman pairs-trading
strategies, print performance metrics, and save plots to disk.

Usage:
    python -m src.main --ticker-x HDFCBANK.NS --ticker-y KOTAKBANK.NS --strategy both
    python -m src.main --csv data/raw/HDFCBANK.NS_KOTAKBANK.NS.csv --ticker-x HDFCBANK.NS --ticker-y KOTAKBANK.NS

Author: Udit Chauhan
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; plots are saved to disk, not shown
import matplotlib.pyplot as plt

from . import backtest, cointegration, config, data_loader, kalman_filter, signals


def run_static_strategy(x, y, entry_z=config.ENTRY_Z, exit_z=config.EXIT_Z,
                         window=config.ROLLING_WINDOW, plot_dir=config.PLOT_DIR):
    pvalue, score = cointegration.test_cointegration(x, y)
    print(f"Cointegration test p-value: {pvalue:.5f}")
    if pvalue >= 0.05:
        print("  Warning: series may not be cointegrated (p >= 0.05). Results may be unreliable.")

    hedge_ratio, intercept = cointegration.static_hedge_ratio(x, y)
    print(f"Static hedge ratio (OLS): {hedge_ratio:.4f}, intercept: {intercept:.4f}")

    spread = cointegration.compute_spread(x, y, hedge_ratio, intercept)
    z = cointegration.rolling_zscore(spread, window=window)

    position = signals.generate_signals(z, entry_z, exit_z)
    pnl_df = backtest.compute_pnl(x, y, hedge_ratio, position, intercept)
    metrics = backtest.performance_metrics(pnl_df)

    _print_metrics("Static Hedge Ratio Strategy", metrics)
    _save_plots(pnl_df, z, hedge_ratio_series=None, plot_dir=plot_dir, prefix="static")
    return pnl_df, metrics


def run_kalman_strategy(x, y, entry_z=config.ENTRY_Z, exit_z=config.EXIT_Z,
                         window=config.ROLLING_WINDOW, plot_dir=config.PLOT_DIR):
    hedge_ratio, intercept = kalman_filter.estimate_dynamic_hedge_ratio(x, y)

    spread = y - (hedge_ratio * x + intercept)
    z = cointegration.rolling_zscore(spread, window=window)

    position = signals.generate_signals(z, entry_z, exit_z)
    pnl_df = backtest.compute_pnl(x, y, hedge_ratio, position, intercept)
    metrics = backtest.performance_metrics(pnl_df)

    _print_metrics("Kalman Filter Dynamic Hedge Ratio Strategy", metrics)
    _save_plots(pnl_df, z, hedge_ratio_series=hedge_ratio, plot_dir=plot_dir, prefix="kalman")
    return pnl_df, metrics


def _print_metrics(title, metrics):
    print(f"\n--- {title} ---")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}")
    print(f"Total PnL:       {metrics['total_pnl']:.2f}")
    print(f"Number of Trades:{metrics['num_trades']:>6d}")


def _save_plots(pnl_df, z, hedge_ratio_series, plot_dir, prefix):
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(pnl_df["cum_pnl"], label="Cumulative PnL")
    plt.title(f"{prefix.title()} Strategy: Cumulative PnL")
    plt.xlabel("Date"); plt.ylabel("PnL")
    plt.legend(); plt.grid(True); plt.tight_layout()
    path1 = os.path.join(plot_dir, f"{prefix}_cum_pnl.png")
    plt.savefig(path1, dpi=120)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(z, label="Z-Score of Spread")
    plt.axhline(2.0, color="red", linestyle="--", label="Short Entry (+2)")
    plt.axhline(-2.0, color="green", linestyle="--", label="Long Entry (-2)")
    plt.axhline(0.0, color="black", linewidth=1, label="Mean")
    plt.title(f"{prefix.title()} Strategy: Z-Score of Spread")
    plt.legend(); plt.grid(True); plt.tight_layout()
    path2 = os.path.join(plot_dir, f"{prefix}_zscore.png")
    plt.savefig(path2, dpi=120)
    plt.close()

    saved = [path1, path2]
    if hedge_ratio_series is not None:
        plt.figure(figsize=(14, 4))
        plt.plot(hedge_ratio_series, label="Dynamic Hedge Ratio")
        plt.title(f"{prefix.title()} Strategy: Hedge Ratio")
        plt.legend(); plt.grid(True); plt.tight_layout()
        path3 = os.path.join(plot_dir, f"{prefix}_hedge_ratio.png")
        plt.savefig(path3, dpi=120)
        plt.close()
        saved.append(path3)

    print(f"Saved plots: {', '.join(saved)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical arbitrage pairs-trading backtester")
    parser.add_argument("--ticker-x", default="V", help="First ticker (independent variable)")
    parser.add_argument("--ticker-y", default="MA", help="Second ticker (dependent variable)")
    parser.add_argument("--start", default=config.DEFAULT_START)
    parser.add_argument("--end", default=config.DEFAULT_END)
    parser.add_argument("--strategy", choices=["static", "kalman", "both"], default="both")
    parser.add_argument("--entry-z", type=float, default=config.ENTRY_Z)
    parser.add_argument("--exit-z", type=float, default=config.EXIT_Z)
    parser.add_argument("--window", type=int, default=config.ROLLING_WINDOW)
    parser.add_argument("--csv", default=None,
                         help="Path to a pre-downloaded 2-column CSV (see data_loader.download_pair). "
                              "Skips the network download if provided.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Statistical Arbitrage on Equity Pairs")
    print("by Udit Chauhan\n")

    if args.csv:
        df = data_loader.load_pair_csv(args.csv)
    else:
        df = data_loader.download_pair(args.ticker_x, args.ticker_y, args.start, args.end)

    x, y = df[args.ticker_x], df[args.ticker_y]

    if args.strategy in ("static", "both"):
        run_static_strategy(x, y, args.entry_z, args.exit_z, args.window)
    if args.strategy in ("kalman", "both"):
        run_kalman_strategy(x, y, args.entry_z, args.exit_z, args.window)


if __name__ == "__main__":
    main()
