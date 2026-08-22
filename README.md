# Dynamic Statistical Arbitrage using Kalman Filtering

**Author: Udit Chauhan**

A pairs-trading backtester with two hedge-ratio approaches:

- **Static**: OLS-regression hedge ratio + Engle-Granger cointegration test
- **Kalman**: time-varying hedge ratio estimated with a Kalman filter

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --ticker-x HDFCBANK.NS --ticker-y KOTAKBANK.NS --strategy both
python -m src.main --csv data/raw/HDFCBANK.NS_KOTAKBANK.NS.csv --ticker-x HDFCBANK.NS --ticker-y KOTAKBANK.NS
```

Plots (cumulative PnL, z-score, and — for the Kalman strategy — the dynamic
hedge ratio) are saved as PNGs under `plots/`.

## Project layout

```
src/
    __init__.py         # package metadata (__author__ = "Udit Chauhan")
    config.py            # all tunable defaults in one place
    data_loader.py         # download (yfinance) / load pair price CSVs
    cointegration.py       # cointegration test, OLS hedge ratio, spread, rolling z-score
    kalman_filter.py       # dynamic hedge ratio via Kalman filter
    signals.py               # z-score -> position (-1/0/+1)
    backtest.py               # PnL simulation + performance metrics
    main.py                    # CLI: ties everything together, saves plots
requirements.txt
```

---
© Udit Chauhan
