"""Central configuration / default parameters for the pairs-trading backtester.

Author: Udit Chauhan
"""

# Data
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-07-31"
DATA_DIR = "data/raw"

# Signal generation
ENTRY_Z = 2.0
EXIT_Z = 0.5
ROLLING_WINDOW = 20  # bars used for rolling mean/std of the spread

# Backtest / costs
STARTING_CAPITAL = 100_000
SLIPPAGE_PCT = 0.001  # 0.1% of spread value charged on each position change

# Output
PLOT_DIR = "plots"
