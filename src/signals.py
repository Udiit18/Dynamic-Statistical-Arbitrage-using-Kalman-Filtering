"""Turn a z-score series into a spread position: -1 (short spread), 0 (flat),
or +1 (long spread).

Author: Udit Chauhan
"""

import pandas as pd


def generate_signals(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """Enter when |z| crosses entry_z, exit when |z| falls back below exit_z.
    entry_z must be greater than exit_z or the position will never hold.
    """
    if entry_z <= exit_z:
        raise ValueError(f"entry_z ({entry_z}) must be greater than exit_z ({exit_z})")

    position = pd.Series(0, index=zscore.index, dtype=int)
    in_trade = False
    current = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if pd.isna(z):
            position.iloc[i] = current
            continue

        if not in_trade:
            if z > entry_z:
                current = -1  # spread too high -> short the spread
                in_trade = True
            elif z < -entry_z:
                current = 1  # spread too low -> long the spread
                in_trade = True
        else:
            if abs(z) < exit_z:
                current = 0
                in_trade = False

        position.iloc[i] = current

    return position
