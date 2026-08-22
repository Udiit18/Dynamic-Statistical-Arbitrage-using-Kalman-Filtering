"""Dynamic (time-varying) hedge ratio estimation via a Kalman filter.

y_t = hedge_ratio_t * x_t + intercept_t + noise

The state [hedge_ratio, intercept] is allowed to drift slowly (controlled by
`delta`), and each new price observation updates the filter's estimate.

Author: Udit Chauhan
"""

from typing import Tuple
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


def estimate_dynamic_hedge_ratio(
    x: pd.Series,
    y: pd.Series,
    delta: float = 1e-5,
    observation_covariance: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (hedge_ratio, intercept) as time-indexed Series aligned to y.index."""
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([x.values, np.ones(len(x))]).T, axis=1)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=observation_covariance,
        transition_covariance=trans_cov,
    )

    y_values = y.values.astype(np.float64).reshape(-1, 1)
    state_means, _ = kf.filter(y_values)

    hedge_ratio = pd.Series(state_means[:, 0], index=y.index, name="hedge_ratio")
    intercept = pd.Series(state_means[:, 1], index=y.index, name="intercept")
    return hedge_ratio, intercept
