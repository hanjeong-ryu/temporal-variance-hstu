import numpy as np
import pandas as pd

from .data import DAY


def make_synthetic_log(n_users=500, n_events=120, n_items=300,
                       median_tau_days=1.0, sd_log_lambda=1.0,
                       t0=1_000_000_000, seed=0):
    # 활동률은 로그정규, 간격은 지수분포. sd_log_lambda=0 이면 참값 rho_k = 0
    rng = np.random.default_rng(seed)

    lam = np.exp(rng.normal(0, sd_log_lambda, n_users)) / (median_tau_days * DAY)
    gaps = rng.exponential(1.0 / lam[:, None], size=(n_users, n_events - 1))
    ts = np.cumsum(np.hstack([np.zeros((n_users, 1)), gaps]), axis=1)

    return pd.DataFrame({
        "userId": np.repeat(np.arange(1, n_users + 1), n_events).astype("int32"),
        "movieId": rng.integers(1, n_items + 1, n_users * n_events).astype("int32"),
        "timestamp": (t0 + ts.ravel()).astype("int64"),
    })
