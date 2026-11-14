"""Synthetic (Poisson / Weibull) and Google Cluster Data workloads."""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import beta as beta_func, gamma as gamma_func

from . import config as cfg


def weibull_params(mean, cov):
    """Return (scale, shape) of a Weibull with the given mean and CoV."""
    d = (1.0 + cov**2) / 2.0
    lo, hi = 1e-3, 60.0
    for _ in range(300):
        x = (lo + hi) / 2.0
        if x / beta_func(1.0 / x, 1.0 / x) > d:
            lo = x
        else:
            hi = x
        if (hi - lo) / ((hi + lo) / 2 + 1e-30) < 1e-10:
            break
    return mean * x / gamma_func(1.0 / x), x


@lru_cache(maxsize=8)
def _arrivals(nt, cov, seed, ET_local):
    rng   = np.random.RandomState(seed)          # legacy RNG: keeps thesis numbers
    T_seq = -ET_local * np.log(rng.rand(nt))
    v_raw = np.maximum(1e-20, -np.log(rng.rand(nt)))
    a_w, b_w = weibull_params(cfg.EC, cov)
    C_seq = a_w * (v_raw ** (1.0 / b_w))
    T_seq.setflags(write=False)
    C_seq.setflags(write=False)
    return T_seq, C_seq


def get_arrival_sequences(nt, cov, seed, ET_local=None):
    ET_local = cfg.ET if ET_local is None else ET_local
    return _arrivals(int(nt), float(cov), int(seed), float(ET_local))


def load_google_trace(path, verbose=True):
    """Aggregate tasks to jobs; return (T_seq, C_seq, cov_t, cov_c, EC_real, ET_real)."""
    df = pd.read_csv(path, usecols=["Job_ID", "Arrival_Time", "CPU"])
    df_agg = (df.groupby("Job_ID")
                .agg(Arrival_Time=("Arrival_Time", "min"), CPU=("CPU", "sum"))
                .reset_index()
                .sort_values("Arrival_Time")
                .reset_index(drop=True))
    df_agg = df_agg[df_agg["CPU"] > 0]

    arrivals = df_agg["Arrival_Time"].values / 1000.0
    T_seq = np.maximum(np.diff(arrivals), 1e-9)
    C_seq = df_agg["CPU"].values[1:]   # NOTE: off-by-one, see Faz B

    EC_real, ET_real = float(C_seq.mean()), float(T_seq.mean())
    cov_t = T_seq.std() / T_seq.mean()
    cov_c = C_seq.std() / C_seq.mean()
    if verbose:
        print(f"Loaded {len(T_seq):,} jobs | T: mean={ET_real:.4f}s CoV={cov_t:.2f}"
              f" | C: mean={EC_real:.6f} CoV={cov_c:.2f}")
    return T_seq, C_seq, cov_t, cov_c, EC_real, ET_real
