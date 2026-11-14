"""Calibration factor psi: M/G/1 (Pollaczek-Khinchine) vs M/G/n (Erlang-C)."""
import numpy as np
from scipy.special import gammaln


def erlang_c(n, rho):
    """Erlang-C waiting probability for M/M/n, computed in log-space."""
    log_a     = np.log(n * rho)
    log_terms = np.array([k * log_a - gammaln(k + 1) for k in range(n)])
    log_max   = log_terms.max()
    sum_term  = np.exp(log_terms - log_max).sum()
    last_term = np.exp(n * log_a - gammaln(n + 1) - np.log(1 - rho) - log_max)
    return last_term / (sum_term + last_term)


def psi_mg1(rho, cov):
    return 1 + (rho / (1 - rho)) * (1 + cov**2) / 2


def psi_mgn(n, rho, cov):
    """Lee-Longton form; reduces to psi_mg1 at n=1 and to exact M/M/n at CoV=1."""
    return 1 + erlang_c(n, rho) / (n * (1 - rho)) * (1 + cov**2) / 2


def psi_mgn_legacy(n, rho, cov):
    """Earlier (incorrect) form, kept only for the Ch.4 comparison."""
    return 1 + erlang_c(n, rho) * (rho / (1 - rho)) * (1 + cov**2) / 2 
