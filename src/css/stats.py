"""Multi-seed aggregation with Student-t confidence intervals."""
import numpy as np
from scipy.stats import t as _t_dist

from . import config as cfg
from .engine import run_sim


def ci_mult(n_seeds, conf=0.95):
    if n_seeds <= 1:
        return float("nan")
    return float(_t_dist.ppf(1 - (1 - conf) / 2, df=n_seeds - 1))


CI = ci_mult(len(cfg.SEEDS))


def _agg(runs):
    nr = len(runs)
    def m(k): return float(np.mean([r[k] for r in runs]))
    def s(k):
        a = np.array([r[k] for r in runs], dtype=float)
        return float(a.std(ddof=1) / np.sqrt(nr)) if nr > 1 else 0.0
    return m, s


def run_multi(policy, seeds=None, psi_override="mgn", **kwargs):
    seeds = seeds or cfg.SEEDS
    kwargs.pop("seed", None)
    runs = [run_sim(policy, seed=s, psi_override=psi_override, **kwargs) for s in seeds]
    m, s = _agg(runs)
    r0 = runs[0]
    return dict(mean_power=m("mean_power"), sem_power=s("mean_power"),
                mean_resp=m("mean_resp"),   sem_resp=s("mean_resp"),
                p95_resp=m("p95_resp"),     sem_p95=s("p95_resp"),
                p99_resp=m("p99_resp"),     sem_p99=s("p99_resp"),
                psi=r0["psi"], mumax=r0["mumax"], n=r0["n"], rho=r0["rho"],
                cov=r0["cov"], ET_local=r0["ET_local"], n_seeds=len(runs), runs=runs)


def run_multi_ws(policy, seeds=None, psi_override="mgn", **kwargs):
    seeds = seeds or cfg.SEEDS
    kwargs.pop("seed", None)
    runs = [run_sim(policy, seed=s, psi_override=psi_override, record_ws=True, **kwargs)
            for s in seeds]
    m, s = _agg(runs)
    return dict(mean_wait=m("mean_wait"),       sem_wait=s("mean_wait"),
                mean_service=m("mean_service"), sem_service=s("mean_service"),
                mean_resp=m("mean_resp"),       sem_resp=s("mean_resp"),
                p95_wait=m("p95_wait"),         sem_p95_wait=s("p95_wait"),
                p95_service=m("p95_service"),   sem_p95_service=s("p95_service"),
                p95_resp=m("p95_resp"),         sem_p95_resp=s("p95_resp"),
                size_p95=m("size_p95"),
                large_job_wait=m("large_job_wait"),       sem_large_wait=s("large_job_wait"),
                large_job_service=m("large_job_service"), sem_large_service=s("large_job_service"),
                large_job_resp=m("large_job_resp"),       sem_large_resp=s("large_job_resp"),
                large_job_frac=m("large_job_frac"),
                max_gap=max(r["max_gap"] for r in runs))
