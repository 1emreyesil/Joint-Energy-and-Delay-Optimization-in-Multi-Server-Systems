"""Event-driven multi-server simulator (one engine for all thesis experiments)."""
from collections import deque

import numpy as np

from . import config as cfg
from .psi import psi_mg1, psi_mgn
from .workloads import get_arrival_sequences


def _argmin_random(arr, rng):
    candidates = np.where(arr == arr.min())[0]
    return int(candidates[rng.randint(len(candidates))])


def _resolve_psi(psi_override, n, rho, cov):
    if psi_override is None or psi_override == "mgn":
        return psi_mgn(n, rho, cov)
    if psi_override == "mg1":
        return psi_mg1(rho, cov)
    return float(psi_override)


def _dvfs_dispatch(U, C, mumax, ET_local, pen_speed, pen_phi, rng):
    """Phi-min routing + closed-form speed law. Returns (j, mu_vec); U is not modified."""
    mu_base  = mumax * np.minimum(1, np.sqrt(np.maximum(U, 0) / (3 * pen_speed)))
    phi_base = U * (-mu_base * ET_local) + pen_phi * ET_local * (mu_base / mumax)**3
    U_plus   = U + C
    mu_plus  = mumax * np.minimum(1, np.sqrt(U_plus / (3 * pen_speed)))
    phi_plus = U_plus * (C - mu_plus * ET_local) + pen_phi * ET_local * (mu_plus / mumax)**3
    Phiv = np.sum(phi_base) - phi_base + phi_plus
    j = _argmin_random(Phiv, rng)
    mu_vec = mu_base.copy()
    mu_vec[j] = mu_plus[j]
    return j, mu_vec


def _ws_summary(wait, service, sojourn, sizes):
    wt, st, sj, cs = map(np.asarray, (wait, service, sojourn, sizes))
    if len(cs):
        size_p95 = float(np.percentile(cs, 95))
        mask = cs >= size_p95
        any_ = mask.sum() > 0
        large = dict(
            large_job_wait=float(wt[mask].mean()) if any_ else 0.0,
            large_job_service=float(st[mask].mean()) if any_ else 0.0,
            large_job_resp=float(sj[mask].mean()) if any_ else 0.0,
            large_job_frac=float(mask.sum() / len(cs)),
        )
    else:
        size_p95 = 0.0
        large = dict(large_job_wait=0.0, large_job_service=0.0,
                     large_job_resp=0.0, large_job_frac=0.0)
    pct = lambda a: float(np.percentile(a, 95)) if len(a) else 0.0
    return dict(
        mean_wait=float(wt.mean()) if len(wt) else 0.0,
        mean_service=float(st.mean()) if len(st) else 0.0,
        p95_wait=pct(wt), p95_service=pct(st),
        size_p95=size_p95, **large,
        max_gap=float(np.max(np.abs((wt + st) - sj))) if len(sj) else 0.0,
    )


def run_sim(policy, kappa=None, n=None, rho=None,
            nt=None, warmup=None, seed=None,
            psi_override=None, cov_override=None,
            mumax_override=None, ET_override=None,
            T_seq_ext=None, C_seq_ext=None,
            *, record_ws=False, trace_len=0, record_end=False):
    """policy: 'simple' (DVFS), 'lwl', 'jsq', 'random'."""
    kappa  = cfg.KAPPA  if kappa  is None else kappa
    n      = cfg.N      if n      is None else n
    rho    = cfg.RHO    if rho    is None else rho
    nt     = cfg.NT     if nt     is None else nt
    warmup = cfg.WARMUP if warmup is None else warmup
    seed   = cfg.SEED   if seed   is None else seed
    cov    = cfg.CoVC   if cov_override is None else cov_override

    if mumax_override is not None:
        mumax    = float(mumax_override)
        ET_local = ET_override if ET_override is not None else cfg.EC / (n * mumax * rho)
    else:
        ET_local = ET_override if ET_override is not None else cfg.ET
        mumax    = cfg.EC / (n * rho * ET_local)
    psi = _resolve_psi(psi_override, n, rho, cov)

    external = T_seq_ext is not None and C_seq_ext is not None
    if external:
        T_seq, C_seq = T_seq_ext, C_seq_ext
        nt = len(T_seq)
        warmup = min(warmup, nt // 5)
    else:
        T_seq, C_seq = get_arrival_sequences(nt, cov, seed, ET_local=ET_local)

    EC_used   = float(np.mean(C_seq)) if external else cfg.EC
    pen_speed = kappa * psi * EC_used
    pen_phi   = kappa * psi * EC_used * mumax

    U               = np.zeros(n)
    total_proc_work = np.zeros(n)
    queues          = [deque() for _ in range(n)]
    qlen            = np.zeros(n, dtype=np.int64)
    pow_rec         = np.zeros(nt - warmup)
    sojourn         = []
    absolute_time   = 0.0

    rng_disp = np.random.RandomState(seed + 999)
    rng_rand = np.random.RandomState(seed + 1999)

    if record_ws:
        prev_finish = np.zeros(n)
        wait, service, sizes = [], [], []
    tl = int(min(trace_len, max(0, nt - warmup)))
    if trace_len:
        U_rec, mu_rec = np.zeros((tl, n)), np.zeros((tl, n))
        busy, speed_work, total_time = np.zeros(n), np.zeros(n), 0.0

    for tt in range(nt):
        T, C = T_seq[tt], C_seq[tt]

        if policy == "simple":
            j, mu_vec = _dvfs_dispatch(U, C, mumax, ET_local, pen_speed, pen_phi, rng_disp)
            U[j] += C
        else:
            if policy == "lwl":
                j = _argmin_random(U, rng_disp)
            elif policy == "jsq":
                j = _argmin_random(qlen, rng_disp)
            elif policy == "random":
                j = int(rng_rand.randint(0, n))
            else:
                raise ValueError(f"unknown policy: {policy!r}")
            U[j] += C
            mu_vec = np.where(U > 0, mumax, 0.0)

        queues[j].append((absolute_time, total_proc_work[j] + U[j], C))
        qlen[j] += 1

        safe_mu = np.where(mu_vec > 0, mu_vec, 1e-15)
        active  = np.where(mu_vec > 0, np.minimum(T, U / safe_mu), 0.0)
        energy  = n * cfg.PMIN * T + (cfg.PMAX - cfg.PMIN) * np.sum((mu_vec / mumax)**3 * active)

        cleared          = np.maximum(mu_vec, 0.0) * active
        total_proc_work += cleared
        U                = np.maximum(0.0, U - cleared)

        post = tt >= warmup
        for s in np.flatnonzero(qlen):
            q      = queues[s]
            wstart = total_proc_work[s] - cleared[s]
            tol    = 1e-9 * max(1.0, abs(total_proc_work[s]))
            while q and q[0][1] <= total_proc_work[s] + tol:
                arr_time, threshold, size = q.popleft()
                qlen[s] -= 1
                fin = (absolute_time + min(active[s], (threshold - wstart) / mu_vec[s])
                       if mu_vec[s] > 0 else absolute_time)
                if record_ws:
                    start = max(arr_time, prev_finish[s])
                    prev_finish[s] = fin
                if post:
                    sojourn.append(fin - arr_time)
                    if record_ws:
                        wait.append(start - arr_time)
                        service.append(fin - start)
                        sizes.append(size)

        if post:
            pow_rec[tt - warmup] = energy / T
            if trace_len:
                busy       += active
                speed_work += mu_vec * active
                total_time += T
                k = tt - warmup
                if k < tl:
                    U_rec[k], mu_rec[k] = U, mu_vec

        absolute_time += T

    T_post     = T_seq[warmup:]
    mean_power = np.sum(pow_rec * T_post) / np.sum(T_post)
    st = np.asarray(sojourn)
    out = dict(mean_power=mean_power,
               mean_resp=float(st.mean()) if len(st) else 0.0,
               p95_resp=float(np.percentile(st, 95)) if len(st) else 0.0,
               p99_resp=float(np.percentile(st, 99)) if len(st) else 0.0,
               mumax=mumax, psi=psi, n=n, rho=rho, cov=cov, ET_local=ET_local)

    if record_ws:
        out.update(_ws_summary(wait, service, sojourn, sizes))
    if trace_len:
        out.update(U_rec=U_rec, mu_rec=mu_rec, trace_len=tl,
                   utilization_per_server=busy / max(total_time, 1e-15),
                   mu_avg_busy_per_server=speed_work / np.maximum(busy, 1e-15))
    if record_end:
        out.update(backlog_end=float(U.sum()), work_done=float(total_proc_work.sum()),
                   n_done=len(sojourn), T_total=float(absolute_time))
    return out
