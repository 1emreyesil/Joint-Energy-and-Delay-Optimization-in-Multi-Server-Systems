"""Run thesis experiments and save raw results to results/ (JSON, NPZ for arrays).

Examples:
  python experiments/run_experiments.py --exp trace
  python experiments/run_experiments.py --exp kappa rho n
  python experiments/run_experiments.py --exp google --google-path /path/Cell_a.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from css import config as cfg                                   # noqa: E402
from css.engine import run_sim                                  # noqa: E402
from css.psi import erlang_c, psi_mg1, psi_mgn, psi_mgn_legacy  # noqa: E402
from css.stats import run_multi, run_multi_ws                   # noqa: E402
from css.workloads import get_arrival_sequences, load_google_trace  # noqa: E402

OUT = ROOT / "results"
NT_FAST, WU_FAST = 100_000, 5_000
KAPPA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0,
              1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 7.0, 10.0]      # union of cells 16/18/32/48
KAPPA_GOOGLE = [0.5, 0.7, 1.0, 1.5, 2.0]
GOOGLE_COV_C = 175.29                                          # thesis value, psi table only


# ---------- I/O ----------
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if k != "runs"}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save(name, data):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(_clean(data), indent=1))
    print(f"  -> {path.relative_to(ROOT)}")


# ---------- experiments ----------
def exp_benchmark(args):                      # cells 11, 45
    for n in (cfg.N, 20):
        save(f"benchmark_n{n}", {p: run_sim(p, n=n) for p in cfg.POLICIES})


def exp_psi(args):                            # cells 9, 14
    runs = dict(mgn=run_multi("simple", psi_override="mgn"),
                mg1=run_multi("simple", psi_override="mg1"),
                lwl=run_multi("lwl"))
    points = ([(n, 0.7, 2.0) for n in (10, 20, 50, 100, 200, 500, 1000)]
              + [(100, r, 2.0) for r in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]
              + [(100, r, c) for r in (0.6, 0.7, 0.8) for c in (0.5, 2.0, 5.0)]
              + [(100, 0.7, GOOGLE_COV_C)])
    table = [dict(n=n, rho=r, cov=c, erlang_c=erlang_c(n, r), psi_mg1=psi_mg1(r, c),
                  psi_mgn=psi_mgn(n, r, c), psi_legacy=psi_mgn_legacy(n, r, c))
             for n, r, c in points]
    save("psi", dict(runs=runs, table=table))


def exp_kappa(args):                          # cells 16, 18, 32, 48 (merged)
    nt, wu = (cfg.NT, cfg.WARMUP) if args.full else (NT_FAST, WU_FAST)
    dvfs = []
    for k in KAPPA_GRID:
        r = run_multi("simple", kappa=float(k), nt=nt, warmup=wu)
        r["kappa"] = float(k)
        dvfs.append(r)
        print(f"  kappa={k:g}  P={r['mean_power']:.0f}W  MRT={r['mean_resp']:.1f}s")
    base = {p: run_multi(p, nt=nt, warmup=wu) for p in ("lwl", "jsq", "random")}
    save("kappa", dict(dvfs=dvfs, baselines=base, nt=nt, warmup=wu))


def exp_rho(args):                            # cell 20 (fixed hardware, mumax at rho=0.7)
    mumax = cfg.EC / (cfg.N * 0.7 * cfg.ET)
    total_T, warm_T = cfg.NT_SWEEP * cfg.ET, cfg.WARMUP_SWEEP * cfg.ET
    res = {p: [] for p in cfg.POLICIES}
    for rho in cfg.RHO_VEC_DEFAULT:
        ET_local = cfg.EC / (cfg.N * mumax * rho)
        nt = max(1000, int(round(total_T / ET_local)))
        wu = max(100, int(round(warm_T / ET_local)))
        if wu >= nt:
            wu = nt // 5
        for p in cfg.POLICIES:
            res[p].append(run_multi(p, n=cfg.N, rho=rho, mumax_override=mumax,
                                    ET_override=ET_local, nt=nt, warmup=wu))
        print(f"  rho={rho:.2f} done")
    save("rho", dict(rho=cfg.RHO_VEC_DEFAULT, **res))


def exp_n(args):                              # cell 23
    rho = 0.7
    mumax = cfg.EC / (cfg.N * rho * cfg.ET)
    total_T, warm_T = cfg.NT_SWEEP * cfg.ET, cfg.WARMUP_SWEEP * cfg.ET
    res = {p: [] for p in cfg.POLICIES}
    for n_val in cfg.N_VEC_DEFAULT:
        ni = int(n_val)
        ET_local = cfg.EC / (ni * mumax * rho)
        nt = min(int(round(total_T / ET_local)), 2_000_000)
        wu = min(int(round(warm_T / ET_local)), 500_000)
        if wu >= nt:
            wu = nt // 5
        for p in cfg.POLICIES:
            res[p].append(run_multi(p, n=ni, rho=rho, mumax_override=mumax,
                                    ET_override=ET_local, nt=nt, warmup=wu))
        print(f"  n={ni} done")
    save("n", dict(n=cfg.N_VEC_DEFAULT, **res))


def _scale(cov):
    return 1.0 if cov <= 2.0 else min(4, 1 + (cov - 2.0))


def exp_cov(args):                            # cells 26, 28 (hardware rescaled with rho)
    pols, out = ("simple", "lwl"), {}
    scenarios = [("rho=0.6", 0.6, cfg.COV_VEC_DEFAULT, cfg.NT_SWEEP, cfg.WARMUP_SWEEP),
                 ("rho=0.8", 0.8, cfg.COV_VEC_DEFAULT, cfg.NT_SWEEP, cfg.WARMUP_SWEEP),
                 ("rho=0.7", 0.7, np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
                  100_000, 1_000)]
    for label, rho, covs, nt0, wu0 in scenarios:
        out[label] = dict(cov=covs, **{p: [] for p in pols})
        for c in covs:
            s = _scale(c)
            for p in pols:
                out[label][p].append(run_multi(p, n=100, rho=rho, nt=int(nt0 * s),
                                               warmup=int(wu0 * s), cov_override=c))
        print(f"  {label} done")
    save("cov", out)


def exp_ws(args):                             # cell 30
    rhos = (0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95)
    mumax = cfg.EC / (cfg.N * 0.7 * cfg.ET)
    total_T, warm_T = 100_000 * cfg.ET, 1_000 * cfg.ET
    res = {p: [] for p in cfg.POLICIES}
    for rho in rhos:
        ET_local = cfg.EC / (cfg.N * mumax * rho)
        nt = max(1000, int(round(total_T / ET_local)))
        wu = max(100, int(round(warm_T / ET_local)))
        if wu >= nt:
            wu = nt // 5
        for p in cfg.POLICIES:
            res[p].append(run_multi_ws(p, n=cfg.N, rho=rho, mumax_override=mumax,
                                       ET_override=ET_local, nt=nt, warmup=wu))
        print(f"  rho={rho} done")
    save("ws", dict(rho=rhos, **res))


def exp_trace(args):                          # cell 35
    arrays = {}
    for p in ("simple", "lwl"):
        r = run_sim(p, kappa=cfg.KAPPA, n=cfg.N, rho=cfg.RHO, nt=100_000,
                    warmup=5_000, seed=cfg.SEED, trace_len=2000)
        for k in ("U_rec", "mu_rec", "utilization_per_server", "mu_avg_busy_per_server"):
            arrays[f"{p}_{k}"] = r[k]
        arrays[f"{p}_mumax"] = np.array(r["mumax"])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "trace.npz", **arrays)
    print("  -> results/trace.npz")


def _unfinished(T_seq, C_seq, **kw):
    rows = {p: run_sim(p, kappa=cfg.KAPPA, n=cfg.N, rho=cfg.RHO, warmup=0,
                       T_seq_ext=T_seq, C_seq_ext=C_seq, record_end=True, **kw)
            for p in cfg.POLICIES}
    return dict(W_in=float(np.sum(C_seq)), J_in=len(C_seq), policies=rows)


def exp_unfinished(args):                     # cell 43
    T, C = get_arrival_sequences(cfg.NT, cfg.CoVC, seed=cfg.SEEDS[0])
    save("unfinished_synthetic", _unfinished(T, C))


def exp_google(args):                         # cells 38-40, 42
    if not args.google_path:
        print("  skipped: --google-path not given")
        return
    T_r, C_r, ct, cc, EC_real, ET_real = load_google_trace(args.google_path)
    mumax = EC_real / (cfg.N * cfg.RHO * ET_real)
    kw = dict(mumax_override=mumax, ET_override=ET_real, cov_override=cc,
              T_seq_ext=T_r, C_seq_ext=C_r)

    bench = {p: run_sim(policy=p, **kw) for p in cfg.POLICIES}
    sweep = []
    for k in KAPPA_GOOGLE:
        r = run_sim(policy="simple", kappa=k, n=cfg.N, rho=cfg.RHO, **kw)
        r["kappa"] = k
        sweep.append(r)
    meta = dict(cov_t=ct, cov_c=cc, EC_real=EC_real, ET_real=ET_real, mumax=mumax)
    save("google", dict(meta=meta, benchmark=bench, kappa=sweep))

    unf_kw = {k: v for k, v in kw.items() if k not in ("T_seq_ext", "C_seq_ext")}
    save("unfinished_google", _unfinished(T_r, C_r, **unf_kw))

    df = pd.read_csv(args.google_path, usecols=["Job_ID", "Task_ID", "CPU"])
    agg = df.groupby("Job_ID").agg(CPU=("CPU", "sum"), n_tasks=("Task_ID", "count"))
    agg = agg[agg["CPU"] > 0]
    np.savez_compressed(OUT / "google_jobsize.npz",
                        C=agg["CPU"].values, n_tasks=agg["n_tasks"].values)
    print("  -> results/google_jobsize.npz")


EXPERIMENTS = dict(benchmark=exp_benchmark, psi=exp_psi, kappa=exp_kappa, rho=exp_rho,
                   n=exp_n, cov=exp_cov, ws=exp_ws, trace=exp_trace,
                   unfinished=exp_unfinished, google=exp_google)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"], choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--google-path")
    ap.add_argument("--full", action="store_true",
                    help="kappa sweep with NT/WARMUP instead of 100k/5k (thesis used 100k/5k)")
    args = ap.parse_args()
    names = list(EXPERIMENTS) if "all" in args.exp else args.exp
    for name in names:
        t0 = time.time()
        print(f"[{name}]")
        EXPERIMENTS[name](args)
        print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
