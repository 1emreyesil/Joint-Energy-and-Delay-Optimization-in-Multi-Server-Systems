"""Build thesis figures from results/*.json|npz (no simulation is run here).

Examples:
  python experiments/make_figures.py            # every figure whose input exists
  python experiments/make_figures.py --fig rho n
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.ticker import ScalarFormatter         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from css.stats import CI                              # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"

POLICIES = ["simple", "lwl", "jsq", "random"]
LABELS   = dict(simple="DVFS", lwl="LWL", jsq="JSQ", random="Random")
COLORS   = dict(simple="steelblue", lwl="green", jsq="purple", random="orange")
MARKERS  = dict(simple="o", lwl="^", jsq="v", random="s")
WAIT_C, SERV_C = "indianred", "steelblue"

plt.rcParams.update({"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})


# ---------- helpers ----------
def load(name):
    path = RES / name
    if not path.exists():
        return None
    return np.load(path) if path.suffix == ".npz" else json.loads(path.read_text())


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  -> results/figures/{name}.png")


def band(ax, x, rows, key, se_key, p, **kw):
    v, se = col(rows, key), col(rows, se_key)
    ax.plot(x, v, f"{MARKERS[p]}-", color=COLORS[p], lw=2, ms=8, label=LABELS[p], **kw)
    ax.fill_between(x, v - CI * se, v + CI * se, color=COLORS[p], alpha=0.15)


def saving(d, b):
    return 100 * (1 - d / b)


# ---------- figures ----------
def fig_benchmark(d):                                 # Table 7.1 companion
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, ("mean_power", "mean_resp", "p95_resp"),
                              ("Power (W)", "Mean response time (s)", "P95 response time (s)")):
        vals = [d[p][key] for p in POLICIES]
        bars = ax.bar([LABELS[p] for p in POLICIES], vals,
                      color=[COLORS[p] for p in POLICIES], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                    f"{v:,.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    return fig


def fig_kappa_pareto(d):
    dv, base = d["dvfs"], d["baselines"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(col(dv, "mean_resp"), col(dv, "mean_power"),
                xerr=CI * col(dv, "sem_resp"), yerr=CI * col(dv, "sem_power"),
                fmt="o-", color=COLORS["simple"], lw=2, ms=7, capsize=3,
                label="DVFS (κ curve)", zorder=5)
    for r in dv:
        ax.annotate(f"κ={r['kappa']:g}", (r["mean_resp"], r["mean_power"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=7,
                    color=COLORS["simple"])
    for p in ("lwl", "jsq", "random"):
        b = base[p]
        ax.errorbar(b["mean_resp"], b["mean_power"], xerr=CI * b["sem_resp"],
                    yerr=CI * b["sem_power"], fmt=MARKERS[p], color=COLORS[p],
                    ms=13, capsize=4, label=LABELS[p], zorder=6)
    ax.set_xlabel("Mean response time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    return fig


def fig_kappa_p95(d):
    dv, lwl = d["dvfs"], d["baselines"]["lwl"]
    k = col(dv, "kappa")
    sv = saving(col(dv, "mean_power"), lwl["mean_power"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogx(k, col(dv, "p95_resp"), "o-", color=COLORS["simple"], lw=2, label="DVFS P95")
    ax.semilogx(k, col(dv, "mean_resp"), "s--", color=COLORS["simple"], alpha=0.6, label="DVFS MRT")
    ax.axhline(lwl["p95_resp"], color=COLORS["lwl"], lw=2, label="LWL P95")
    ax.axhline(lwl["mean_resp"], color=COLORS["lwl"], ls="--", alpha=0.6, label="LWL MRT")
    ax.set_xlabel("κ")
    ax.set_ylabel("Response time (s)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(k, sv, "o-", color=COLORS["simple"], lw=2, label="Power saving (%)")
    ax.set_xlabel("κ")
    ax.set_ylabel("Power saving vs LWL (%)")
    ax2 = ax.twinx()
    ax2.semilogx(k, col(dv, "p95_resp") / lwl["p95_resp"], "^-", color="crimson", label="P95 ×LWL")
    ax2.semilogx(k, col(dv, "mean_resp") / lwl["mean_resp"], "v--", color="crimson",
                 alpha=0.55, label="MRT ×LWL")
    ax2.set_ylabel("Inflation vs LWL (×)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    return fig


def _sweep(d, xkey, xlabel, logx=False):
    x = np.array(d[xkey], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, se_key, ylabel in ((axes[0], "mean_power", "sem_power", "Power (W)"),
                                    (axes[1], "mean_resp", "sem_resp", "Mean response time (s)")):
        for p in POLICIES:
            band(ax, x, d[p], key, se_key, p)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(x)
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    return fig


def fig_rho(d):
    return _sweep(d, "rho", "ρ (fixed hardware, μmax at ρ=0.7)")


def fig_n(d):
    return _sweep(d, "n", "n (log scale)", logx=True)


def fig_cov(d):
    labels = [l for l in ("rho=0.6", "rho=0.8") if l in d]
    keep = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    fig, axes = plt.subplots(len(labels), 2, figsize=(14, 5 * len(labels)), squeeze=False)
    for row, lbl in enumerate(labels):
        cov = np.array(d[lbl]["cov"])
        idx = np.isin(cov, keep)
        for c, (key, se_key, ylabel) in enumerate((("mean_power", "sem_power", "Power (W)"),
                                                   ("mean_resp", "sem_resp", "Mean response time (s)"))):
            ax = axes[row][c]
            for p in ("simple", "lwl"):
                v, se = col(d[lbl][p], key)[idx], col(d[lbl][p], se_key)[idx]
                ax.plot(cov[idx], v, f"{MARKERS[p]}-", color=COLORS[p], lw=2.5, ms=9, label=LABELS[p])
                ax.fill_between(cov[idx], v - CI * se, v + CI * se, color=COLORS[p], alpha=0.1)
            ax.set_xticks(cov[idx])
            ax.set_xlabel("CoV of job size")
            ax.set_ylabel(ylabel)
            ax.set_title(lbl.replace("rho", "ρ"))
            ax.legend()
    return fig


def fig_decomposition(d):
    rho = np.array(d["rho"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, p in zip(axes.flat, POLICIES):
        w, s, t = col(d[p], "mean_wait"), col(d[p], "mean_service"), col(d[p], "mean_resp")
        ax.fill_between(rho, 0, s, color=SERV_C, alpha=0.5, label="Service time")
        ax.fill_between(rho, s, s + w, color=WAIT_C, alpha=0.5, label="Waiting time")
        ax.plot(rho, t, "k--", marker="o", ms=5, label="Total")
        ax.set_xlabel("ρ")
        ax.set_ylabel("Time (s)")
        ax.set_title(LABELS[p])
        ax.legend(fontsize=9, loc="upper left")
    return fig


def fig_per_server(z):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    ids = np.arange(len(z["simple_utilization_per_server"]))
    for c, p in enumerate(("simple", "lwl")):
        speed = z[f"{p}_mu_avg_busy_per_server"] / float(z[f"{p}_mumax"])
        util = z[f"{p}_utilization_per_server"]
        for r, (vals, ylab) in enumerate(((speed, "E[μ | busy] / μmax"), (util, "Utilization"))):
            ax = axes[r][c]
            ax.bar(ids, vals, color=COLORS[p], edgecolor="none")
            ax.axhline(vals.mean(), color="k", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Server id")
            ax.set_ylabel(ylab)
            ax.set_title(LABELS[p])
            ax.legend(fontsize=9)
    return fig


def fig_google_benchmark(d):
    b = d["benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, log in ((axes[0], "mean_power", "Power (W)", False),
                                (axes[1], "mean_resp", "Mean response time (s, log)", True)):
        vals = [b[p][key] for p in POLICIES]
        bars = ax.bar([LABELS[p] for p in POLICIES], vals,
                      color=[COLORS[p] for p in POLICIES], edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * (1.15 if log else 1.02),
                    f"{v:,.0f}", ha="center", fontsize=9, fontweight="bold")
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    return fig


def fig_google_kappa(d):
    sw, lwl = d["kappa"], d["benchmark"]["lwl"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(col(sw, "mean_resp"), col(sw, "mean_power"), "o-", color=COLORS["simple"],
            lw=2, ms=8, label="DVFS (κ curve)")
    for r in sw:
        ax.annotate(f"κ={r['kappa']:g}  (−{saving(r['mean_power'], lwl['mean_power']):.1f}%)",
                    (r["mean_resp"], r["mean_power"]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=COLORS["simple"])
    ax.plot(lwl["mean_resp"], lwl["mean_power"], MARKERS["lwl"], color=COLORS["lwl"],
            ms=13, label="LWL")
    ax.set_xlabel("Mean response time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    return fig


def fig_google_jobsize(z):
    C, nt = z["C"], z["n_tasks"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(C, bins=np.logspace(np.log10(C.min()), np.log10(C.max()), 60), color=SERV_C)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(C.mean(), color="k", ls="--", label=f"mean={C.mean():.2f}")
    ax.axvline(np.percentile(C, 99), color=WAIT_C, ls=":", label="P99")
    ax.set_xlabel("Job size (GNCU·s)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    pct = np.array([90, 95, 99, 99.5, 99.9, 99.99, 100])
    cov = [np.std(np.minimum(C, np.percentile(C, q))) / np.mean(np.minimum(C, np.percentile(C, q)))
           for q in pct]
    ax.plot(range(len(pct)), cov, "o-", color=COLORS["simple"], lw=2)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([f"{q:g}" for q in pct])
    ax.set_yscale("log")
    ax.set_xlabel("Clipping percentile")
    ax.set_ylabel("CoV of job size")

    ax = axes[2]
    ax.scatter(nt, C, s=3, alpha=0.3, color=COLORS["simple"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tasks per job")
    ax.set_ylabel("Job size (GNCU·s)")
    return fig


FIGURES = {   # name: (input file, builder)
    "benchmark":        ("benchmark_n100.json", fig_benchmark),
    "benchmark_n20":    ("benchmark_n20.json",  fig_benchmark),
    "kappa_pareto":     ("kappa.json",          fig_kappa_pareto),
    "kappa_p95":        ("kappa.json",          fig_kappa_p95),
    "rho_sweep":        ("rho.json",            fig_rho),
    "n_sweep":          ("n.json",              fig_n),
    "cov_sweep":        ("cov.json",            fig_cov),
    "decomposition":    ("ws.json",             fig_decomposition),
    "per_server":       ("trace.npz",           fig_per_server),
    "google_benchmark": ("google.json",         fig_google_benchmark),
    "google_kappa":     ("google.json",         fig_google_kappa),
    "google_job_size":  ("google_jobsize.npz",  fig_google_jobsize),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", default=["all"], choices=[*FIGURES, "all"])
    req = ap.parse_args().fig
    names = list(FIGURES) if "all" in req else req
    for name in names:
        src, build = FIGURES[name]
        data = load(src)
        if data is None:
            print(f"[{name}] skipped: results/{src} not found")
            continue
        print(f"[{name}]")
        save(build(data), name)


if __name__ == "__main__":
    main()
