# Joint Energy and Delay Optimization in Multi-Server Systems
**Calibrated Speed Scaling**


MSc thesis, Data Science, Sapienza University of Rome (2026)
Supervisors: Prof. Andrea Baiocchi, Prof. Paolo Di Lorenzo · Co-advisor: PhD Student Mert Yıldız

## Overview
Data-center servers can save energy by running slower (DVFS), but slower servers
mean longer response times. This project builds a dispatching and speed-scaling
policy that balances the two through a single parameter, κ: a small κ favours
fast service, a large κ favours energy saving, and the operator picks the point
on this trade-off that fits their needs. The policy is calibrated for
multi-server (M/G/n) systems; the standard single-server (M/G/1) calibration
inflates mean response time by **5.8×** in a 100-server cluster.

## Problem
Data centers used about 415 TWh of electricity in 2024 (≈1.5% of global
demand), and the IEA expects this to more than double by 2030
([IEA, *Energy and AI*, 2025](https://www.iea.org/reports/energy-and-ai)).
Electricity is one of the largest operating costs of a data center, yet jobs
must still be served quickly. Each arriving job raises two questions: which
server should take it, and how fast should every server run? The goal is to
minimise long-run average power + κ · mean response time.

## Approach
- **Lyapunov drift-plus-penalty**: turns the long-run objective into a decision per arrival.
- **Closed-form speed rule**: μ\* = μ_max · min(1, √(U / (3κψE[C]))), where U is the server backlog.
- **Φ-minimisation routing**: each job goes to the server with the lowest marginal cost.
- **ψ_{M/G/n} calibration (main contribution)**: replaces the M/G/1 Pollaczek–Khinchine
  factor with an Erlang-C–corrected (Lee–Longton) version, computed in log-space so it
  stays stable for large n.
- **Workloads**: Poisson arrivals with Weibull job sizes (tunable CoV), and Google
  Cluster Data 2019 (Cell a, 1.15M jobs).
- **Baselines**: LWL (least work left), JSQ (Join Shortest Queue), Random, all at full speed.

## Key results
![Energy vs. response time trade-off](results/figures/kappa_pareto.png)

| Setting | Power saving vs LWL | Mean response time |
|---|---|---|
| Synthetic benchmark (n=100, ρ=0.7, CoV=2, κ=1) | **36.6%** | 2.3× LWL (P95: 1.43×) |
| Load sweep, ρ = 0.3 → 0.9 | 49.7% → 15.8% | ≤ 3× LWL |
| Scale sweep, n = 10 → 1000 | 33.1% → 37.3% | ≈ 2.3× LWL |
| Job-size variability, CoV = 0.5 → 5 (ρ=0.6) | 59.7% → 28.2% | — |
| Google Cluster Data 2019, κ=1 | **8.9%** | 1.01× LWL |
| Google, κ = 0.5 → 2 | 7.0% → 11.0% | 1.007× → 1.018× LWL |

- Calibration matters: with the M/G/1 factor the same policy runs servers much
  slower than intended. It draws less power (36,083 W vs 44,355 W) but its mean
  response time rises from 160.4 s to 923.9 s (5.8×; n=100, ρ=0.7).
- On real traffic the saving is smaller but almost free in delay. κ ∈ [0.5, 2]
  is the recommended range: on synthetic data it spans roughly 29–42% saving,
  on the Google trace 7–11% at ≤ 1.02× LWL delay.

## Repository layout
```
src/css/                 simulator package
  engine.py              event-driven simulator (all policies)
  psi.py                 Erlang-C, ψ_{M/G/1}, ψ_{M/G/n}
  workloads.py           Weibull workload, Google trace loader
  stats.py               multi-seed runs with Student-t confidence intervals
  config.py              default parameters
experiments/
  run_experiments.py     runs the experiments → results/*.json, *.npz
  make_figures.py        builds results/figures/*.png from saved results
notebooks/demo.ipynb     short tour of the simulator (~2 min)
results/                 saved results and figures reproducing the thesis
data/README.md           how to obtain the Google trace
tests/                   unit tests (ψ limits, Weibull moments)
```

## How to run
Tested with Python 3.13 on Google Colab (versions pinned in `requirements.txt`).
```bash
pip install -r requirements.txt
python -m pytest                                      # unit tests
python experiments/run_experiments.py --exp rho kappa # selected experiments
python experiments/make_figures.py                    # figures from saved results
```
For a quick look, open `notebooks/demo.ipynb`. Google trace: see
[`data/README.md`](data/README.md).

Run times (single core): sweeps 20–120 min each; experiments with 10⁶ jobs
× 5 seeds about 75 min; the full set (`--exp all`) about 6–7 hours.

## Known limitations
- ψ_{M/G/n} is an approximation motivated by the Erlang-C formula, not an exact
  M/G/n result; it is exact for M/M/n (CoV = 1) and reduces to M/G/1 at n = 1.
- In the Google trace loader each job size is paired with the inter-arrival
  gap before it rather than after it (one-position shift). Kept as-is so the
  repository reproduces the thesis numbers exactly.
- The ρ sweep keeps the hardware fixed (μ_max set at ρ=0.7), while the CoV
  scenarios rescale μ_max with ρ; results at the same ρ are not comparable
  across the two experiments.

## My contribution
I wrote the simulator from scratch, proposed the ψ_{M/G/n} correction
(building on the Erlang-C formula and the Lee–Longton approximation) and
validated it in simulation, ran all experiments, and built the Google trace
pipeline. The base model formulation comes from the supervisors' working notes.

## Thesis
The full thesis is available on request; a paper based on this work is in
preparation.

## License
Code: MIT (see [`LICENSE`](LICENSE)). To cite this work, see [`CITATION.cff`](CITATION.cff).
