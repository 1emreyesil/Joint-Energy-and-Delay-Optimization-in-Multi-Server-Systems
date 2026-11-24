# Data

Synthetic workloads are generated in code (`src/css/workloads.py`); no data file is needed.

## Google Cluster Data 2019 (Cell a)
The raw trace is not stored in this repository (size and licence).

1. Get the 2019 trace from https://github.com/google/cluster-data (ClusterData2019).
2. Export a CSV with one row per task and the columns `Job_ID`, `Task_ID`,
   `Arrival_Time` (milliseconds; the loader divides by 1000) and `CPU` (task CPU usage
   integrated over time).
3. Run:
   ```bash
   python experiments/run_experiments.py --exp google --google-path /path/to/Cell_a.csv
   ```

`load_google_trace` aggregates tasks into jobs (CPU summed, earliest arrival),
drops zero-size jobs, and sets μ_max so that ρ = 0.7.
Expected summary: 1,153,438 jobs, CoV_T = 1.89, CoV_C = 175.29.
