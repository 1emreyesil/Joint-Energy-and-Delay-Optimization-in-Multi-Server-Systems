"""Default parameters (thesis Table 6.1)."""
import numpy as np

EC, CoVC, ET     = 1.0, 2.0, 1.0
N, RHO, KAPPA    = 100, 0.7, 1.0
PMIN, PMAX       = 1.0, 1000.0
NT, WARMUP, SEED = 1_000_000, 50_000, 42

NT_SWEEP, WARMUP_SWEEP = 50_000, 5_000
SEEDS = [42, 43, 44, 45, 46]

KAPPA_VEC_DEFAULT = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
RHO_VEC_DEFAULT   = np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
N_VEC_DEFAULT     = np.array([10, 20, 50, 100, 200, 500, 1000])
COV_VEC_DEFAULT   = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

MUMAX    = EC / (N * RHO * ET)
POLICIES = ["simple", "lwl", "jsq", "random"]
LABELS   = ["DVFS", "LWL", "JSQ", "Random"]
