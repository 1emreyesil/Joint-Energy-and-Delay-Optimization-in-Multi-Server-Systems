import numpy as np
import pytest
from css.psi import erlang_c, psi_mg1, psi_mgn

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
COVS = [0.5, 1.0, 2.0, 5.0]

@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("cov", COVS)
def test_n1_reduces_to_mg1(rho, cov):
    assert np.isclose(psi_mgn(1, rho, cov), psi_mg1(rho, cov))

@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("rho", RHOS)
def test_cov1_is_exact_mmn(n, rho):
    assert np.isclose(psi_mgn(n, rho, 1.0), 1 + erlang_c(n, rho) / (n * (1 - rho)))

def test_large_n_psi_tends_to_one():
    assert abs(psi_mgn(1000, 0.7, 2.0) - 1) < 1e-6
