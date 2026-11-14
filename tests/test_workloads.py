import numpy as np
import pytest
from scipy.special import gamma
from css.workloads import weibull_params

@pytest.mark.parametrize("cov", [0.5, 1.0, 2.0, 5.0])
def test_weibull_moments(cov):
    a, k = weibull_params(1.0, cov)
    m1 = a * gamma(1 + 1 / k)
    m2 = a**2 * gamma(1 + 2 / k)
    assert np.isclose(m1, 1.0, rtol=1e-8)
    assert np.isclose(np.sqrt(m2 - m1**2) / m1, cov, rtol=1e-6)
