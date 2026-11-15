"""S2 test 6: two Q4IM blocks + one cohesive segment under displacement control:
reaction = traction x area exactly, and the traction follows the law of the opening, through
complete failure.  Units: mm, N, MPa."""
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid import laws
from tslid.fe.benchmarks import two_block_bar
from tslid.fe.interface import AlphaLawParams


@pytest.mark.parametrize("alpha", [0.0, 3.0])
def test_force_equals_traction_times_area(alpha):
    L = laws.AlphaLaw(8500.0, 20.0, 0.4, alpha)
    r = two_block_bar(AlphaLawParams(8500.0, 20.0, 0.4, alpha), np.linspace(0, 1.15 * L.df, 24)[1:])
    assert np.abs(r["R"] - r["t_n"] * r["area"]).max() <= 1e-10 * 20.0 * r["area"]
    assert max(abs(t - L.traction(d)) for t, d in zip(r["t_n"], r["delta_n"])) <= 1e-12 * 20.0
    assert r["D"][-1] == 1.0
