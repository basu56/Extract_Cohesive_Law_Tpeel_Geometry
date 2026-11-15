"""S2 test 1: material point, tslid.fe.laws_jax vs tslid.laws.  Threshold <= 1e-12 sigc.

Units: mm, MPa.  Parameter sets: the 14 Phase 0 single-element runs
(results/phase0/single_element/single_element_summary.csv) plus alpha in {-3, -1.5, 0, 1.5, 3, 5, 7}
at sigc = 21 and 40, Gc = 0.4, K0 = 8500.
"""
import csv
import os

import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid import laws
from tslid.fe import laws_jax as LJ

K0 = 8500.0
SUMMARY = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase0", "single_element",
                       "single_element_summary.csv")


def param_sets():
    with open(SUMMARY) as fh:
        se = sorted({(float(r["sigc"]), float(r["Gc"]), float(r["alpha"])) for r in csv.DictReader(fh)})
    extra = [(s, 0.4, a) for s in (21.0, 40.0) for a in (-3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 7.0)]
    return se + extra


@pytest.mark.parametrize("sigc,Gc,alpha", param_sets())
def test_traction_and_damage_match_tslid_laws(sigc, Gc, alpha):
    L = laws.AlphaLaw(K0, sigc, Gc, alpha)
    d0, df = LJ.law_params(K0, sigc, Gc, alpha)
    assert abs(float(d0) - L.d0) <= 1e-15 * L.d0
    assert abs(float(df) - L.df) <= 1e-13 * L.df
    d = np.r_[np.linspace(0.0, 1.3 * L.df, 4001), L.d0, L.df, 0.5 * (L.d0 + L.df)]
    t = np.asarray(LJ.traction(jnp.asarray(d), K0, sigc, Gc, alpha))
    D = np.asarray(LJ.damage(jnp.asarray(d), K0, sigc, Gc, alpha))
    t_ref = np.array([L.traction(x) for x in d])
    D_ref = np.array([L.damage(x) for x in d])
    assert np.abs(t - t_ref).max() <= 1e-12 * sigc
    assert np.abs(D - D_ref).max() <= 1e-12
    # secant consistency: sigma = (1 - D) K0 d
    assert np.abs((1 - D) * K0 * d - t_ref).max() <= 1e-12 * sigc


def test_g_alpha_series_is_exact_near_zero():
    """g(alpha) is accurate to ~eps on both sides of the series/closed-form switch at |alpha| = 0.5
    (tslid.laws' closed form loses ~1e-12 to cancellation near alpha = 1e-4)."""
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    for a in (-0.49, -1e-4, 1e-7, 1e-4, 1e-3, 0.49, 0.51, -0.51, 3.0, -3.0):
        A = Decimal(a)
        ref = float(1 / A - 1 / (A.exp() - 1))
        assert abs(float(LJ.g_alpha(a)) / ref - 1) <= 2e-15, a
