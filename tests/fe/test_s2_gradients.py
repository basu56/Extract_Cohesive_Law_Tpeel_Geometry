"""S2 test 5: AD vs central finite differences of traction and dissipated energy wrt
theta = (K0, sigc, Gc, alpha).  Threshold 1e-6 relative; d(dissipated energy)/dGc = 1 within 1e-8.

Units: mm, MPa, N/mm.  FD step: h = 1e-6 * |theta_i| (1e-6 for alpha = 0); truncation ~h^2,
roundoff ~eps/h, so the expected FD error is ~1e-9.  At the exact kinks (d = d0, d = df) the law
is not differentiable in theta: there we require finite AD values equal to one of the one-sided
FDs, and compare against central FD at points 1e-3 inside each branch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe import laws_jax as LJ
from tslid.fe.interface import psi

K0, SIGC, GC = 8500.0, 20.0, 0.4
NAMES = ("K0", "sigc", "Gc", "alpha")


def theta(alpha):
    return np.array([K0, SIGC, GC, alpha])


def fd(f, th, i, sided=0):
    h = 1e-6 * abs(th[i]) if th[i] != 0 else 1e-6
    tp, tm = th.copy(), th.copy()
    tp[i] += h
    tm[i] -= h
    if sided > 0:
        return (f(tp) - f(th)) / h
    if sided < 0:
        return (f(th) - f(tm)) / h
    return (f(tp) - f(tm)) / (2 * h)


def check(f, th, rtol=1e-6, sided=False, fscale=SIGC):
    """Compare AD and FD of d f / d ln(theta_i), relative to the largest such sensitivity
    (at least fscale * 1e-3, so points where f is identically 0 do not divide by 0)."""
    g = np.asarray(jax.grad(lambda t: f(t))(jnp.asarray(th)))
    assert np.all(np.isfinite(g)), g
    scale = max(np.abs(g * th).max(), 1e-3 * fscale)
    for i in range(4):
        ref = [fd(lambda t: float(f(jnp.asarray(t))), th, i, s) for s in ((0,) if not sided else (-1, 1))]
        err = min(abs(g[i] - r) * max(abs(th[i]), 1.0) / scale for r in ref)
        assert err <= (rtol if not sided else 1e-4), (NAMES[i], g[i], ref)


def trac(d):
    return lambda t: LJ.traction(d, t[0], t[1], t[2], t[3])


@pytest.mark.parametrize("alpha", [0.0, 3.0, -1.5, 5e-4])
def test_traction_gradients_inside_branches(alpha):
    d0, df = (float(x) for x in LJ.law_params(K0, SIGC, GC, alpha))
    for d in (0.5 * d0, d0 * (1 - 1e-3), d0 + 1e-3 * (df - d0), 0.5 * (d0 + df), df - 1e-3 * (df - d0)):
        check(trac(d), theta(alpha))


@pytest.mark.parametrize("alpha", [0.0, 3.0, -1.5])
def test_traction_gradients_at_kinks_are_finite_and_one_sided(alpha):
    d0, df = (float(x) for x in LJ.law_params(K0, SIGC, GC, alpha))
    for d in (0.0, d0, df, 1.2 * df):
        check(trac(d), theta(alpha), sided=True)


def test_compression_and_zero_separation_have_finite_gradients():
    for dn in (-1e-3, 0.0):
        g = jax.grad(lambda a, b, D, K: psi(a, b, D, K), argnums=(0, 1, 2, 3))(dn, 0.0, 0.5, K0)
        assert all(np.isfinite(float(x)) for x in g)
        assert abs(float(g[0]) - K0 * min(dn, 0.0)) <= 1e-12 * K0      # no damage in compression


def dissipated(t, n=64):
    """int_0^df sigma(d) dd by Gauss-Legendre on [0, d0] and [d0, df]."""
    d0, df = LJ.law_params(t[0], t[1], t[2], t[3])
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = jnp.asarray(x), jnp.asarray(w)
    seg = lambda a, b: 0.5 * (b - a) * jnp.sum(w * LJ.traction(0.5 * (b - a) * x + 0.5 * (a + b), *t))
    return seg(0.0, d0) + seg(d0, df)


@pytest.mark.parametrize("alpha", [0.0, 3.0, -1.5, 7.0, 5e-4])
def test_dissipated_energy_gradient(alpha):
    th = theta(alpha)
    assert abs(float(dissipated(jnp.asarray(th))) - GC) <= 1e-12 * GC
    g = np.asarray(jax.grad(dissipated)(jnp.asarray(th)))
    assert abs(g[2] - 1.0) <= 1e-8
    assert np.abs(g[[0, 1, 3]]).max() <= 1e-8 * GC / min(th[0], th[1])
    check(dissipated, th, fscale=GC)
