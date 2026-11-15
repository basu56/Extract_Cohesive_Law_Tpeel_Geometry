"""S5.0 gate 3: per-increment partials dR/dtheta, dH/dtheta, dY/dtheta (jax.jvp) == central finite
differences at frozen q and history, <= 1e-7 relative.

Units: mm, N, MPa, s.  theta = (ln K0, ln sigc, ln Gc, alpha).  FD: Richardson-extrapolated central
differences (h = 2e-4, 1e-4).  Point level: the cohesive traction (gradient of psi_incremental, which
is what R assembles) and the history update, at quadrature points in each regime (elastic loading,
softening, unloading, fully failed, compressive), inviscid and viscous, for several alpha.  Points are
placed at least 1e-3 (relative) away from every switch, so no FD straddles one.  Error = max |AD - FD|
over the components / max(max |AD|, floor), floor 1e-6 sigc (tractions) or 1e-9 (history) for partials
that vanish analytically.  Structure level: R, P
and H of the coarse T-peel (element-assembly path) at a damaged state.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.interface import AlphaLawParams, psi_incremental, update_damage
from tslid.fe.laws_jax import law_params
from tslid.fe.sensitivity import ALPHA_LOG, alpha_log_basis, perturb_alpha_log

K0, SIGC, GC = 8500.0, 40.0, 0.4


def _points(law):
    """(dn, ds, dmax_n, Dv_n) per regime; separations relative to d0, df of the law."""
    d0, df = (float(v) for v in law_params(law.K0, law.sigc, law.Gc, law.alpha))
    m = lambda x: d0 + x * (df - d0)                     # x in (0, 1): softening range
    P = {
        "elastic": [(0.5 * d0, 0.1 * d0, 0.0, 0.0), (0.8 * d0, 0.0, 0.3 * d0, 0.0)],
        "softening": [(m(0.3), 0.0, m(0.2), 0.0), (0.8 * m(0.6), 0.6 * m(0.6), m(0.4), 0.0),
                      (m(0.1), 0.0, 0.5 * d0, 0.0)],
        "unloading": [(m(0.2), 0.0, m(0.55), 0.0),          # not m(0.5): d(1-D)/d ln sigc = 0 there (alpha 0)
                       (0.3 * m(0.7), 0.2 * m(0.7), m(0.7), 0.0)],
        "failed": [(1.5 * df, 0.0, 1.2 * df, 0.0), (0.5 * df, 0.1 * df, 1.3 * df, 0.0)],
        "compressive": [(-0.5 * d0, 0.2 * d0, m(0.45), 0.0), (-0.3 * d0, 0.0, 0.0, 0.0)],
    }
    return P


def _fd(f, law, name):
    """Richardson-extrapolated central difference of f(law) in theta_j."""
    d = {}
    for h in (2e-4, 1e-4):
        d[h] = (np.asarray(f(perturb_alpha_log(law, name, h))) - np.asarray(f(perturb_alpha_log(law, name, -h)))) / (2 * h)
    return (4 * d[1e-4] - d[2e-4]) / 3


def _jvp(f, law, j):
    E = alpha_log_basis(law)
    e = jax.tree_util.tree_map(lambda x: x[j], E)
    return np.asarray(jax.jvp(f, (law,), (e,))[1])


@pytest.mark.parametrize("alpha", [pytest.param(-3.0, marks=pytest.mark.slow), 0.0,
                                   pytest.param(0.3, marks=pytest.mark.slow), 5.0])
@pytest.mark.parametrize("visc", [None, (0.6, 0.4)])
def test_point_partials(alpha, visc):
    law = AlphaLawParams(K0, SIGC, GC, alpha)
    a, b = (0.0, 1.0) if visc is None else visc
    mu, dt = (0.0, 1.0) if visc is None else (0.6, 0.4)                 # a = mu/(mu+dt), b = dt/(mu+dt)
    for regime, pts in _points(law).items():
        for dn, ds, dmax_n, Dv_n in pts:
            Dv_n = Dv_n if visc is None else 0.5 * float(law.damage(dmax_n))   # lagging viscous damage

            def trac(lw):
                return jnp.stack(jax.grad(lambda x, y: psi_incremental(x, y, dmax_n, Dv_n, lw, a, b, lw.K0),
                                          argnums=(0, 1))(dn, ds))

            def hist(lw):
                dm = np.sqrt(max(dn, 0.0) ** 2 + ds ** 2)
                return jnp.stack(update_damage(jnp.asarray(dm), dmax_n, Dv_n, lw, mu, dt))

            for j, name in enumerate(ALPHA_LOG):
                for f, what in ((trac, "traction"), (hist, "history")):
                    ad, fd = _jvp(f, law, j), _fd(f, law, name)
                    assert np.all(np.isfinite(ad)), (regime, what, name)
                    scale = np.abs(ad).max()
                    if regime == "failed" and visc is None and what == "traction":
                        assert scale == 0.0 and np.abs(fd).max() <= 1e-12, (regime, name, ad, fd)
                        continue
                    # relative to the partial, with a floor for partials that vanish analytically (e.g.
                    # d(1 - D)/d ln sigc = 0 at the middle of a linear softening range): 1e-6 sigc for
                    # tractions (MPa), 1e-9 for the history (D, dmax / mm)
                    floor = 1e-6 * SIGC if what == "traction" else 1e-9
                    err = np.abs(ad - fd).max() / max(scale, floor)
                    assert err <= 1e-7, (regime, what, name, alpha, visc, ad, fd, err)


# ------------------------------------------------------------------ structure level (element assembly)
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BUILDER = os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py")
COARSE = ["--ny", "4", "--h-y0", "0.5", "--h-coh", "0.5", "--h-far", "1.5", "--h-leg", "1.5", "--delta", "1.6"]


def test_structure_partials(tmp_path):
    from tslid.fe.assembly import ElementSystem
    from tslid.fe.tpeel import TPeelModel
    subprocess.run([sys.executable, BUILDER, "--outdir", str(tmp_path), "--job", "c", "--law", "alpha", "--alpha", "2",
                    "--Gc", "0.4", "--sigc", "40", "--viscosity", "1e-4", "--n-inc", "800"] + COARSE,
                   check=True, capture_output=True)
    m = TPeelModel(str(tmp_path / "c.inp"))
    S = ElementSystem(m, condense=True)
    law = m.law
    rng = np.random.default_rng(1)
    q = 1e-2 * rng.standard_normal(m.n_q)
    q[m.n_full:] = [0.05, -0.03]
    q = m.apply_bc(q, 0.3)
    hist = S.update_history(q, m.zero_history(), 0.1)                    # damaged history
    q2 = q + 2e-4 * rng.standard_normal(m.n_q)                           # some points load, some unload
    h = m.frozen_args(hist, 0.1)
    D = np.asarray(hist["D"])
    assert (D > 0).any() and (D < 1).any() and (D == 1).any()
    vg, upd = S._raw["value_and_grad"], S._raw["update_history"]
    R = lambda lw: vg(jnp.asarray(q2), h, lw)[1]
    H = lambda lw: jnp.stack(upd(jnp.asarray(q2), hist["dmax"], hist["Dv"], 0.1, lw))
    for j, name in enumerate(ALPHA_LOG):
        for f, what in ((R, "R"), (H, "H")):
            ad, fd = _jvp(f, law, j), _fd(f, law, name)
            err = np.abs(ad - fd).max() / np.abs(ad).max()
            assert err <= 1e-7, (what, name, err)
        ip = m.out["rp1_u2"]                                              # Y = P = R at RP1 U2
        ad, fd = _jvp(R, law, j)[ip], _fd(R, law, name)[ip]
        assert abs(ad - fd) <= 1e-7 * max(abs(ad), 1e-12 * np.abs(_jvp(R, law, j)).max()), (name, ad, fd)
