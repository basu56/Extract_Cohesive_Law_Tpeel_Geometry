"""I1a tools: Levenberg-Marquardt (synthetic problem), admissibility projection, Forward reuse.

Units: mm, N, MPa.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.inverse.gauss_newton import Bounds, default_bounds, latin_hypercube, levenberg_marquardt

X = np.linspace(0.0, 1.0, 60)


def _model(theta):
    """A nonlinear 3-parameter curve with a mild sigc-alpha trade-off (like the T-peel)."""
    a, b, c = theta
    y = np.exp(a) * X * np.exp(-X * np.exp(b)) + 0.1 * c * X ** 2 + 0.02 * np.exp(a) * c * X
    J = np.stack([y - 0.1 * c * X ** 2, -np.exp(a) * X ** 2 * np.exp(b) * np.exp(-X * np.exp(b)),
                  0.1 * X ** 2 + 0.02 * np.exp(a) * X], axis=1)
    J[:, 0] = np.exp(a) * X * np.exp(-X * np.exp(b)) + 0.02 * np.exp(a) * c * X
    return y, J


def test_lm_recovers_truth_and_projects():
    b = Bounds(lo=np.array([-2.0, -2.0, -5.0]), hi=np.array([3.0, 3.0, 10.0]), lnK0=10.0)
    truth = np.array([1.0, 0.5, 2.0])
    y_data, _ = _model(truth)
    calls = []

    def ev(th):
        calls.append(th.copy())
        if th[2] > 9.0:                                   # a "failed forward run" region
            return None, None, dict(ok=False)
        y, J = _model(th)
        return y, J, dict(ok=True)

    for th0 in latin_hypercube(6, b, seed=1):
        res = levenberg_marquardt(ev, y_data, 1e-3, th0, b, max_iter=60)
        assert res.converged, res.reason
        assert np.abs(res.theta - truth).max() < 1e-6, (th0, res.theta)
    # optimum on a bound: the truth's alpha lies beyond the box -> converge onto the bound
    b2 = Bounds(lo=np.array([-2.0, -2.0, -5.0]), hi=np.array([3.0, 3.0, 1.0]), lnK0=10.0)
    res = levenberg_marquardt(ev, y_data, 1e-3, np.array([0.0, 0.0, -4.0]), b2, max_iter=60)
    assert res.converged and abs(res.theta[2] - 1.0) < 1e-12, (res.theta, res.reason)
    # step cap: same answers, every evaluated step within the cap, no point evaluated twice
    cap = np.array([0.5, 0.5, 3.0])
    for th0 in latin_hypercube(6, b, seed=1):
        calls.clear()
        res = levenberg_marquardt(ev, y_data, 1e-3, th0, b, max_iter=60, max_step=cap)
        assert res.converged and np.abs(res.theta - truth).max() < 1e-6, (th0, res.theta, res.reason)
        base = [np.asarray(res.log[0]["theta"])]
        for e in res.log[1:]:
            assert np.all(np.abs(np.asarray(e["theta"]) - base[-1]) <= cap + 1e-12)
            if e["accepted"]:
                base.append(np.asarray(e["theta"]))
        pts = np.array(calls)
        assert len(np.unique(pts, axis=0)) == len(pts)
    # E2 projection: ln sigc <= (ln K0 + ln Gc)/2
    bb = default_bounds()
    th = bb.project([np.log(60.0), np.log(0.1), 0.0])
    assert th[0] <= 0.5 * (bb.lnK0 + th[1]) + 1e-12
    assert np.all(th >= bb.lo - 1e-12) and np.all(th <= bb.hi + 1e-12)


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COARSE = ["--ny", "4", "--h-y0", "0.5", "--h-coh", "0.5", "--h-far", "1.5", "--h-leg", "1.5", "--delta", "1.6"]


def test_forward_reuses_kernels_and_resets(tmp_path):
    from tslid.inverse.forward import Forward
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py"), "--outdir", str(tmp_path),
                    "--job", "c", "--law", "alpha", "--alpha", "0", "--Gc", "0.4", "--sigc", "2", "--viscosity", "1e-4",
                    "--n-inc", "800"] + COARSE, check=True, capture_output=True)
    F = Forward(str(tmp_path / "c.inp"), max_increments=20, linsolve="superlu_mmd")   # deterministic solver
    edges = {"w": np.linspace(0.0, 0.02, 6)}                         # 20 increments reach delta ~0.024
    th0 = F.theta_true()
    a = F.evaluate(th0, edges)
    caches = F.S.jit_cache_sizes()
    b = F.evaluate(th0 + np.array([0.1, -0.05, 0.5]), edges)
    c = F.evaluate(th0, edges)
    assert F.S.jit_cache_sizes() == caches                          # new theta: no recompilation
    for x in (a, b, c):                                              # stopped by max_increments, bins covered
        assert "max_increments" in x["message"] and "obs" in x
    assert np.array_equal(a["P"], c["P"]) and np.array_equal(a["S"], c["S"])   # state fully reset
    assert np.abs(b["P"] - a["P"]).max() > 0
    y, Sb = a["obs"]["w"]
    assert y.shape == (5,) and Sb.shape == (5, 3)
