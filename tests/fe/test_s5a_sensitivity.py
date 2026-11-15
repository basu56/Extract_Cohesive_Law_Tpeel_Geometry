"""S5.0 / S5a: law parameters as differentiable inputs; direct sensitivities (sensitivity.py).

Units: mm, N, MPa, s.  theta = (ln K0, ln sigc, ln Gc, alpha).
1. single cohesive element (S2 setting, displacement-driven, cyclic + compression, with and without
   viscosity): recursion == jax.jacfwd of the analytic traction history along the same path, <= 1e-10;
2. two-block bar (S2, monolithic): recursion == central finite differences (Richardson-extrapolated,
   same load steps) <= 1e-9;
3. coarse T-peel: condensed vs uncondensed sensitivities agree; the solve is consistent (res_free);
   the early increments of a weak bond match central finite differences with schedule replay.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.interface import AlphaLawParams, CohesiveInterface, effective_separation, update_damage
from tslid.fe.laws_jax import damage
from tslid.fe.sensitivity import (ALPHA_LOG, DenseProblem, DirectSensitivity, ElementProblem, alpha_log_basis,
                                  perturb_alpha_log)

LAW = AlphaLawParams(8500.0, 40.0, 0.4, 0.0)        # d0 = 4.7e-3, df = 2.0e-2 mm


# ------------------------------------------------------------------ 1. single cohesive element
def _single(law, mu):
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    itf = CohesiveInterface(coords, [0, 1], [3, 2], 1.0, "nodal")

    def energy(q, h, lw):
        return itf.energy_incremental(q.reshape(4, 2), h[0], h[1], lw, h[2], h[3], lw.K0)

    def hist(q, dmax, Dv, dt, lw):
        return update_damage(effective_separation(*itf.separations(q.reshape(4, 2))), dmax, Dv, lw, mu, dt)

    dm = lambda q: effective_separation(*itf.separations(q.reshape(4, 2)))
    prob = DenseProblem(energy, hist, dm, 8, [], {"Fn": ("r", 5), "Fs": ("r", 4)})
    return itf, prob


def _path():
    """Mixed-mode cyclic path: soften, unload, reload, compress, reload to failure."""
    knots_t = [0.0, 0.25, 0.4, 0.6, 0.7, 1.0]
    knots_n = [0.0, 0.012, 0.004, 0.016, -0.001, 0.026]
    t = np.linspace(0.0, 1.0, 301)
    un = np.interp(t, knots_t, knots_n)
    us = 0.3 * np.abs(un) * np.sin(3.0 * t)
    return t, un, us


@pytest.mark.parametrize("mu", [0.0, 2e-3])
@pytest.mark.parametrize("alpha", [0.0, 4.0])
def test_single_element_recursion_equals_ad_of_analytic_history(mu, alpha):
    law = AlphaLawParams(LAW.K0, LAW.sigc, LAW.Gc, alpha)
    itf, prob = _single(law, mu)
    t, un, us = _path()
    sens = DirectSensitivity(prob, law)
    hist = itf.zero_history()
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        q = np.zeros(8)
        q[[4, 6]] = us[k]
        q[[5, 7]] = un[k]
        a, b = (0.0, 1.0) if mu == 0.0 else (mu / (mu + dt), dt / (mu + dt))
        h = (hist["dmax"], hist["Dv"], jnp.asarray(a), jnp.asarray(b))
        dmx, D, Dv = prob.hist(jnp.asarray(q), hist["dmax"], hist["Dv"], dt, law)
        new = dict(dmax=dmx, D=D, Dv=Dv)
        sens.step(q, hist, h, new, dt)
        hist = new
    S = sens.as_arrays()
    assert S["finite"].all()

    def analytic(th):                       # (Fn, Fs) at node 2 from the secant law, explicit loop
        K0, sigc, Gc, al = jnp.exp(th[0]), jnp.exp(th[1]), jnp.exp(th[2]), th[3]
        dmax, Dv, out = 0.0, 0.0, []
        for k in range(1, len(t)):
            dt = t[k] - t[k - 1]
            dn, ds = un[k], us[k]
            m = jnp.sqrt(max(dn, 0.0) ** 2 + ds ** 2)
            dmax = jnp.where(m > dmax, m, dmax)
            Dk = damage(dmax, K0, sigc, Gc, al)
            Dv = Dk if mu == 0.0 else (dt * Dk + mu * Dv) / (mu + dt)
            tn = (1.0 - Dv) * K0 * max(dn, 0.0) + K0 * min(dn, 0.0)
            ts = (1.0 - Dv) * K0 * ds
            out.append(jnp.stack([0.5 * tn, 0.5 * ts]))           # nodal weight L/2 = 0.5, width 1
        return jnp.stack(out)

    th0 = jnp.array([np.log(law.K0), np.log(law.sigc), np.log(law.Gc), alpha])
    ref = np.asarray(jax.jacfwd(analytic)(th0))                       # (n, 2, 4)
    assert np.abs(np.asarray(analytic(th0))[:, 0]).max() > 1.0
    for j, name in enumerate(("Fn", "Fs")):
        err = np.abs(S[name] - ref[:, j, :]).max(axis=0) / np.abs(ref[:, j, :]).max(axis=0)
        assert np.all(err <= 1e-10), (name, err)


# ------------------------------------------------------------------ 2. two-block bar (S2), monolithic
def _two_block(mu):
    from tslid.fe.benchmarks import AL
    from tslid.fe.elements import BulkModel
    gap, W = 0.4, 15.0
    y0 = [0.0, 1.0, 1.0 + gap, 2.0 + gap]
    coords = np.array([[0.0, y0[0]], [1.0, y0[0]], [1.0, y0[1]], [0.0, y0[1]],
                       [0.0, y0[2]], [1.0, y0[2]], [1.0, y0[3]], [0.0, y0[3]]])
    bulk = BulkModel(coords, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), AL["E"], AL["nu"], W, "Q4IM", "linear")
    itf = CohesiveInterface(coords, [3, 2], [4, 5], W)
    nn = len(coords)

    def energy(U, h, lw):
        return bulk.energy(U) + itf.energy_incremental(U[: 2 * nn].reshape(-1, 2), h[0], h[1], lw, h[2], h[3], lw.K0)

    def hist(U, dmax, Dv, dt, lw):
        return update_damage(effective_separation(*itf.separations(U[: 2 * nn].reshape(-1, 2))), dmax, Dv, lw, mu, dt)

    dm = lambda U: effective_separation(*itf.separations(U[: 2 * nn].reshape(-1, 2)))
    fixed = np.array([0, 1, 3, 14, 13, 15])
    top_y = np.array([13, 15])
    free = np.setdiff1d(np.arange(bulk.n_dof), fixed)
    prob = DenseProblem(energy, hist, dm, bulk.n_dof, free, {"R6": ("r", 13), "R7": ("r", 15), "u5": ("q", 11)})
    return prob, itf, top_y, free


def _run_two_block(law, mu, deltas, times, sens=None):
    prob, itf, top_y, free = _two_block(mu)
    g = jax.jit(prob.grad)
    U = np.zeros(prob.n_q)
    hist = itf.zero_history()
    out = []
    for k in range(1, len(deltas)):
        dt = times[k] - times[k - 1]
        a, b = (0.0, 1.0) if mu == 0.0 else (mu / (mu + dt), dt / (mu + dt))
        h = (hist["dmax"], hist["Dv"], jnp.asarray(a), jnp.asarray(b))
        U[top_y] = deltas[k]
        prev = np.inf
        for _ in range(60):                 # Newton to roundoff (stagnation): the FD reference needs it
            r = np.asarray(g(jnp.asarray(U), h, law))
            dU = prob.solve_many(U, h, law, r[None, :])[0]
            U = U - dU
            n = np.abs(dU).max() / np.abs(U).max()
            if n <= 1e-16 or (n <= 1e-13 and n >= 0.5 * prev):
                r = np.asarray(g(jnp.asarray(U), h, law))
                break
            prev = n
        else:
            raise AssertionError("no convergence")
        dmx, D, Dv = prob.hist(jnp.asarray(U), hist["dmax"], hist["Dv"], dt, law)
        new = dict(dmax=dmx, D=D, Dv=Dv)
        if sens is not None:
            sens.step(U, hist, h, new, dt)
        hist = new
        out.append([r[13], r[15], U[11], float(jnp.max(dmx))])
    return np.array(out), prob


@pytest.mark.parametrize("mu", [0.0, 0.05])
def test_two_block_recursion_equals_finite_differences(mu):
    # soften, unload, reload (still softening), then past df = 0.02 mm (fully failed: zero derivatives).
    # No load step lands within the FD step of a switch (df itself would be one: checked via n_fail).
    deltas = np.r_[np.linspace(0.0, 0.0172, 31), np.linspace(0.016, 0.008, 6), np.linspace(0.0095, 0.0187, 9),
                   [0.0213, 0.026, 0.031]]
    times = np.linspace(0.0, 1.0, len(deltas))
    prob, *_ = _two_block(mu)
    sens = DirectSensitivity(prob, LAW)
    base, _ = _run_two_block(LAW, mu, deltas, times, sens)
    S = sens.as_arrays()
    assert S["finite"].all() and S["res_free"].max() < 1e-10
    assert S["n_switch"].sum() == 0 and S["n_fail"].sum() == 0
    assert base[-1, 3] > 0.02                       # failed; inviscid: zero traction and derivative
    if mu == 0.0:
        assert np.abs(S["R6"][-1]).max() == 0.0
    for j, name in enumerate(ALPHA_LOG):
        fd = {}
        for hh in (2e-4, 1e-4):
            p = _run_two_block(perturb_alpha_log(LAW, name, hh), mu, deltas, times)[0]
            m = _run_two_block(perturb_alpha_log(LAW, name, -hh), mu, deltas, times)[0]
            fd[hh] = (p - m) / (2 * hh)
        rich = (4 * fd[1e-4] - fd[2e-4]) / 3                          # O(h^4)
        for c, key in ((0, "R6"), (1, "R7"), (2, "u5")):
            ref = rich[:, c]
            err = np.abs(S[key][:, j] - ref).max() / np.abs(ref).max()
            assert err <= 1e-9, (name, key, err)


# ------------------------------------------------------------------ 3. coarse T-peel
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BUILDER = os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py")
COARSE = ["--ny", "4", "--h-y0", "0.5", "--h-coh", "0.5", "--h-far", "1.5", "--h-leg", "1.5", "--delta", "1.6"]


@pytest.fixture(scope="module")
def weak_deck(tmp_path_factory):
    d = tmp_path_factory.mktemp("decks")
    subprocess.run([sys.executable, BUILDER, "--outdir", str(d), "--job", "c_s2", "--law", "alpha", "--alpha", "0",
                    "--Gc", "0.4", "--sigc", "2", "--viscosity", "1e-4", "--n-inc", "800"] + COARSE,
                   check=True, capture_output=True)
    return str(d / "c_s2.inp")


def _tpeel_run(deck, theta=None, condense=True, n_inc=12, schedule=None, sens=False):
    from tslid.fe.assembly import ElementSystem
    from tslid.fe.incremental import SolverOptions, newton_energy, run
    from tslid.fe.linsolve import SuperLU_MMD
    from tslid.fe.tpeel import TPeelModel
    m = TPeelModel(deck)
    S = ElementSystem(m, condense=condense, theta=theta)
    st = m.deck.step["static"]
    solver = SuperLU_MMD()
    ds = DirectSensitivity(ElementProblem(S, solver), S.theta) if sens else None
    rec = lambda mm, t, q, hist, h, r: dict(t=t, P=float(r[m.out["rp1_u2"]]), mu=float(q[m.out["mouth_u_u2"]]))
    res = run(S, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(rtol=1e-12, linsolve=solver),
              record=rec, max_increments=n_inc, newton=newton_energy, schedule=schedule,
              on_converged=None if ds is None else ds.step)
    return res, ds, m


def test_tpeel_condensed_equals_uncondensed_and_consistent(weak_deck):
    _, a, m = _tpeel_run(weak_deck, sens=True, n_inc=8)
    _, b, _ = _tpeel_run(weak_deck, sens=True, n_inc=8, condense=False)
    A, B = a.as_arrays(), b.as_arrays()
    assert A["finite"].all() and A["res_free"].max() < 1e-8, A["res_free"]
    for k in ("P", "mouth_u", "mouth_l"):
        assert np.abs(A[k] - B[k]).max() <= 1e-9 * np.abs(B[k]).max(), k


@pytest.mark.slow
def test_tpeel_matches_finite_differences_with_replay(weak_deck):
    """First 20 increments of the weak bond (damage from increment 10 on), rtol 1e-12, h = 1e-3:
    measured curve-wise errors 4e-8 (ln K0) ... 2e-6 (alpha); h = 1e-4 is noise-dominated for alpha."""
    n = 20
    res, ds, m = _tpeel_run(weak_deck, sens=True, n_inc=n)
    S = ds.as_arrays()
    assert S["finite"].all() and S["n_switch"].sum() == 0 and S["n_fail"].sum() == 0
    sched = [r["t"] for r in res.records[1:]]
    for j, name in enumerate(ALPHA_LOG):
        hh = 1e-3
        runs = [_tpeel_run(weak_deck, perturb_alpha_log(m.law, name, s * hh), n_inc=n, schedule=sched)[0]
                for s in (1, -1)]
        for r in runs:
            assert [x["t"] for x in r.records[1:]] == sched                # replay: same increments
        Pp, Pm = (np.array([x["P"] for x in r.records[1:]]) for r in runs)
        fd = (Pp - Pm) / (2 * hh)
        err = np.linalg.norm(S["P"][:, j] - fd) / np.linalg.norm(fd)
        assert err <= 1e-5, (name, err)
