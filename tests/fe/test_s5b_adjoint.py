"""S5b: observation operator (D29), piecewise-linear many-parameter law, discrete adjoint vs direct.

Units: mm, N, MPa, s.  J = 1/2 sum_k w (O P - y_data)_k^2 with the bin-average operator O; the adjoint
gradient (adjoint.py) must equal the direct one, g^T S with g = O^T w (O P - y), to <= 1e-10 on the same
converged states.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.adjoint import Adjoint, StateRecorder
from tslid.fe.interface import AlphaLawParams, PiecewiseLawParams
from tslid.fe.observation import bin_edges, observation_matrix, observe
from tslid.fe.sensitivity import DirectSensitivity, identity_basis

import test_s5a_sensitivity as T5a

LAW = AlphaLawParams(8500.0, 40.0, 0.4, 2.0)


# ------------------------------------------------------------------ observation operator
def test_observation_operator_exact_for_piecewise_linear():
    rng = np.random.default_rng(0)
    xk = np.r_[0.0, np.sort(rng.uniform(0, 1, 15)), 1.0]                # a piecewise-linear "truth"
    yk = rng.standard_normal(len(xk))
    f = lambda x: np.interp(x, xk, yk)
    edges = bin_edges(0.97, 0.05)
    fine = np.unique(np.r_[np.linspace(0, 1, 2001), edges, xk])           # trapezoid exact: kinks + edges
    ref = np.array([np.trapezoid(f(fine[(fine >= a) & (fine <= b)]), fine[(fine >= a) & (fine <= b)]) / (b - a)
                    for a, b in zip(edges[:-1], edges[1:])])
    for grid in (np.unique(np.r_[xk, np.linspace(0, 1, 37)]), np.unique(np.r_[xk, rng.uniform(0, 1, 300)])):
        O = observation_matrix(grid, edges)
        assert np.allclose(O.sum(axis=1), 1.0, atol=1e-14)
        assert np.abs(O @ f(grid) - ref).max() < 1e-13                   # same bins, any grid containing xk
    assert np.allclose(observe(grid, np.c_[f(grid), 2 * f(grid)], edges)[:, 1], 2 * (O @ f(grid)))


# ------------------------------------------------------------------ piecewise law
def test_piecewise_law_partials_and_energy():
    pw = PiecewiseLawParams.from_alpha(LAW, 20)
    assert abs(float(pw.Gc) - LAW.Gc) < 2e-3 * LAW.Gc
    d = jnp.linspace(0.0, 1.2 * float(pw.df), 41)
    # dF/dd = T: envelope derivative equals the traction (1 - D) K0 d inside the softening range
    s = np.asarray(d)
    soft = (s > float(pw.d0) * 1.001) & (s < float(pw.df) * 0.999)
    dF = np.asarray(jax.vmap(jax.grad(pw.envelope_integral))(d))
    T = (1 - np.asarray(pw.damage(d))) * pw.K0 * s
    assert np.abs(dF - T)[soft].max() < 1e-10 * pw.sigc
    # partials in every leaf vs central FD (knots away from the evaluation points)
    B, names = identity_basis(pw)
    f = lambda lw: jnp.stack([lw.damage(jnp.asarray(0.37 * float(pw.df))), lw.envelope_integral(0.61 * float(pw.df))])
    for j in range(len(names)):
        e = jax.tree_util.tree_map(lambda x: x[j], B)
        ad = np.asarray(jax.jvp(f, (pw,), (e,))[1])
        h = 1e-6 * (abs(float(np.ravel(jax.tree_util.tree_leaves(pw)[0])[0])) if j == 0 else pw.sigc)
        lp = jax.tree_util.tree_map(lambda x, ee: x + h * ee, pw, e)
        lm = jax.tree_util.tree_map(lambda x, ee: x - h * ee, pw, e)
        fd = (np.asarray(f(lp)) - np.asarray(f(lm))) / (2 * h)
        assert np.abs(ad - fd).max() <= 1e-6 * max(np.abs(ad).max(), 1e-12) + 1e-12, (names[j], ad, fd)


# ------------------------------------------------------------------ adjoint vs direct
def _objective(delta, P, edges, data_delta, data_P):
    O = observation_matrix(np.r_[0.0, delta], edges)[:, 1:]               # P_0 = 0 (theta-independent)
    y = O @ P
    yd = observe(data_delta, data_P, edges)
    w = edges[1] - edges[0]
    r = y - yd
    return 0.5 * w * float(r @ r), O.T @ (w * r)                           # J, g = dJ/dP_n


@pytest.mark.parametrize("lawkind", ["alpha", "piecewise"])
def test_two_block_adjoint_equals_direct(lawkind):
    mu = 0.05
    law = LAW if lawkind == "alpha" else PiecewiseLawParams.from_alpha(LAW, 24)
    basis, names = identity_basis(law)
    deltas = np.r_[np.linspace(0.0, 0.03, 41), np.linspace(0.028, 0.012, 6), np.linspace(0.013, 0.045, 25)]
    deltas = np.maximum.accumulate(deltas) + 1e-6 * np.arange(len(deltas))   # monotone record for O
    times = np.linspace(0.0, 1.0, len(deltas))
    prob, *_ = T5a._two_block(mu)
    sens = DirectSensitivity(prob, law, basis=basis, names=names)
    rec = StateRecorder()

    def hook(q, ho, h, hn, dt):
        sens.step(q, ho, h, hn, dt)
        rec(q, ho, h, hn, dt)

    out = _run(law, mu, deltas, times, hook)
    P = out[:, 0]
    d = deltas[1:]
    S = sens.as_arrays()["R6"]
    edges = bin_edges(d[-1], 0.004)
    J, g = _objective(d, P, edges, np.r_[0.0, d], np.r_[0.0, 0.9 * P + 3.0 * np.sin(40 * d)])
    direct = g @ S
    adj = Adjoint(prob, law, 13).gradient(rec.states, g, basis)
    err = np.abs(adj - direct).max() / np.abs(direct).max()
    assert err <= 1e-10, (err, adj, direct)
    assert len(names) == (4 if lawkind == "alpha" else 25)


def _run(law, mu, deltas, times, hook):
    prob, itf, top_y, free = T5a._two_block(mu)
    g = jax.jit(prob.grad)
    U = np.zeros(prob.n_q)
    hist = itf.zero_history()
    out = []
    for k in range(1, len(deltas)):
        dt = times[k] - times[k - 1]
        a, b = mu / (mu + dt), dt / (mu + dt)
        h = (hist["dmax"], hist["Dv"], jnp.asarray(a), jnp.asarray(b))
        U[top_y] = deltas[k]
        prev = np.inf
        for _ in range(80):
            r = np.asarray(g(jnp.asarray(U), h, law))
            dU = prob.solve_many(U, h, law, r[None, :])[0]
            U = U - dU
            n = np.abs(dU).max() / np.abs(U).max()
            if n <= 1e-16 or (n <= 1e-12 and n >= 0.5 * prev):
                r = np.asarray(g(jnp.asarray(U), h, law))
                break
            prev = n
        dmx, D, Dv = prob.hist(jnp.asarray(U), hist["dmax"], hist["Dv"], dt, law)
        new = dict(dmax=dmx, D=D, Dv=Dv)
        hook(U, hist, h, new, dt)
        hist = new
        out.append([r[13]])
    return np.array(out)


def test_single_element_adjoint_equals_direct():
    itf, prob = T5a._single(LAW, 2e-3)
    t, un, us = T5a._path()
    sens = DirectSensitivity(prob, LAW)
    rec = StateRecorder()
    hist = itf.zero_history()
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        q = np.zeros(8)
        q[[4, 6]], q[[5, 7]] = us[k], un[k]
        a, b = 2e-3 / (2e-3 + dt), dt / (2e-3 + dt)
        h = (hist["dmax"], hist["Dv"], jnp.asarray(a), jnp.asarray(b))
        dmx, D, Dv = prob.hist(jnp.asarray(q), hist["dmax"], hist["Dv"], dt, LAW)
        new = dict(dmax=dmx, D=D, Dv=Dv)
        sens.step(q, hist, h, new, dt)
        rec(q, hist, h, new, dt)
        hist = new
    g = np.cos(7 * t[1:])                                                   # any dJ/dFn_n
    direct = g @ sens.as_arrays()["Fn"]
    adj = Adjoint(prob, LAW, 5).gradient(rec.states, g, sens.basis)
    assert np.abs(adj - direct).max() <= 1e-10 * np.abs(direct).max(), (adj, direct)


def test_tpeel_adjoint_equals_direct(weak_deck):
    from tslid.fe.assembly import ElementSystem
    from tslid.fe.incremental import SolverOptions, newton_energy, run
    from tslid.fe.linsolve import SuperLU_MMD
    from tslid.fe.sensitivity import ElementProblem
    from tslid.fe.tpeel import TPeelModel
    m = TPeelModel(weak_deck)
    S = ElementSystem(m, condense=True)
    st = m.deck.step["static"]
    solver = SuperLU_MMD()
    prob = ElementProblem(S, solver)
    ds = DirectSensitivity(prob, S.theta)
    rec = StateRecorder()

    def hook(*a):
        ds.step(*a)
        rec(*a)

    out = run(S, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(rtol=1e-12, linsolve=solver),
              record=lambda mm, t, q, hist, h, r: dict(d=float(q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]]),
                                                       P=float(r[m.out["rp1_u2"]])),
              max_increments=25, newton=newton_energy, on_converged=hook)
    d = np.array([r["d"] for r in out.records[1:]])
    P = np.array([r["P"] for r in out.records[1:]])
    edges = bin_edges(d[-1], 0.003)
    _, g = _objective(d, P, edges, np.r_[0.0, d], np.r_[0.0, 1.05 * P])
    direct = g @ ds.as_arrays()["P"]
    adj = Adjoint(prob, S.theta, m.out["rp1_u2"]).gradient(rec.states, g, ds.basis)
    assert np.abs(adj - direct).max() <= 1e-10 * np.abs(direct).max(), (adj, direct)


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.fixture(scope="module")
def weak_deck(tmp_path_factory):
    d = tmp_path_factory.mktemp("decks")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py"), "--outdir", str(d),
                    "--job", "c_s2", "--law", "alpha", "--alpha", "0", "--Gc", "0.4", "--sigc", "2",
                    "--viscosity", "1e-4", "--n-inc", "800"] + T5a.COARSE, check=True, capture_output=True)
    return str(d / "c_s2.inp")
