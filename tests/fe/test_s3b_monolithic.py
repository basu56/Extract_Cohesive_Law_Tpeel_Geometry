"""S3b: incremental potential (monolithic damage + displacement).  Units: mm, N, MPa, N/mm, s.

Gate (S3_INSTRUCTIONS.md): with one Newton solve per increment the monolithic scheme reproduces
all 20 S2 single-element ABAQUS runs within the S2 tolerances, and the S2 two-block test.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid import laws
from tslid.fe.benchmarks import se_case, single_cohesive, two_block_bar
from tslid.fe.interface import (AlphaLawParams, TabularLawParams, effective_separation, psi_incremental,
                                traction, update_damage, viscous_weights)
from tslid.fe.laws_jax import envelope_integral, s_shape, s_integral

REF = os.path.join(os.path.dirname(__file__), "..", "..", "results", "solver", "S2_interface",
                   "abaqus_se_reference.npz")
Z = np.load(REF)
JOBS = sorted({k.split("__")[0] for k in Z.files})


@pytest.mark.parametrize("alpha", [-3.0, -1e-3, 0.0, 1e-3, 0.3, 3.0, 7.0])
def test_s_integral(alpha):
    x = jnp.linspace(0.0, 1.0, 401)
    S = s_integral(x, alpha)
    dS = jax.vmap(jax.grad(lambda t: s_integral(t, alpha)))(x[1:-1])
    assert np.abs(np.asarray(dS - s_shape(x[1:-1], alpha))).max() < 1e-12      # S' = s
    assert abs(float(S[0])) < 1e-15
    assert abs(float(S[-1]) - (1 / alpha - 1 / np.expm1(alpha) if alpha else 0.5)) < 1e-12


@pytest.mark.parametrize("alpha", [-3.0, 0.0, 7.0])
def test_envelope_integral_energy(alpha):
    L = laws.AlphaLaw(8500.0, 21.0, 0.4, alpha)
    F = float(envelope_integral(1.5 * L.df, 8500.0, 21.0, 0.4, alpha))
    assert abs(F + 0.5 * 21.0 * L.d0 - 0.4) < 1e-14                        # total = Gc


@pytest.mark.parametrize("tab", [False, True])
@pytest.mark.parametrize("alpha,mu", [(0.0, 0.0), (3.0, 0.0), (-3.0, 1e-4), (7.0, 1e-4), (1e-3, 0.0)])
def test_potential_gradient_is_abaqus_update(alpha, mu, tab):
    """grad Phi_n (frozen history) = traction of update_damage at the new separation, everywhere."""
    L = laws.AlphaLaw(8500.0, 21.0, 0.4, alpha)
    law = TabularLawParams.from_alpha_law(L, 1000) if tab else AlphaLawParams(8500.0, 21.0, 0.4, alpha)
    a, b = viscous_weights(mu, 3e-4)
    rng = np.random.default_rng(7)
    n = 3000
    dmax = jnp.asarray(np.r_[rng.uniform(0, 1.2 * L.df, n - 4), 0.0, L.d0, L.df, 2 * L.df])
    Dv = jnp.asarray(rng.uniform(0, 1, n)) * (mu > 0)
    dn = jnp.asarray(rng.uniform(-0.5, 1.3, n) * L.df)
    ds = jnp.asarray(rng.uniform(-1, 1, n) * L.df)
    tn, ts = jax.grad(lambda x, y: jnp.sum(psi_incremental(x, y, dmax, Dv, law, a, b, 8500.0)),
                      argnums=(0, 1))(dn, ds)
    _, _, Dv1 = update_damage(effective_separation(dn, ds), dmax, Dv, law, mu, 3e-4)
    tn0, ts0 = traction(dn, ds, Dv1, 8500.0)
    assert max(np.abs(tn - tn0).max(), np.abs(ts - ts0).max()) <= 1e-13 * 21.0


def test_potential_gradient_safety():
    """Finite first and second derivatives (in separation and in law parameters) at the kinks:
    dm = 0, d0, d* = dmax_n, df, compression, alpha = 0."""
    L = laws.AlphaLaw(8500.0, 21.0, 0.4, 0.0)
    pts = [0.0, L.d0, 0.5 * (L.d0 + L.df), L.df, 2 * L.df, -L.d0]
    for alpha in (0.0, 3.0):
        def phi(dn, ds, dmax, K0, sig, Gc, al):
            return psi_incremental(dn, ds, dmax, 0.0, AlphaLawParams(K0, sig, Gc, al), 0.0, 1.0, K0)
        for dmax in (0.0, L.d0, 0.5 * (L.d0 + L.df), L.df):
            for d in pts:
                args = (jnp.asarray(d), jnp.asarray(0.0), jnp.asarray(dmax), 8500.0, 21.0, 0.4, alpha)
                g = jax.grad(phi, argnums=(0, 1, 3, 4, 5, 6))(*args)
                hss = jax.hessian(phi, argnums=(0, 1))(*args)
                assert all(np.isfinite(float(x)) for x in g), (alpha, dmax, d, g)
                assert np.isfinite(np.asarray(jax.tree_util.tree_leaves(hss))).all(), (alpha, dmax, d)


def case(job, history):
    meta = json.loads(str(Z[f"{job}__meta"]))
    law, t, un, us, mu = se_case(meta)
    return meta, single_cohesive(law, un, us, t, mu=mu, history=history), single_cohesive(law, un, us, t, mu=mu)


@pytest.mark.parametrize("job", JOBS)
def test_monolithic_single_element_matches_abaqus(job):
    meta, r, r_upd = case(job, "monolithic")
    sigc, Gc = meta["params"]["sigc"], meta["params"]["Gc"]
    for key, ours in (("S22", r["tn"]), ("S12", r["ts"])):
        assert np.abs(ours - r_upd[{"S22": "tn", "S12": "ts"}[key]]).max() <= 1e-12 * sigc   # = S2 scheme
    if "visc" in job:
        k1 = int(np.argmax(r["D"] >= 1.0))
        assert np.abs(r["tn"] - Z[f"{job}__S22"])[:k1].max() / sigc <= 1e-6
        assert np.abs(r["tn"] - Z[f"{job}__S22"]).max() / sigc <= 5e-5
    else:
        tol = 2e-5 if meta["representation"]["softening"] == "tabular" else 1e-6
        for key, ours in (("S22", r["tn"]), ("S12", r["ts"])):
            assert np.abs(ours - Z[f"{job}__{key}"]).max() / sigc <= min(tol, 2e-7)
        assert abs(r["diss"][-1] - float(Z[f"{job}__diss"])) <= 1e-7 * Gc


@pytest.mark.parametrize("alpha", [0.0, 3.0])
def test_two_block_monolithic(alpha):
    L = laws.AlphaLaw(8500.0, 20.0, 0.4, alpha)
    d = np.linspace(0, 1.15 * L.df, 24)[1:]
    law = AlphaLawParams(8500.0, 20.0, 0.4, alpha)
    rm = two_block_bar(law, d, history="monolithic")
    rs = two_block_bar(law, d)
    assert np.abs(rm["R"] - rm["t_n"] * rm["area"]).max() <= 1e-10 * 20.0 * rm["area"]
    assert max(abs(t - L.traction(x)) for t, x in zip(rm["t_n"], rm["delta_n"])) <= 1e-12 * 20.0
    assert rm["D"][-1] == 1.0
    assert np.abs(rm["R"] - rs["R"]).max() <= 1e-10 * 20.0 * rm["area"]
    assert (rm["n_stagger"] == 1).all() and rm["newton"].max() <= 4          # one solve, <= 4 iterations
    assert rm["newton"].sum() < rs["newton"].sum()
