"""S3a: deck import, rigid-coupling elimination, closed-form Hencky.  Units: mm, N, MPa.

Uses a coarse deck written by the unchanged ABAQUS-side builder (src/tslid/build_tpeel_inp.py)
into a temporary directory, so the test needs no cluster files.
"""
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl

import tslid.fe  # noqa: F401
from tslid.fe.inp_import import cohesive_law, read_inp, summary
from tslid.fe.kinematics import lame, psi_hencky
from tslid.fe.tpeel import TPeelModel, newton

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BUILDER = os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py")
COARSE = ["--ny", "4", "--h-y0", "0.5", "--h-coh", "0.5", "--h-far", "1.5", "--h-leg", "1.5", "--delta", "1.6"]


@pytest.fixture(scope="module")
def decks(tmp_path_factory):
    d = tmp_path_factory.mktemp("decks")
    for job, law in (("c_el", ["--law", "elastic", "--n-inc", "4"]),
                     ("c_a7", ["--law", "alpha", "--sigc", "21", "--alpha", "7", "--viscosity", "1e-4", "--n-inc", "800"])):
        subprocess.run([sys.executable, BUILDER, "--outdir", str(d), "--job", job] + law + COARSE, check=True,
                       capture_output=True)
    return d


def test_import_matches_builder(decks):
    for job in ("c_el", "c_a7"):
        deck = read_inp(str(decks / f"{job}.inp"))
        meta = json.load(open(decks / f"{job}.json"))
        g, p = meta["geometry"], meta["params"]
        s = summary(deck)
        assert s["n_nodes"] == g["n_nodes"] and s["n_elems"] == g["n_elems"] and s["n_coh"] == g["n_bond"]
        assert s["n_unused_nodes"] == 2                                     # the two reference points
        H, L, gr, xg, tp, lh = p["H"], p["L"], p["groove"], g["x_groove_end"], p["taper"], p["leg_h"]
        area = 2 * (L * H - gr * xg - gr * tp / 2 + H * lh)                 # arms (minus groove) + legs
        assert abs(s["bulk_area"] - area) <= 1e-9 * area
        assert s["bond_x0"] == p["a0"] and abs(s["bond_x1"] - L) < 1e-9
        assert deck.solid_section()["thickness"] == p["W"] and deck.cohesive_section()["width"] == p["W"]
        assert [c["kind"] for c in deck.couplings] == ["kinematic", "kinematic"]
    kind, lp = cohesive_law(read_inp(str(decks / "c_a7.inp")), "MAT_TIP")
    assert kind == "alpha" and lp["alpha"] == 7.0 and abs(lp["Gc"] - 0.4) < 1e-7 and lp["sigc"] == 21.0


def test_rigid_translation_and_fd_tangent(decks):
    m = TPeelModel(str(decks / "c_el.inp"))
    h = m.frozen_args(m.zero_history(), 1.0)
    q = np.zeros(m.n_q)
    q[: 2 * m.n_nodes: 2], q[1: 2 * m.n_nodes: 2] = 0.3, -0.7
    assert abs(m.energy(q, h)) < 1e-12 and np.abs(m.residual(q, h)).max() < 1e-6
    rng = np.random.default_rng(0)
    q = 1e-3 * rng.standard_normal(m.n_q)
    q[m.n_full:] = [0.05, -0.03]                                            # finite RP rotations
    K, r = m.tangent(q, h, return_residual=True)
    indep = np.setdiff1d(np.arange(m.n_q), m.slave_dofs)
    v = np.zeros(m.n_q)
    v[indep] = rng.standard_normal(len(indep))
    eps = 1e-6
    fd = (m.residual(q + eps * v, h) - m.residual(q - eps * v, h)) / (2 * eps)
    kv = K @ v
    assert np.abs(fd[indep] - kv[indep]).max() <= 1e-8 * np.abs(kv[indep]).max()
    for c in m.couplings:                                                   # theta rows incl. G term
        assert abs(fd[c["theta"]] - kv[c["theta"]]) <= 1e-8 * abs(kv[c["theta"]])


def test_linear_coupling_solve_is_linear(decks):
    """NLGEOM = NO analogue: linear kinematics + linearised coupling -> RF2 exactly proportional."""
    m = TPeelModel(str(decks / "c_el.inp"), kinematics="linear", coupling="linear")
    h = m.frozen_args(m.zero_history(), 1.0)
    out = []
    for lam in (0.5, 1.0):
        q, info = newton(m, m.apply_bc(m.zero_q(), lam), h)
        assert info["converged"]
        out.append(m.outputs(q, h))
    assert abs(out[1]["rp1_rf2"] - 2 * out[0]["rp1_rf2"]) <= 1e-9 * out[1]["rp1_rf2"]
    assert abs(out[1]["rp1_rf2"] + out[1]["rp2_rf2"]) <= 1e-8 * out[1]["rp1_rf2"]   # global equilibrium


def test_hencky_closed_form():
    lam, mu = lame(70000.0, 0.33)
    rng = np.random.default_rng(3)
    for scale in (1e-3, 0.05, 0.5):
        H = scale * rng.standard_normal((50, 2, 2))
        e = np.asarray(psi_hencky(jnp.asarray(H), lam, mu))
        ref = []
        for hh in H:
            C = (np.eye(2) + hh).T @ (np.eye(2) + hh)
            l = 0.5 * sl.logm(C).real
            ref.append(0.5 * lam * np.trace(l) ** 2 + mu * np.sum(l * l))
        assert np.max(np.abs(e - ref) / np.array(ref)) < 1e-10
    g = jax.grad(lambda H: psi_hencky(H, lam, mu).sum())
    assert np.isfinite(np.asarray(g(jnp.zeros((1, 2, 2))))).all()
    c, s = np.cos(0.3), np.sin(0.3)
    assert np.abs(np.asarray(g(jnp.asarray([[[c - 1, -s], [s, c - 1]]])))).max() < 1e-9   # rotation: no stress
