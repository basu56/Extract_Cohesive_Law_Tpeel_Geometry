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
    m = TPeelModel(str(decks / "c_el.inp"), kinematics="linear", coupling="linear", frame="corotational_linear")
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


# ABAQUS 2026 COH2D4 probes (scripts/fe/build_coh_probe.py, Negishi job 45868344): S22, S12 (MPa) of
# one elastic element (K = 8500 MPa/mm, length 1, width 1), all node displacements prescribed.
COH2D4_PROBES = {  # job: (gap, motion, nlgeom, S22, S12)
    "pg0_shear": (0.0, "shear", False, 0.0, 8.5),
    "pg4_shear": (0.4, "shear", False, 0.0, 8.5),
    "pg4_shear_nl": (0.4, "shear", True, 0.0, 8.5),
    "pg0_rot_nl": (0.0, "rot", True, 0.0, 0.0),
    "pg4_rot_nl": (0.4, "rot", True, 0.0, 0.0),
    "pg4_rot": (0.4, "rot", False, -16.98583794, 0.0),
    "pg4_open_rot_nl": (0.4, "open_rot", True, 8.5, 0.0),
}


@pytest.mark.parametrize("job", sorted(COH2D4_PROBES))
def test_cohesive_kinematics_match_coh2d4_probes(job):
    import math
    from tslid.fe.interface import CohesiveInterface
    g, motion, nl, S22, S12 = COH2D4_PROBES[job]
    X = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, g), (0.0, g)])
    c, s = math.cos(0.1), math.sin(0.1)

    def disp(k, x, y):
        if motion == "shear":
            return (1e-3, 0.0) if k in (2, 3) else (0.0, 0.0)
        dx, dy = x - 0.5, y - g / 2
        if motion == "open_rot" and k in (2, 3):
            dy += 1e-3
        return (0.5 + c * dx - s * dy - x, g / 2 + s * dx + c * dy - y)

    itf = CohesiveInterface(X, [0, 1], [3, 2], 1.0, frame="corotational" if nl else "corotational_linear")
    dn, ds = itf.separations(jnp.asarray([disp(k, *X[k]) for k in range(4)]))
    assert np.allclose(8500.0 * np.asarray(dn), S22, atol=1e-6) and np.allclose(8500.0 * np.asarray(ds), S12, atol=1e-6)
