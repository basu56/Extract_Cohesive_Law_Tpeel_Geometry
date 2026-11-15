"""S3c: element-local assembly, condensation of the incompatible modes, energy-based Newton.

Units: mm, N, MPa, s.  Coarse decks from the unchanged ABAQUS-side builder in a temporary directory.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import tslid.fe  # noqa: F401
from tslid.fe.assembly import ElementSystem
from tslid.fe.incremental import SolverOptions, newton_energy, newton_ls, run
from tslid.fe.tpeel import TPeelModel

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BUILDER = os.path.join(ROOT, "src", "tslid", "build_tpeel_inp.py")
COARSE = ["--ny", "4", "--h-y0", "0.5", "--h-coh", "0.5", "--h-far", "1.5", "--h-leg", "1.5", "--delta", "1.6"]


def _build(d, job, sigc):
    subprocess.run([sys.executable, BUILDER, "--outdir", str(d), "--job", job, "--law", "alpha", "--alpha", "0",
                    "--Gc", "0.4", "--sigc", str(sigc), "--viscosity", "1e-4", "--n-inc", "800"] + COARSE,
                   check=True, capture_output=True)
    return str(d / f"{job}.inp")


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("decks"), "c_s40", 40)


@pytest.fixture(scope="module")
def weak_deck(tmp_path_factory):
    """sigma_c = 2 MPa: the bond damages within the first increments."""
    return _build(tmp_path_factory.mktemp("decks"), "c_s2", 2)


def _state(m, seed=0):
    """A deformed, partly damaged state with finite RP rotations."""
    rng = np.random.default_rng(seed)
    q = 2e-3 * rng.standard_normal(m.n_q)
    q[m.n_full:] = [0.04, -0.02]
    q = m.apply_bc(q, 0.05)
    q[m.slave_dofs] = m.lift(q)[m.slave_dofs]
    hist = m.update_history(q, m.zero_history(), 0.1)
    return q, hist, m.frozen_args(hist, 0.1)


@pytest.mark.parametrize("opts", [dict(), dict(coupling="linear", frame="corotational_linear"),
                                  dict(history="staggered"), dict(kinematics="hencky")])
def test_element_assembly_equals_jacfwd(deck, opts):
    m = TPeelModel(deck, **opts)
    S = ElementSystem(m)
    q, hist, h = _state(m)
    assert float(np.max(hist["D"])) > 0.5                                   # damage active
    E0, E1 = m.energy(q, h), S.energy(q, h)
    assert abs(E1 - E0) <= 1e-13 * abs(E0)
    r0 = m.residual(q, h)
    _, r1 = S.value_and_grad(q, h)
    assert np.abs(r1 - r0).max() <= 1e-12 * np.abs(r0).max()
    _, r2 = m.value_and_grad(q, h)                                          # jitted q-energy (lift in JAX)
    assert np.abs(r2 - r0).max() <= 1e-12 * np.abs(r0).max()
    K0 = m.tangent(q, h)[m.free][:, m.free]
    K1 = S.tangent_free(q, h)
    assert abs(K1 - K0).max() <= 1e-12 * abs(K0).max()
    if m.law is not None:
        a, b = m.update_history(q, hist, 0.1), S.update_history(q, hist, 0.1)
        for k in ("dmax", "D", "Dv"):
            assert np.abs(np.asarray(a[k]) - np.asarray(b[k])).max() <= 1e-13


def test_condensed_step_equals_full_step(deck):
    m = TPeelModel(deck)
    q, hist, h = _state(m)
    S, Sc = ElementSystem(m), ElementSystem(m, condense=True)
    assert Sc.n_sys == S.n_sys - 4 * m.bulk.n_el
    _, r = S.value_and_grad(q, h)
    K = S.tangent_free(q, h)
    rf = r[m.free]
    lu = spla.splu(K.tocsc())
    x0 = lu.solve(rf)
    x1 = x0 + lu.solve(rf - K @ x0)                                          # one refinement step
    dqc = Sc.direction(q, h, r, lambda A, b: spla.spsolve(A.tocsc(), b))
    n = lambda v: np.abs(v).max()
    # The condensed step solves the FULL system to roundoff (backward error as a direct full solve).
    assert n(K @ dqc[m.free] - rf) <= 1e-14 * n(rf)
    assert n(K @ x0 - rf) <= 1e-14 * n(rf)
    # Forward differences are set by cond(K) (1-norm estimate 3.3e8 on this state, eps * cond ~ 7e-8):
    # two SuperLU solves of the same full system (with / without refinement) already differ by up to
    # 8.6e-11 relative [D, S3c log], so 1e-12 forward agreement is not a meaningful criterion.
    assert n(dqc[m.free] - x1) <= 1e-9 * n(x1)
    assert np.all(dqc[m.fixed] == 0.0)


@pytest.mark.parametrize("assembly", ["jacfwd", "element", "element_condensed"])
def test_newton_energy_matches_newton_ls(weak_deck, assembly):
    """15 damaged increments: same converged response as the S3b newton_ls path."""
    m = TPeelModel(weak_deck)
    M = m if assembly == "jacfwd" else ElementSystem(m, condense=assembly.endswith("condensed"))
    st = m.deck.step["static"]
    rec = lambda mm, t, q, hist, h, r: dict(t=t, P=r[m.out["rp1_u2"]], D=float(np.max(np.asarray(hist["D"]))))
    a = run(m, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(), record=rec, max_increments=15,
            newton=newton_ls)
    b = run(M, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(), record=rec, max_increments=15,
            newton=newton_energy)
    ta, tb = [r["t"] for r in a.records], [r["t"] for r in b.records]
    assert ta == tb
    Pa, Pb = np.array([r["P"] for r in a.records]), np.array([r["P"] for r in b.records])
    assert np.abs(Pa - Pb).max() <= 1e-9 * np.abs(Pa).max()
    assert max(r["D"] for r in b.records) > 0.0
    assert M.jit_cache_sizes()[("vg_q" if assembly == "jacfwd" else "vg")] == 1   # no recompilation


def test_abaqus_like_outputs_and_schedule_replay(weak_deck, tmp_path):
    """tpeel_output.TPeelWriter: energy balance closes, damage dissipates, the files carry the
    extract_tpeel_odb.py names; run(schedule=...) reproduces the increment times of another run."""
    import csv
    import json
    from tslid.fe.tpeel_output import TPeelWriter
    m = TPeelModel(weak_deck)
    S = ElementSystem(m, condense=True)
    st = m.deck.step["static"]
    wr = TPeelWriter(S, weak_deck[:-4] + ".json", str(tmp_path), "w")

    def rec(mm, t, q, hist, h, r):
        wr.add(t, q, hist, float(r[m.out["rp1_u2"]]), float(q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]]))
        return dict(t=t, P=float(r[m.out["rp1_u2"]]))

    a = run(S, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(), record=rec, max_increments=25,
            newton=newton_energy)
    wr.close(completed=False)
    rows = list(csv.DictReader(open(tmp_path / "w_response.csv")))
    for k in ("time", "delta", "P", "W_ext", "ALLSE", "ALLPD", "ALLDMD", "ALLVD", "ALLCD"):
        assert k in rows[0]
    last = {k: float(v) for k, v in rows[-1].items()}
    assert last["ALLDMD"] > 0.01 * last["W_ext"]
    bal = abs(last["W_ext"] - last["ALLSE"] - last["ALLDMD"] - last["ALLCD"]) / last["W_ext"]
    assert bal <= 1e-2, bal          # E4 tolerance; 2.5e-3 here: trapezoid on Y dD over large damage steps
    z = np.load(tmp_path / "w_profiles.npz")
    assert z["opening"].shape == (len(rows), len(z["x_face"]))
    assert json.load(open(tmp_path / "w.json"))["solver"] == "tslid.fe"
    assert not (tmp_path / "w.sta").exists()                               # not completed
    times = [r["t"] for r in a.records[1:]]
    b = run(S, st["T"], st["dt0"], st["dtmin"], st["dtmax"], SolverOptions(rtol=1e-6), max_increments=25,
            newton=newton_energy, schedule=times)
    assert [l.t for l in b.log] == pytest.approx(times, rel=1e-13)
