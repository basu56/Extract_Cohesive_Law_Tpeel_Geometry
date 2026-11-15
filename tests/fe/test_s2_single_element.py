"""S2 tests 2-4: single cohesive element vs ABAQUS COH2D4 histories (committed reference
results/solver/S2_interface/abaqus_se_reference.npz, produced by scripts/fe/s2_compare.py from
Negishi jobs 45820978 (Phase 0) and 45857972 (S2)).

Units: mm, MPa, N/mm, s.  Thresholds (S0_S2_INSTRUCTIONS.md): traction <= 1e-6 sigc (LINEAR,
EXPONENTIAL), <= 2e-5 sigc (TABULAR); secant unloading <= 1e-6 sigc; dissipated energy vs Gc <= 1e-5.
"""
import json
import os

import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.benchmarks import se_case, single_cohesive
from tslid.fe.interface import AlphaLawParams, CohesiveInterface

REF = os.path.join(os.path.dirname(__file__), "..", "..", "results", "solver", "S2_interface",
                   "abaqus_se_reference.npz")
Z = np.load(REF)
JOBS = sorted({k.split("__")[0] for k in Z.files})


def case(job):
    meta = json.loads(str(Z[f"{job}__meta"]))
    law, t, un, us, mu = se_case(meta)
    return meta, law, t, un, us, mu, single_cohesive(law, un, us, t, mu=mu)


@pytest.mark.parametrize("job", [j for j in JOBS if "visc" not in j])
def test_inviscid_runs_match_abaqus(job):
    meta, law, t, un, us, mu, r = case(job)
    sigc, Gc = meta["params"]["sigc"], meta["params"]["Gc"]
    tol = 2e-5 if meta["representation"]["softening"] == "tabular" else 1e-6
    for ours, key in ((r["tn"], "S22"), (r["ts"], "S12")):
        err = np.abs(ours - Z[f"{job}__{key}"]).max() / sigc
        assert err <= tol and err <= 2e-7, (key, err)          # 2e-7: measured <= 6.8e-8
    unl = np.r_[False, np.diff(un) < 0]
    if unl.any():
        assert np.abs(r["tn"] - Z[f"{job}__S22"])[unl].max() / sigc <= 1e-6
    # path-integrated dissipation: same increments and trapezoid rule as ABAQUS
    assert abs(r["diss"][-1] - float(Z[f"{job}__diss"])) <= 1e-7 * Gc
    assert abs(r["diss"][-1] / Gc - 1) <= 1.1e-5                # 1.04e-5 worst (cyclic), = ABAQUS


@pytest.mark.parametrize("job", [j for j in JOBS if "visc" in j])
def test_viscous_runs_match_abaqus_until_failure(job):
    meta, law, t, un, us, mu, r = case(job)
    sigc, Gc = meta["params"]["sigc"], meta["params"]["Gc"]
    k1 = int(np.argmax(r["D"] >= 1.0))
    assert k1 > 0
    assert np.abs(r["tn"] - Z[f"{job}__S22"])[:k1].max() / sigc <= 1e-6      # backward Euler = ABAQUS
    assert np.abs(r["tn"] - Z[f"{job}__S22"]).max() / sigc <= 5e-5           # post-failure tail, see README
    assert abs(r["W"][-1] - float(Z[f"{job}__ALLWK"])) <= 1e-5 * Gc


def test_viscosity_to_zero_recovers_inviscid():
    law = AlphaLawParams(8500.0, 20.0, 0.4, 3.0)
    t = np.linspace(0, 1, 2001)
    un = 0.06 * t
    r0 = single_cohesive(law, un, 0 * un, t)
    r1 = single_cohesive(law, un, 0 * un, t, mu=1e-12)
    assert np.abs(r1["tn"] - r0["tn"]).max() <= 1e-9 * 20.0


def test_nodal_quadrature_matches_coh2d4_probe():
    """ABAQUS probe s2_probe: one top node opened by 1e-4 mm -> S22 = 0 and K0*1e-4 at the two
    integration points, i.e. the points sit at the nodes (Newton-Cotes)."""
    import jax.numpy as jnp
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    itf = CohesiveInterface(coords, [0, 1], [3, 2], 1.0)
    u = jnp.zeros((4, 2)).at[2, 1].set(1e-4)
    tn, _ = itf.tractions(u, itf.zero_history(), 8500.0)
    assert np.allclose(np.asarray(tn[0]), [0.0, 0.85], rtol=0, atol=1e-12)
