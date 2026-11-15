"""S1 tests 2-4: plane-strain cantilever vs ABAQUS CPE4I (reference values committed in
results/solver/S1_bulk/abaqus_reference.csv, produced by scripts/fe/run_s1_abaqus.sh + s1_compare.py).

Units: mm, N.  Thresholds (S0_S2_INSTRUCTIONS.md): linear Q4IM at ny = 20 within 0.05 % of
ABAQUS in tip deflection and reaction; NLGEOM within 0.1 % (tip displacement, reaction).
"""
import csv
import os

import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.benchmarks import cantilever, timoshenko_tip

REF = os.path.join(os.path.dirname(__file__), "..", "..", "results", "solver", "S1_bulk", "abaqus_reference.csv")


def ref(case, frame=None):
    with open(REF) as fh:
        rows = [r for r in csv.DictReader(fh) if r["case"] == case]
    rows = [{k: (v if k == "case" else float(v)) for k, v in r.items()} for r in rows]
    return rows if frame is None else [r for r in rows if int(r["frame"]) == frame][0]


@pytest.mark.parametrize("ny", [2, 4, 8, 20])
def test_q4im_linear_matches_cpe4i(ny):
    a = ref(f"lin_ny{ny}_F")[-1]
    r = cantilever(ny, "Q4IM", "linear", "force", P=230.0)
    assert abs(r["v_tip"][-1] / a["v_tip"] - 1) <= 5e-4          # spec threshold
    assert abs(r["v_tip"][-1] / a["v_tip"] - 1) <= 1e-6          # regression: measured ~1e-8
    d = ref(f"lin_ny{ny}_D")[-1]
    rd = cantilever(ny, "Q4IM", "linear", "disp", v=0.67)
    assert abs(rd["R_tip"][-1] / d["R_tip"] - 1) <= 5e-4
    assert abs(rd["R_tip"][-1] / d["R_tip"] - 1) <= 1e-6


def test_q4_locks_and_timoshenko_is_reported():
    """Q4 is > 10 % too stiff at ny = 2 (shear locking); Q4IM is within 0.5 % of Timoshenko
    (end effects of the full clamp; reported, not a pass/fail criterion of the plan)."""
    a = ref("lin_ny2_F")[-1]
    q4 = cantilever(2, "Q4", "linear", "force", P=230.0)
    assert q4["v_tip"][-1] / a["v_tip"] - 1 < -0.10
    im = cantilever(20, "Q4IM", "linear", "force", P=230.0)
    assert abs(im["v_tip"][-1] / timoshenko_tip(230.0) - 1) < 5e-3


def test_q4im_distorted_mesh_matches_cpe4i():
    a = ref("skew_ny4_F")[-1]
    r = cantilever(4, "Q4IM", "linear", "force", P=230.0, hx=1.0, h_y0=None, skew=0.3)
    assert abs(r["v_tip"][-1] / a["v_tip"] - 1) <= 1e-4         # measured 4.7e-6


@pytest.mark.slow
@pytest.mark.parametrize("kin", ["svk", "hencky"])
def test_nlgeom_matches_cpe4i(kin):
    rows = ref("nl_ny20_F")
    r = cantilever(20, "Q4IM", kin, "force", P=1400.0, n_inc=20)
    for k, a in enumerate(rows):
        assert abs(r["v_tip"][k] / a["v_tip"] - 1) <= 1e-3
        assert abs(r["M_clamp"][k] / a["M_clamp"] - 1) <= 1e-3
        # u_tip is small and changes sign near 0.01 rad: measure it against |v_tip|
        assert abs(r["u_tip"][k] - a["u_tip"]) <= 1e-3 * abs(a["v_tip"])
