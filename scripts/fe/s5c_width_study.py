"""s5c_width_study.py -- S5c: schedule stability of the Fisher matrix vs the bin width of the observation
operator (D29 uses one 0.1 mm element passage, 0.003 mm).  Light.

    python scripts/fe/s5c_width_study.py <tslid_runs root> [outdir=results/solver/S5_sensitivity]

Runs: A (production rtol 1e-6), C (A's schedule shifted by half an increment), D (0.75 x increment), see
s5c_fisher.py.  For widths w = 1, 2, 4, 8, 16 passages (x 0.003 mm), pre-peak and full windows, theta3 and
theta4: relative change of every eigenvalue of F = (O S)^T (O S) (C, D vs A), angle of the weak eigenvector,
relative change of the bin-averaged S.  Writes s5c_width_study.csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.fe.observation import DELTA_BIN, observation_matrix, windows  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S5_sensitivity"
os.makedirs(OUT, exist_ok=True)
ref = load(os.path.join(ROOT, "phase1a_sigc_sweep"), "sw_s40")
dpk = float(ref["d"][int(np.argmax(ref["P"]))])
R = {k: np.load(os.path.join(ROOT, p)) for k, p in (("A", "solver_S5a_cost/sw_s40_sens_sens.npz"),
                                                     ("C", "solver_S5c/sw_s40_shift_sens.npz"),
                                                     ("D", "solver_S5c/sw_s40_dt075_sens.npz"))}
SETS = {"theta3": [1, 2, 3], "theta4": [0, 1, 2, 3]}
rows = []
for npass in (1, 2, 4, 8, 16):
    w = npass * DELTA_BIN
    W = windows(ref["d_end"], dpk, w)
    for win, e in W.items():
        Y = {k: observation_matrix(np.r_[0.0, z["delta"]], e)[:, 1:] @ z["dP"] for k, z in R.items()}
        for sn, cols in SETS.items():
            F = {k: y[:, cols].T @ y[:, cols] for k, y in Y.items()}
            la, Va = np.linalg.eigh(F["A"])
            for k in ("C", "D"):
                lb, Vb = np.linalg.eigh(F[k])
                dl = (lb - la) / la
                ang = np.degrees(np.arccos(min(1.0, abs(float(Va[:, 0] @ Vb[:, 0])))))
                rows.append(dict(passages=npass, width_mm=w, window=win, bins=len(e) - 1, params=sn, run=k,
                                 max_dlambda_rel=float(np.abs(dl).max()), dlambda_min_rel=float(dl[0]),
                                 weak_angle_deg=float(ang), dS_rel=float(np.linalg.norm(Y[k][:, cols] - Y["A"][:, cols])
                                                                         / np.linalg.norm(Y["A"][:, cols])),
                                 cond=float(la[-1] / la[0])))
with open(os.path.join(OUT, "s5c_width_study.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    if r["params"] == "theta3":
        print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()})
