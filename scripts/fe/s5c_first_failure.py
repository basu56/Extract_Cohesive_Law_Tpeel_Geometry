"""s5c_first_failure.py -- D30 diagnostic: Fisher matrix of sw_s40 on a window that ends at the first full
cohesive-point failure (D = 1), and its stability under the replayed schedules.  Light.

Units: mm, N.  theta3 = (ln sigc, ln Gc, alpha), theta4 adds ln K0.
    python scripts/fe/s5c_first_failure.py <tslid_runs root> [outdir=results/solver/S5_sensitivity]

Runs (s5c_fisher.py): A production rtol 1e-6, C = A's schedule shifted by half an increment, D = 0.75 x
the increment.  delta_ff = delta of the first increment with a quadrature point at D >= 1 (column n_failed
of the run's _jax.csv), per run.  Window: the bins (observation.py, 0.003 mm) with upper edge <= delta_ff of
A, the same bins for all runs (a fixed observation operator).  F = (O S)^T (O S) / s^2, s = 1e-3 P_peak
(assumed).  Reports eigenvalues, condition number, CRB, weak eigenvector, and C/D vs A: relative change of
every eigenvalue, weak-direction angle, relative change of O S.  Writes s5c_first_failure.csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.fe.observation import DELTA_BIN, bin_edges, observation_matrix  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S5_sensitivity"
os.makedirs(OUT, exist_ok=True)
NAMES = ["ln_K0", "ln_sigc", "ln_Gc", "alpha"]
RUNS = {"A_rtol1e-6": ("solver_S5a_cost", "sw_s40_sens"), "C_shift_half": ("solver_S5c", "sw_s40_shift"),
        "D_dt075": ("solver_S5c", "sw_s40_dt075")}
ref = load(os.path.join(ROOT, "phase1a_sigc_sweep"), "sw_s40")
P_PEAK = float(ref["P"].max())
D_PK = float(ref["d"][int(np.argmax(ref["P"]))])
Z, dff = {}, {}
for k, (d, j) in RUNS.items():
    Z[k] = np.load(os.path.join(ROOT, d, j + "_sens.npz"))
    rows = list(csv.DictReader(open(os.path.join(ROOT, d, j + "_jax.csv"))))
    first = next(r for r in rows if int(float(r["n_failed"])) > 0)
    dff[k] = float(first["delta"])
edges = bin_edges(dff["A_rtol1e-6"], DELTA_BIN)
s2 = (1e-3 * P_PEAK) ** 2
out = []
for sn, cols in (("theta3", [1, 2, 3]), ("theta4", [0, 1, 2, 3])):
    Y = {k: (observation_matrix(np.r_[0.0, z["delta"]], edges)[:, 1:] @ z["dP"])[:, cols] for k, z in Z.items()}
    E = {k: np.linalg.eigh(y.T @ y) for k, y in Y.items()}
    la, Va = E["A_rtol1e-6"]
    for k in Z:
        lam, V = E[k]
        F = Y[k].T @ Y[k] / s2
        crb = np.sqrt(np.diag(np.linalg.inv(F)))
        r = dict(run=k, params=sn, delta_first_failure=dff[k], window_end=float(edges[-1]), delta_peak_abaqus=D_PK,
                 bins=len(edges) - 1, cond=float(lam[-1] / lam[0]))
        for i, l in enumerate(lam):
            r[f"lambda{i}"] = float(l / s2)
        for i, c in enumerate(cols):
            r[f"crb_{NAMES[c]}"] = float(crb[i])
            r[f"weak_{NAMES[c]}"] = float(V[i, 0] * np.sign(V[-1, 0]))
        if k != "A_rtol1e-6":
            dl = (lam - la) / la
            r.update(max_dlambda_rel=float(np.abs(dl).max()), dlambda_min_rel=float(dl[0]),
                     weak_angle_deg=float(np.degrees(np.arccos(min(1.0, abs(float(V[:, 0] @ Va[:, 0])))))),
                     dS_rel=float(np.linalg.norm(Y[k] - Y["A_rtol1e-6"]) / np.linalg.norm(Y["A_rtol1e-6"])))
        out.append(r)
with open(os.path.join(OUT, "s5c_first_failure.csv"), "w", newline="") as fh:
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in out:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in out:
    print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
