"""i1a_analysis.py -- I1a tables and figure: multi-start recovery, measured vs predicted solver-mismatch bias.
Light.

Units: theta = (ln sigc, ln Gc, alpha).
    python scripts/inverse/i1a_analysis.py <rundir> [outdir=results/inverse/I1_parametric]

Inputs: <rundir>/<case>_<window>_s<k>.json (i1a_invert.py, 8 starts per case and window) and
<case>_full_truth.json/.npz (model and binned Jacobian at the truth, both windows).
- Starts: converged theta_hat, J, evaluations, reason, wall.  Minima: starts are grouped with the
  lowest-J start if every parameter agrees to 1e-3 (D31); other groups are reported as distinct minima.
- Bias (D31 I1a.4): measured d_theta = theta_hat(best start) - theta_true; predicted
  d_theta_pred = (S^T S)^-1 S^T (y_ABAQUS - y_ours(theta_true)), S the binned Jacobian at the truth.
  Gate per parameter: |meas - pred| <= 0.2 |pred|, or |meas| and |pred| both <= 1e-4.
  For scale: the Cramer-Rao sd at s = 1e-3 P_peak (assumed noise) from the same S.
Writes i1a_starts.csv, i1a_bias.csv, i1a.png/.pdf.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/inverse/I1_parametric"
os.makedirs(OUT, exist_ok=True)
NAMES = ["ln_sigc", "ln_Gc", "alpha"]
CASES = ["sw_s40", "as_s40_a7", "sw_s21", "as_s21_am3"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1495b", "#6b6b6b", "#e3a21a", "#3cb4c4"]


def write(rows, name):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    for r in rows:
        print(name, {k: (f"{v:.4g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def theta_of(t):
    return np.array([np.log(t["sigc"]), np.log(t["Gc"]), t["alpha"]])


starts, bias, runs = [], [], {}
for case in CASES:
    tj = os.path.join(RUN, f"{case}_full_truth.json")
    tz = os.path.join(RUN, f"{case}_full_truth.npz")
    Z = np.load(tz) if os.path.exists(tz) else None
    for win in ("prepeak", "full"):
        R = []
        for f in sorted(glob.glob(os.path.join(RUN, f"{case}_{win}_s*.json"))):
            r = json.load(open(f))
            if not r.get("done"):
                starts.append(dict(case=case, window=win, start=r["tag"].split("_s")[-1], done=False,
                                   evaluations=len(r.get("log", []))))
                continue
            R.append(r)
        if not R:
            continue
        tt = theta_of(R[0]["truth"])
        best = min(R, key=lambda r: r["J"])
        tb = np.array(best["theta_hat"])
        groups = []
        for r in sorted(R, key=lambda r: r["J"]):
            th = np.array(r["theta_hat"])
            for gi, g in enumerate(groups):
                if np.all(np.abs(th - g) <= 1e-3):
                    r["_min"] = gi
                    break
            else:
                groups.append(th)
                r["_min"] = len(groups) - 1
        runs[(case, win)] = R
        for r in R:
            th = np.array(r["theta_hat"])
            row = dict(case=case, window=win, start=r["tag"].split("_s")[-1], done=True, minimum=r["_min"],
                       J=r["J"], converged=r["converged"], reason=r["reason"], evaluations=r["evaluations"],
                       wall_h=r["wall_total"] / 3600.0, cutbacks_total=sum((e.get("cutbacks") or 0) for e in r["log"]))
            for i, n in enumerate(NAMES):
                row[f"theta0_{n}"] = r["theta0"][i]
                row[f"hat_{n}"] = th[i]
                row[f"err_{n}"] = th[i] - tt[i]
                row[f"diff_best_{n}"] = th[i] - tb[i]
            starts.append(row)
        row = dict(case=case, window=win, n_starts=len(R), n_minima=len(groups),
                   n_at_best=sum(r["_min"] == 0 for r in R), J_best=best["J"], bins=best["bins"])
        if Z is not None:
            S = Z[f"{win}_S"]
            d = Z[f"{win}_ydata"] - Z[f"{win}_y"]
            pred = np.linalg.lstsq(S, d, rcond=None)[0]
            s = 1e-3 * best["P_peak"]
            crb = np.sqrt(np.diag(np.linalg.inv(S.T @ S / s ** 2)))
            row["max_mismatch_rel"] = float(np.abs(d).max() / best["P_peak"])
            for i, n in enumerate(NAMES):
                meas = tb[i] - tt[i]
                ok = abs(meas - pred[i]) <= 0.2 * abs(pred[i]) or (abs(meas) <= 1e-4 and abs(pred[i]) <= 1e-4)
                row.update({f"meas_{n}": meas, f"pred_{n}": pred[i], f"gate_{n}": bool(ok),
                            f"crb_{n}_s1e-3": crb[i]})
            row["gate_all"] = all(row[f"gate_{n}"] for n in NAMES)
        bias.append(row)
if starts:
    write(starts, "i1a_starts.csv")
if bias:
    write(bias, "i1a_bias.csv")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 9})
fig, ax = plt.subplots(2, 4, figsize=(17, 7))
for col, case in enumerate(CASES):
    for row_i, win in enumerate(("prepeak", "full")):
        a = ax[row_i, col]
        for r, c in zip(runs.get((case, win), []), C):
            J = [e["J"] for e in r["log"] if e["accepted"] and e["J"] is not None]
            a.semilogy(np.arange(len(J)), np.maximum(J, 1e-12), "o-", ms=2.5, lw=0.9, color=c,
                       label="start %s" % r["tag"].split("_s")[-1])
        a.set(title=f"{case}, {win}", xlabel="accepted iterate", ylabel="J (s = 1e-3 P_peak)")
        if row_i == 0 and col == 0:
            a.legend(frameon=False, fontsize=6, ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"i1a.{ext}"), dpi=180)
print("done")
