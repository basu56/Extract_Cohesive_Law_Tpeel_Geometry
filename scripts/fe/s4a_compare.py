"""s4a_compare.py -- S4a: the 23 sweep runs at G_c = 0.4 (21 laws), tslid.fe vs ABAQUS, and the D18
matched-peak pairs in both solvers.  Light.

Units: mm, N.
    python scripts/fe/s4a_compare.py <tslid_runs root> [outdir=results/solver/S4a_sweeps]

Runs: tslid.fe <root>/solver_S4a/<job>_default (D24 defaults, scripts/fe/run_s3c_prod.sh); ABAQUS
<root>/phase1a_sigc_sweep/sw_s* (sigma_c sweep, alpha 0) and phase1b_alpha_sweep/as_s* (alpha sweeps at
sigma_c 40 and 21).  as_s40_a0 / as_s21_a0 repeat the laws of sw_s40 / sw_s21 (23 runs, 21 laws).
Writes
  s4a_compare.csv   one row per run: compare_solvers.py fields (load gates, D26 windowed E4-E6 and the
                    default-window values), sigma_c, alpha
  s4a_pairs.csv     per D18 matched-peak pair (results/phase1b_alpha/README.md) and solver: the pair's
                    pre-peak difference d(delta) = (P_other - P_ref) / P_peak,ref over
                    0 <= delta <= max(delta_peak of both laws), on the union of all four records' deltas:
                    max |d| and its delta, rms; and the agreement between solvers,
                    max |d_tslid - d_ABAQUS| (absolute, in P_peak) and relative to max |d_ABAQUS|
  s4a_duplicates.csv  same law run twice (as_s40_a0 vs sw_s40, as_s21_a0 vs sw_s21), per solver
  s4a.png/.pdf      (a) max |dP|/P_peak vs ABAQUS per run; (b) pair differences, tslid.fe (lines) vs
                    ABAQUS (markers); (c) their difference
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.analysis.compare_solvers import compare, fmt  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S4a_sweeps"
os.makedirs(OUT, exist_ok=True)
NEW = os.path.join(ROOT, "solver_S4a")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1495b"]
PAIRS = [("sw_s25", "as_s21_am1p5"), ("sw_s25", "as_s40_a5"), ("sw_s25", "as_s40_a7"),   # D18 / phase1b README
         ("sw_s30", "as_s21_am3"), ("sw_s30", "as_s40_a3")]
DUPS = [("sw_s40", "as_s40_a0"), ("sw_s21", "as_s21_a0")]


def abq_dir(job):
    return os.path.join(ROOT, "phase1a_sigc_sweep" if job.startswith("sw_") else "phase1b_alpha_sweep")


def done(rd, job):
    p = os.path.join(rd, job + ".sta")
    return os.path.exists(p) and "COMPLETED SUCCESSFULLY" in open(p).read()


def write(rows, name):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


jobs = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "phase1a_sigc_sweep", "sw_s*.inp")) +
              glob.glob(os.path.join(ROOT, "phase1b_alpha_sweep", "as_s*.inp")))
rows, missing = [], []
for j in jobs:
    if not done(NEW, j + "_default"):
        missing.append(j)
        continue
    o = compare(abq_dir(j), j, NEW, j + "_default")
    print(fmt(o))
    p = json.load(open(os.path.join(abq_dir(j), j + ".json")))["params"]
    s = json.load(open(os.path.join(NEW, j + "_default_jax.json")))
    r = dict(job=j, sigc=p["sigc"], alpha=p["alpha"], Gc=p["Gc"], increments=s["increments"], cutbacks=s["cutbacks"],
             iterations_all=s["iterations_all"], wall=s["wall_run"], linsolve=s["linsolve"],
             linsolve_perturbed=s["linsolve_perturbed"])
    for k, v in o.items():
        if k == "gates":
            r.update({"gate_" + g: bool(x) for g, x in v.items()})
        elif not isinstance(v, (dict, list)):
            r[k] = v
    rows.append(r)
print("missing / incomplete:", missing)
if rows:
    write(rows, "s4a_compare.csv")

# ---- D18 matched-peak pairs
curves = {}


def rec(solver, job):
    if (solver, job) not in curves:
        curves[(solver, job)] = load(abq_dir(job), job) if solver == "abaqus" else load(NEW, job + "_default")
    return curves[(solver, job)]


prow, pdat = [], []
for a, b in PAIRS:
    if a in missing or b in missing:
        continue
    A = {s: rec(s, a) for s in ("abaqus", "tslid.fe")}
    B = {s: rec(s, b) for s in ("abaqus", "tslid.fe")}
    Ppk = A["abaqus"]["P"].max()
    d_hi = max(x["d"][int(np.argmax(x["P"]))] for x in list(A.values()) + list(B.values()))
    du = np.unique(np.concatenate([x["d"][x["d"] <= d_hi] for x in list(A.values()) + list(B.values())]))
    dd = {s: (np.interp(du, B[s]["d"], B[s]["P"]) - np.interp(du, A[s]["d"], A[s]["P"])) / Ppk for s in A}
    g = np.linspace(0.0, d_hi, 20000)
    for s in A:
        e = dd[s]
        k = int(np.argmax(np.abs(e)))
        eg = (np.interp(g, B[s]["d"], B[s]["P"]) - np.interp(g, A[s]["d"], A[s]["P"])) / Ppk
        prow.append(dict(ref=a, other=b, solver=s, window_end=d_hi, max_abs_d=float(np.abs(e[k])), d_signed=float(e[k]),
                         delta_at_max=float(du[k]), rms_d=float(np.sqrt(np.mean(eg ** 2)))))
    diff = dd["tslid.fe"] - dd["abaqus"]
    k = int(np.argmax(np.abs(diff)))
    prow.append(dict(ref=a, other=b, solver="tslid.fe - abaqus", window_end=d_hi, max_abs_d=float(np.abs(diff[k])),
                     d_signed=float(diff[k]), delta_at_max=float(du[k]),
                     rel_to_signal=float(np.abs(diff[k]) / np.abs(dd["abaqus"]).max()),
                     signal_peak_ratio=float(np.abs(dd["tslid.fe"]).max() / np.abs(dd["abaqus"]).max())))
    pdat.append((a, b, du, dd))
if prow:
    write(prow, "s4a_pairs.csv")
    for r in prow:
        print(r)

drow = []
for a, b in DUPS:
    if a in missing or b in missing:
        continue
    for s in ("abaqus", "tslid.fe"):
        A, B = rec(s, a), rec(s, b)
        same_d = len(A["d"]) == len(B["d"]) and np.array_equal(A["d"], B["d"])
        du = np.union1d(A["d"], B["d"])
        drow.append(dict(job_a=a, job_b=b, solver=s, identical_delta=bool(same_d),
                         max_dP=float(np.abs(np.interp(du, B["d"], B["P"]) - np.interp(du, A["d"], A["P"])).max()
                                  / A["P"].max())))
if drow:
    write(drow, "s4a_duplicates.csv")
    for r in drow:
        print(r)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
groups = [("sigma_c sweep, alpha 0", [r for r in rows if r["job"].startswith("sw_")], "sigc", C[0], "o"),
          ("alpha sweep, sigma_c 40", [r for r in rows if r["job"].startswith("as_s40")], "alpha", C[1], "s"),
          ("alpha sweep, sigma_c 21", [r for r in rows if r["job"].startswith("as_s21")], "alpha", C[2], "^")]
x0 = 0
ticks, labels = [], []
for lab, rr, key, c, mk in groups:
    rr = sorted(rr, key=lambda r: r[key])
    xs = np.arange(len(rr)) + x0
    ax[0].semilogy(xs, [r["max_dP"] for r in rr], mk, color=c, mfc="w", ms=6, label=lab)
    ticks += list(xs)
    labels += [("%g" % r[key]) for r in rr]
    x0 += len(rr) + 1
ax[0].axhline(5e-3, color="0.4", ls=":", lw=1)
ax[0].axhline(1e-3, color="0.6", ls="--", lw=1)
ax[0].text(0, 5.6e-3, "gate 0.5 %", fontsize=8, color="0.3")
ax[0].text(0, 1.12e-3, "target 0.1 %", fontsize=8, color="0.3")
ax[0].set_xticks(ticks)
ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set(xlabel=r"$\sigma_c$ (MPa) | $\alpha$ | $\alpha$", ylabel=r"max $|P - P_{ABAQUS}| / P_{peak}$",
          title="(a) 23 sweep runs, tslid.fe vs ABAQUS", ylim=(5e-6, 2e-2))
ax[0].legend(frameon=False, fontsize=8, loc="center right")
for (a, b, du, dd), c in zip(pdat, C):
    lab = "%s vs %s" % (b, a)
    ax[1].plot(du, 1e2 * dd["tslid.fe"], color=c, lw=1.2, label=lab)
    sel = slice(None, None, max(1, len(du) // 40))
    ax[1].plot(du[sel], 1e2 * dd["abaqus"][sel], "o", color=c, mfc="w", ms=3.5)
    ax[2].plot(du, 1e2 * (dd["tslid.fe"] - dd["abaqus"]), color=c, lw=0.9)
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$(P_{other} - P_{ref}) / P_{peak}$ (%)",
          title="(b) D18 matched-peak pairs, pre-peak\nlines tslid.fe, markers ABAQUS")
ax[1].legend(frameon=False, fontsize=7)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xlabel=r"$\Delta$ (mm)", ylabel="pair difference: tslid.fe − ABAQUS (% of P_peak)",
          title="(c) agreement of the pair differences")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s4a.{ext}"), dpi=200)
print("done")
