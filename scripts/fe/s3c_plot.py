"""s3c_plot.py -- S3c figure and compact result files.  Light (post-processing).

Units: mm, N, s.
    python scripts/fe/s3c_plot.py <solver_S3c rundir> [outdir=results/solver/S3_tpeel]

Inputs (all under <rundir>, see results/solver/S3_tpeel/README.md):
    summary/s3c_ladder_runs.csv     s3c_summary.py on ladder/ (prefix s3c_ladder)
    summary/s3c_tol_runs.csv        s3c_summary.py on tol/ with S3C_REF=tol_ref (prefix s3c_tol)
    prod/sw_s40_<tag>_jax.csv/.json (tags prod_t1: rtol 1e-10, 1 thread; prod_t1_r6: rtol 1e-6, 1 thread
    = final S3c configuration; prod_t4_r6: 4 threads), prod/sw_s40_response.csv + _profiles.npz (ABAQUS)
Writes s3c_prod_pd.npz (production P-delta of both codes, float32), s3c_prod.csv (one row per code),
and s3c.png/.pdf: (a) coarse wall time per S3c item, 1 thread, split converged / abandoned attempts,
ABAQUS for reference; (b) tolerance study: max |P - P_tight| / P_peak at identical increments vs rtol;
(c) production sw_s40: (P - P_ABAQUS) / P_peak vs delta.
"""
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load as cld_load  # noqa: E402

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
C = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (dataviz default palette)


def rows(p):
    return {r["tag"]: r for r in csv.DictReader(open(p))}


def rd(p):
    return np.genfromtxt(p, delimiter=",", names=True)


lad = rows(os.path.join(RUN, "summary", "s3c_ladder_runs.csv"))
tol = rows(os.path.join(RUN, "summary", "s3c_tol_runs.csv"))

# ---- production
P = os.path.join(RUN, "prod")
ab = rd(os.path.join(P, "sw_s40_response.csv"))
TAGS = ["prod_t1", "prod_t1_r6", "prod_t4_r6"]                  # rtol 1e-10 / 1e-6, 1 / 4 threads
FINAL = "prod_t1_r6"
msg = open(os.path.join(RUN, "..", "phase1a_sigc_sweep", "sw_s40.msg")).read()
num = lambda pat: int(re.findall(pat, msg)[-1])
Ppk = ab["P"].max()
d_win = cld_load(P, "sw_s40")["d_end"]
ka = int(np.argmax(ab["P"]))
prod = [dict(code="abaqus", increments=num(r"TOTAL OF\s+(\d+)\s+INCREMENTS"), cutbacks=num(r"(\d+)\s+CUTBACKS"),
             iterations_all=num(r"(\d+)\s+ITERATIONS INCLUDING"),
             wall=float(re.findall(r"WALLCLOCK TIME \(SEC\) =\s+(\d+)", msg)[-1]), P_peak=Ppk,
             delta_peak=ab["delta"][ka], threads=1)]
curves = dict(abaqus_delta=ab["delta"].astype(np.float32), abaqus_P=ab["P"].astype(np.float32))
for tg in TAGS:
    d = rd(os.path.join(P, f"sw_s40_{tg}_jax.csv"))
    s = json.load(open(os.path.join(P, f"sw_s40_{tg}_jax.json")))
    grid = np.linspace(0.0, min(ab["delta"].max(), d["delta"].max()), 6000)
    e = (np.interp(grid, d["delta"], d["P"]) - np.interp(grid, ab["delta"], ab["P"])) / Ppk
    w = grid <= d_win
    k = int(np.argmax(d["P"]))
    prod.append(dict(code="tslid.fe:" + tg, threads=s.get("affinity"), rtol=s["rtol"], increments=s["increments"],
                     cutbacks=s["cutbacks"], iterations_all=s["iterations_all"], newton_total=s["newton_total"],
                     wall=s["wall_run"], P_peak=d["P"][k], delta_peak=d["delta"][k], n_q=s["n_q"], n_sys=s["n_sys"],
                     t_hessian=s["timers"]["hessian"], t_solve=s["timers"]["solve"],
                     t_value_and_grad=s["timers"]["value_and_grad"], t_history=s["timers"]["history"],
                     t_failed=s["failed_wall"], t_io=s["t_io"], wall_setup=s["wall_setup"],
                     wall_first_call=s["wall_first_call"], err_P_peak=(d["P"][k] - Ppk) / Ppk, d_window=d_win,
                     max_dP_win=float(np.abs(e[w]).max()), rms_dP_win=float(np.sqrt(np.mean(e[w] ** 2))),
                     max_dP_full=float(np.abs(e).max()), linsolve_perturbed=s.get("linsolve_perturbed")))
    curves[tg + "_delta"], curves[tg + "_P"] = d["delta"].astype(np.float32), d["P"].astype(np.float32)
    if tg == FINAL:
        g_fin, e_fin, w_fin, r_fin = grid, e, w, prod[-1]
keys = []
for r in prod:
    keys += [x for x in r if x not in keys]
with open(os.path.join(OUT, "s3c_prod.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=keys)
    wr.writeheader()
    for r in prod:
        wr.writerow({x: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for x, v in r.items()})
np.savez_compressed(os.path.join(OUT, "s3c_prod_pd.npz"), **curves)
print(prod)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

# (a) ladder
steps = [("L0_base", "S3b path"), ("L1_en", "+ jitted energy Newton"), ("L2_div", "+ divergence check"),
         ("L3_el", "+ element assembly"), ("L4_elc", "+ IM condensation"), ("L5_elcp", "+ PARDISO (sym, reuse)")]
y = np.arange(len(steps))[::-1]
for yi, (tg, lab) in zip(y, steps):
    vals = [(float(lad[f"{tg}_{s_}"]["t_increments"]), float(lad[f"{tg}_{s_}"]["failed_wall"]),
             float(lad[f"{tg}_{s_}"]["wall"])) for s_ in "AB" if f"{tg}_{s_}" in lad]
    conv, fail, wall = (np.mean([v[i] for v in vals]) for i in range(3))
    ax[0].barh(yi, conv, color=C[0], height=0.6, label="converged increments" if yi == y[0] else None)
    ax[0].barh(yi, fail, left=conv, color=C[1], height=0.6, label="abandoned attempts" if yi == y[0] else None)
    ax[0].text(wall + 8, yi, "%.0f s" % wall, va="center", fontsize=8, color="0.2")
abq = float(lad["abaqus"]["wall"])
ax[0].axvline(abq, color="0.4", ls=":", lw=1)
ax[0].text(abq + 8, y[0] + 0.45, "ABAQUS %.0f s" % abq, fontsize=8, color="0.3")
ax[0].set_ylim(y[-1] - 0.6, y[0] + 0.8)
ax[0].set_yticks(y)
ax[0].set_yticklabels([lab for _, lab in steps], fontsize=8)
ax[0].set(xlabel="wall time (s), coarse T-peel, 1 thread", title="(a) S3c items, cumulative")
ax[0].legend(frameon=False, fontsize=8, loc="lower right")

# (b) tolerance
rt = ["1e-9", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
err = [float(tol[f"tol_{r}_rep"]["max_dP_ref_same_t_win"]) for r in rt]
its = [int(float(tol[f"tol_{r}_rep"]["newton_total"])) for r in rt]
ax[1].loglog([float(r) for r in rt], err, marker="o", ms=5, lw=1.5, color=C[0], mfc="w")
for r, e_, n in zip(rt, err, its):
    first = r == rt[0]
    ax[1].annotate("%d it." % n, (float(r), e_), textcoords="offset points", xytext=(8, -3) if first else (-8, 7),
                   fontsize=7, color="0.3", ha="left" if first else "right")
ax[1].axhline(1e-5, color="0.4", ls=":", lw=1)
ax[1].text(1.2e-9, 1.3e-5, "criterion 1e-5", fontsize=8, color="0.3")
ax[1].set(xlabel="rtol (Newton, relative to the reaction)",
          ylabel=r"max $|P - P_{\mathrm{rtol}=10^{-10}}| / P_{peak}$",
          title="(b) tolerance, coarse, same increments")

# (c) production
ax[2].plot(g_fin[w_fin], 1e2 * e_fin[w_fin], color=C[0], lw=1.0)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$(P - P_{ABAQUS}) / P_{peak}$ (%)",
          title="(c) production sw_s40, final S3c path")
ax[2].text(0.98, 0.04, "peak %+.1e, max %.1e, rms %.1e\n%d s vs ABAQUS %d s (1 thread)" %
           (r_fin["err_P_peak"], r_fin["max_dP_win"], r_fin["rms_dP_win"], r_fin["wall"], prod[0]["wall"]),
           transform=ax[2].transAxes, ha="right", fontsize=8, color="0.2")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s3c.{ext}"), dpi=220)
print("done")
