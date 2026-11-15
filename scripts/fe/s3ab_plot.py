"""s3ab_plot.py -- S3a/S3b summary tables and figure.

Units: mm, N.  Light.
    python scripts/fe/s3ab_plot.py <coarse rundir> [outdir=results/solver/S3_tpeel]

Inputs: <outdir>/s3a_elastic.csv (scripts/fe/s3a_compare.py), and from the coarse rundir
(scripts/fe/run_s3b_coarse.sh + the corotational rerun): c_s40_response.csv, c_s40.sta, c_s40.msg,
c_s40_<tag>_jax.csv/.json.  Writes s3b_coarse.csv (one row per solver run), s3b_coarse_pd.npz
(compact P-delta curves) and s3ab.png/.pdf.
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
C = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (dataviz default palette)

ab = np.genfromtxt(os.path.join(RUN, "c_s40_response.csv"), delimiter=",", names=True)
msg = open(os.path.join(RUN, "c_s40.msg")).read()
num = lambda pat: int(re.findall(pat, msg)[-1])
Ppk = ab["P"].max()
grid = np.linspace(0, ab["delta"].max(), 2000)
pa = np.interp(grid, ab["delta"], ab["P"])
rows = [dict(solver="abaqus", frame="COH2D4", history="abaqus", increments=num(r"TOTAL OF\s+(\d+)\s+INCREMENTS"),
             cutbacks=num(r"(\d+)\s+CUTBACKS"), newton_total=num(r"(\d+)\s+ITERATIONS INCLUDING"),
             newton_max=np.nan, passes_total=np.nan, ls_total=np.nan,
             wall=float(re.findall(r"WALLCLOCK TIME \(SEC\) =\s+(\d+)", msg)[-1]),
             P_peak=Ppk, err_P_peak=0.0, err_delta_peak=0.0, max_dP=0.0, rms_dP=0.0, max_dP_prepeak=0.0)]
curves = dict(abaqus_delta=ab["delta"], abaqus_P=ab["P"])
for js in sorted(glob.glob(os.path.join(RUN, "c_s40_*_jax.json"))):
    s = json.load(open(js))
    d = np.genfromtxt(js[:-5] + ".csv", delimiter=",", names=True)
    e = (np.interp(grid, d["delta"], d["P"]) - pa) / Ppk
    k = int(d["P"].argmax())
    rows.append(dict(solver="tslid.fe:" + s["tag"], frame=s["frame"], history=s["history"], increments=s["increments"],
                     cutbacks=s["cutbacks"], newton_total=s["newton_total"], newton_max=s["newton_max"],
                     passes_total=s["passes_total"], ls_total=s["ls_total"], wall=s["wall_run"],
                     P_peak=d["P"][k], err_P_peak=(d["P"][k] - Ppk) / Ppk,
                     err_delta_peak=d["delta"][k] - ab["delta"][ab["P"].argmax()],
                     max_dP=float(np.abs(e).max()), rms_dP=float(np.sqrt(np.mean(e ** 2))),
                     max_dP_prepeak=float(np.abs(e[grid < ab["delta"][ab["P"].argmax()]]).max())))
    curves[s["tag"] + "_delta"], curves[s["tag"] + "_P"] = d["delta"], d["P"]
with open(os.path.join(OUT, "s3b_coarse.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
np.savez_compressed(os.path.join(OUT, "s3b_coarse_pd.npz"), **{k: np.asarray(v, dtype=np.float32) for k, v in curves.items()})
for r in rows:
    print(r)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
sa = np.genfromtxt(os.path.join(OUT, "s3a_elastic.csv"), delimiter=",", names=True, dtype=None, encoding=None)
for (tag, lab), c, mk in zip((("el16_svk", "SVK, cohesive 'global' frame"), ("co_el16_svk", "SVK, COH2D4 kinematics"),
                              ("co_el16_hencky", "Hencky, COH2D4 kinematics")), C, ("o", "s", "^")):
    m = sa["solver"] == f"tslid.fe:{tag}"
    ax[0].loglog(sa["RF2_abq"][m], sa["err_RF2"][m], marker=mk, ms=5, lw=1.5, color=c, mfc="w", label=lab)
ax[0].axhline(1e-5, color="0.4", ls=":", lw=1)
ax[0].text(9e3, 1.15e-5, "gate 1e-5", ha="right", fontsize=8, color="0.3")
ax[0].axvline(562.1, color="0.6", ls="--", lw=1)
ax[0].text(590, 4e-6, "T-peel peak\n(sw_s40)", fontsize=8, color="0.3")
ax[0].set(xlabel="P (N), elastic bond, NLGEOM", ylabel=r"$|RF2 - RF2_{ABAQUS}| / RF2_{ABAQUS}$",
          title="(a) S3a elastic gate, production mesh", ylim=(2e-6, 2e-3))
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xticks([300, 1000, 3000, 7000])
ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
ax[1].plot(ab["delta"], ab["P"], color="0.2", lw=3, alpha=0.3, label="ABAQUS (coarse deck)")
d = np.genfromtxt(os.path.join(RUN, "c_s40_mono_coro_jax.csv"), delimiter=",", names=True)
ax[1].plot(d["delta"], d["P"], color=C[1], lw=1.2, label="tslid.fe monolithic")
ax[1].set(xlabel=r"$\Delta$ (mm)", ylabel="P (N)", title=r"(b) S3b coarse T-peel, $\sigma_c$ = 40, $\alpha$ = 0")
ax[1].legend(frameon=False, fontsize=8)
for (tag, lab), c, ls in zip((("monolithic_svk_global", "cohesive 'global' frame"), ("mono_coro", "COH2D4 kinematics")),
                             (C[0], C[1]), ("-", "-")):
    p = os.path.join(RUN, f"c_s40_{tag}_jax.csv")
    if os.path.exists(p):
        d = np.genfromtxt(p, delimiter=",", names=True)
        ax[2].plot(grid, 100 * (np.interp(grid, d["delta"], d["P"]) - pa) / Ppk, color=c, lw=1.0, ls=ls, label=lab)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$(P - P_{ABAQUS}) / P_{peak}$ (%)", title="(c) coarse: difference to ABAQUS")
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s3ab.{ext}"), dpi=220)
print("done")
