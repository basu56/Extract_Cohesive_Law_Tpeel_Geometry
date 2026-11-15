"""s3d_traction_location.py -- where along the bond is the S3a cohesive-traction difference? (D23 item)

Units: mm, MPa.  Light.
    python scripts/fe/s3d_traction_location.py <S3a rundir> [outdir=results/solver/S3_tpeel] [--tag co_el16_svk]

S3a, elastic bond, NLGEOM, production sw_s40 mesh (scripts/fe/run_s3a*.sh): ABAQUS s3a_el16_tight vs
tslid.fe <tag>.  Per frame at T-peel load levels (P <= 1.5 P_peak, D23) and per cohesive integration
point (2 per element, nodal), dt_n = t_n,ours - S22_ABAQUS and dt_s = t_s - S12, normalised by
max |S22| of the frame (the S3a metric).  The ordering of ABAQUS points (element label, ip) is the
chain order checked in S3a (traction error <= 5e-8 in the linear case).
Writes s3d_traction_location.csv (per frame: location of the largest |dt_n|, |dt_s|, distance to the
crack tip (bond start, x = a0) and to the bond end (x = L), the share of sum |dt| within 2 mm of each
end) and s3d_traction_location.png (profiles of dt_n, dt_s along the bond).
"""
import argparse
import csv
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("outdir", nargs="?", default="results/solver/S3_tpeel")
ap.add_argument("--tag", default="co_el16_svk")
ap.add_argument("--job", default="s3a_el16_tight")
a = ap.parse_args()
T = 96.0
P_PEAK = 562.078                                   # sw_s40, ABAQUS (phase1a)
C = ["#2a78d6", "#eb6834", "#1baf7a"]

meta = json.load(open(os.path.join(a.run, a.job + ".json")))
a0, L = meta["params"]["a0"], meta["params"]["L"]
z = np.load(os.path.join(a.run, f"s3a_jax_{a.tag}_U.npz"))
xn = z["coh_x"]
xq = np.stack([xn[:-1], xn[1:]], axis=1)          # nodal integration points: element ends
coh = np.genfromtxt(os.path.join(a.run, a.job + "_coh.csv"), delimiter=",", names=True)
h1 = np.genfromtxt(os.path.join(a.run, a.job + "_hist_2000001.csv"), delimiter=",", names=True)
rows, prof = [], []
for k, lam in enumerate(z["lam"]):
    P = float(h1["RF2"][int(np.argmin(np.abs(h1["time"] / T - lam)))])
    if P > 1.5 * P_PEAK:
        continue
    c = coh[np.isclose(coh["time"] / T, lam, atol=1e-9)]
    o = np.lexsort((c["ip"], c["el"]))
    S22, S12 = c["S_S22"][o].reshape(-1, 2), c["S_S12"][o].reshape(-1, 2)
    sc = np.abs(S22).max()
    dn, ds = (z["tn"][k] - S22) / sc, (z["ts"][k] - S12) / sc
    r = dict(lam=float(lam), P=P, max_S22=float(sc))
    for nm, d in (("n", dn), ("s", ds)):
        i = np.unravel_index(np.argmax(np.abs(d)), d.shape)
        x = float(xq[i])
        tot = np.abs(d).sum()
        r.update(**{f"max_dt{nm}": float(np.abs(d[i])), f"x_max_dt{nm}": x, f"from_tip_dt{nm}": x - a0,
                    f"from_end_dt{nm}": L - x,
                    f"share_tip2mm_dt{nm}": float(np.abs(d)[xq <= a0 + 2.0].sum() / tot),
                    f"share_end2mm_dt{nm}": float(np.abs(d)[xq >= L - 2.0].sum() / tot)})
    i = np.unravel_index(np.argmax(np.abs(S22)), S22.shape)
    r["x_max_S22"] = float(xq[i])
    rows.append(r)
    prof.append((P, dn, ds))
with open(os.path.join(a.outdir, "s3d_traction_location.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
for (P, dn, ds), c in zip(prof, C):
    for axi, d in zip(ax, (dn, ds)):
        axi.plot(xq.ravel(), d.ravel(), ".", ms=2.5, color=c, label="P = %.0f N" % P)
for axi, nm in zip(ax, ("normal", "shear")):
    axi.axhline(0, color="0.6", lw=0.8)
    axi.set(xlabel="x along the bond (mm); crack tip at x = %.0f, bond end at %.1f" % (a0, L),
            ylabel=r"$(t_{%s} - t_{ABAQUS}) / \max|S_{22}|$" % ("n" if nm == "normal" else "s"),
            title="(%s) %s traction difference, S3a elastic, NLGEOM" % ("a" if nm == "normal" else "b", nm))
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, "s3d_traction_location.png"), dpi=200)
print("done")
