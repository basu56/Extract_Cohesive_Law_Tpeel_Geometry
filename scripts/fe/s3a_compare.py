"""s3a_compare.py -- S3a elastic gate: tslid.fe vs ABAQUS on the production T-peel mesh (sw_s40).

Units: mm, N, rad, MPa.  Light (seconds).
    python scripts/fe/s3a_compare.py <rundir> [outdir=results/solver/S3_tpeel]

Pairs compared (ABAQUS job <- tslid.fe tag; files from scripts/fe/run_s3a*.sh):
    s3a_lin_tight  <- co_lin            NLGEOM = NO, linear kinematics, COH2D4 linearised kinematics
    s3a_lin_tight  <- lin               same, cohesive "global" frame (first attempt)
    s3a_el16_tight <- co_el16_svk       NLGEOM, SVK, COH2D4 corotational kinematics
    s3a_el16_tight <- co_el16_hencky    NLGEOM, Hencky, corotational
    s3a_el16_tight <- el16_svk          NLGEOM, SVK, cohesive "global" frame (first attempt)
    s3a_el_tight   <- svk               4-increment deck, SVK, "global" (first attempt)
    s3a_el_tight   <- abaqus default controls (ABAQUS's own Newton tolerance)
Gate (S3_INSTRUCTIONS.md): RF2 at RP1, mouth openings and RP1 rotation within 1e-5 relative.
Writes s3a_elastic.csv (one row per pair and frame).
"""
import csv
import os
import sys

import numpy as np

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
os.makedirs(OUT, exist_ok=True)
T = 96.0
PAIRS = [("s3a_lin_tight", "co_lin"), ("s3a_lin_tight", "lin"),
         ("s3a_el16_tight", "co_el16_svk"), ("s3a_el16_tight", "co_el16_hencky"), ("s3a_el16_tight", "el16_svk"),
         ("s3a_el_tight", "svk")]


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def mouth_labels(job):
    lines = open(os.path.join(RUN, job + ".inp")).read().splitlines()
    out = {}
    for k, l in enumerate(lines):
        for nm in ("MOUTH_U", "MOUTH_L"):
            if l.strip() == f"*Nset, nset={nm}":
                out[nm] = int(lines[k + 1].split(",")[0])
    return out["MOUTH_U"], out["MOUTH_L"]


_cache = {}


def abaqus(job):
    if job in _cache:
        return _cache[job]
    h1 = read(os.path.join(RUN, f"{job}_hist_2000001.csv"))
    h2 = read(os.path.join(RUN, f"{job}_hist_2000002.csv"))
    nd = read(os.path.join(RUN, f"{job}_nodes.csv"))
    cp = os.path.join(RUN, f"{job}_coh.csv")
    coh = read(cp) if os.path.exists(cp) and os.path.getsize(cp) > 0 else None
    _cache[job] = (h1, h2, nd, coh, mouth_labels(job))
    return _cache[job]


def frame_values(job, lam):
    h1, h2, nd, coh, (mu_lab, ml_lab) = abaqus(job)
    i = int(np.argmin(np.abs(h1["time"] / T - lam)))
    assert abs(h1["time"][i] / T - lam) < 1e-9, (job, lam)
    fr = int(np.argmin(np.abs(np.unique(nd["lam"]) / T - lam)))
    s = nd[np.isclose(nd["lam"] / T, lam, atol=1e-9)]
    idx = {int(l): j for j, l in enumerate(s["node"])}
    out = dict(rf2=h1["RF2"][i], ur3=h1["UR3"][i], u1=h1["U1"][i], rp2_rf2=h2["RF2"][i],
               mu=s["U2"][idx[mu_lab]], ml=s["U2"][idx[ml_lab]], labels=s["node"], U=np.c_[s["U1"], s["U2"]])
    if coh is not None:
        c = coh[np.isclose(coh["time"] / T, lam, atol=1e-9)]
        if len(c):
            o = np.lexsort((c["ip"], c["el"]))
            out["S22"] = c["S_S22"][o].reshape(-1, 2)
            out["S12"] = c["S_S12"][o].reshape(-1, 2)
    return out


def rel(a, b):
    return abs(a - b) / abs(b)


rows = []
for job, tag in PAIRS:
    p = os.path.join(RUN, f"s3a_jax_{tag}.csv")
    if not os.path.exists(p):
        print("missing", p)
        continue
    jx = read(p)
    z = np.load(os.path.join(RUN, f"s3a_jax_{tag}_U.npz"))
    jidx = {int(l): j for j, l in enumerate(z["labels"])}
    for k, lam in enumerate(np.atleast_1d(jx["lam"])):
        a = frame_values(job, lam)
        g = lambda key: float(np.atleast_1d(jx[key])[k])
        Uj = z["U"][k][[jidx[int(l)] for l in a["labels"]]]
        open_a = a["mu"] - a["ml"]
        row = dict(abaqus=job, solver=f"tslid.fe:{tag}", lam=lam, RF2_abq=a["rf2"], RF2=g("rp1_rf2"),
                   err_RF2=rel(g("rp1_rf2"), a["rf2"]), err_UR3=rel(g("rp1_ur3"), a["ur3"]),
                   err_mouth_open=rel(g("mouth_u_u2") - g("mouth_l_u2"), open_a),
                   err_mouth_U=rel(g("mouth_u_u2"), a["mu"]),
                   err_mouth_L_over_open=abs(g("mouth_l_u2") - a["ml"]) / abs(open_a),
                   err_RP1_U1=abs(g("rp1_u1") - a["u1"]),
                   err_U_field=float(np.abs(Uj - a["U"]).max() / np.abs(a["U"]).max()),
                   err_tn=np.nan, err_ts=np.nan,
                   jax_equilibrium=abs(g("rp1_rf2") + g("rp2_rf2")) / abs(g("rp1_rf2")),
                   newton=int(g("iterations")))
        if "S22" in a and "tn" in z.files:
            sc = np.abs(a["S22"]).max()
            row["err_tn"] = float(np.abs(z["tn"][k] - a["S22"]).max() / sc)
            row["err_ts"] = float(np.abs(z["ts"][k] - a["S12"]).max() / sc)
        rows.append(row)
for lam in (0.25, 0.5, 0.75, 1.0):
    a, b = frame_values("s3a_el_default", lam), frame_values("s3a_el_tight", lam)
    rows.append(dict(abaqus="s3a_el_tight", solver="abaqus:default-controls", lam=lam, RF2_abq=b["rf2"], RF2=a["rf2"],
                     err_RF2=rel(a["rf2"], b["rf2"]), err_UR3=rel(a["ur3"], b["ur3"]),
                     err_mouth_open=rel(a["mu"] - a["ml"], b["mu"] - b["ml"]), err_mouth_U=rel(a["mu"], b["mu"]),
                     err_mouth_L_over_open=abs(a["ml"] - b["ml"]) / abs(b["mu"] - b["ml"]),
                     err_RP1_U1=abs(a["u1"] - b["u1"]),
                     err_U_field=float(np.abs(a["U"] - b["U"]).max() / np.abs(b["U"]).max()),
                     err_tn=np.nan, err_ts=np.nan, jax_equilibrium=np.nan, newton=-1))
keys = ["err_RF2", "err_UR3", "err_mouth_open", "err_mouth_L_over_open", "err_RP1_U1", "err_U_field", "err_tn", "err_ts"]
print(f"{'abaqus':16s} {'solver':26s} {'lam':>6s} " + " ".join(f"{k[4:]:>13s}" for k in keys))
for r in rows:
    print(f"{r['abaqus']:16s} {r['solver']:26s} {r['lam']:6.3f} " + " ".join(f"{r[k]:13.3e}" for k in keys))
with open(os.path.join(OUT, "s3a_elastic.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
print("wrote", os.path.join(OUT, "s3a_elastic.csv"))
