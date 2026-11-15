"""s3a_compare.py -- S3a elastic gate: tslid.fe vs ABAQUS on the production T-peel mesh.

Units: mm, N, rad.  Light (seconds).
    python scripts/fe/s3a_compare.py <rundir> [outdir=results/solver/S3_tpeel]

Inputs in <rundir> (scripts/fe/run_s3a.sh):
    s3a_el_{tight,default}_hist_2000001.csv / _hist_2000002.csv  RP1 / RP2 history (U, UR3, RF, RM3)
    s3a_el_{tight,default}_nodes.csv                              nodal U, RF per frame
    s3a_jax_{svk,hencky}.csv, s3a_jax_{svk,hencky}_U.npz          tslid.fe
Gate (S3_INSTRUCTIONS.md): RF2 at RP1, mouth openings and RP1 rotation within 1e-5 relative.
Writes s3a_elastic.csv (one row per solver pair and frame).
"""
import csv
import os
import sys

import numpy as np

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
os.makedirs(OUT, exist_ok=True)
MOUTH_U, MOUTH_L = None, None


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def abaqus(job):
    h1 = read(os.path.join(RUN, f"{job}_hist_2000001.csv"))
    h2 = read(os.path.join(RUN, f"{job}_hist_2000002.csv"))
    nd = read(os.path.join(RUN, f"{job}_nodes.csv"))
    T = 96.0
    frames = sorted(set(nd["frame"].tolist()))
    U = {}
    for f in frames:
        s = nd[nd["frame"] == f]
        U[f] = (float(s["lam"][0]) / T, s["node"], np.c_[s["U1"], s["U2"]])
    return h1, h2, U


def mouth_nodes(labels):
    """MOUTH_U / MOUTH_L labels from the deck written next to the runs."""
    deck = os.path.join(RUN, "s3a_el_tight.inp")
    lines = open(deck).read().splitlines()
    out = {}
    for k, l in enumerate(lines):
        for nm in ("MOUTH_U", "MOUTH_L"):
            if l.strip() == f"*Nset, nset={nm}":
                out[nm] = int(lines[k + 1].split(",")[0])
    return out["MOUTH_U"], out["MOUTH_L"]


rows = []
mu_lab, ml_lab = mouth_nodes(None)
ref = {job: abaqus(job) for job in ("s3a_el_tight", "s3a_el_default")}
h1t, h2t, Ut = ref["s3a_el_tight"]


def frame_values(h1, h2, U, lam):
    i = int(np.argmin(np.abs(h1["time"] / 96.0 - lam)))
    assert abs(h1["time"][i] / 96.0 - lam) < 1e-9, (lam, h1["time"])
    f = next(k for k, v in U.items() if abs(v[0] - lam) < 1e-9)
    _, labs, uu = U[f]
    idx = {int(l): j for j, l in enumerate(labs)}
    return dict(rf2=h1["RF2"][i], ur3=h1["UR3"][i], u2=h1["U2"][i], rp2_rf2=h2["RF2"][i],
                mu=uu[idx[mu_lab], 1], ml=uu[idx[ml_lab], 1], labels=labs, U=uu)


def rel(a, b):
    return abs(a - b) / abs(b)


for tag in ("svk", "hencky"):
    p = os.path.join(RUN, f"s3a_jax_{tag}.csv")
    if not os.path.exists(p):
        print("missing", p)
        continue
    jx = read(p)
    z = np.load(os.path.join(RUN, f"s3a_jax_{tag}_U.npz"))
    jidx = {int(l): j for j, l in enumerate(z["labels"])}
    for k, lam in enumerate(np.atleast_1d(jx["lam"])):
        for job in ("s3a_el_tight", "s3a_el_default"):
            a = frame_values(*ref[job], lam)
            Uj = z["U"][k][[jidx[int(l)] for l in a["labels"]]]
            row = dict(solver=f"tslid.fe-{tag}", abaqus=job, lam=lam, RF2_abq=a["rf2"],
                       RF2=float(jx["rp1_rf2"][k]),
                       err_RF2=rel(float(jx["rp1_rf2"][k]), a["rf2"]),
                       err_UR3=rel(float(jx["rp1_ur3"][k]), a["ur3"]),
                       err_mouth_U=rel(float(jx["mouth_u_u2"][k]), a["mu"]),
                       err_mouth_L=abs(float(jx["mouth_l_u2"][k]) - a["ml"]) / abs(a["mu"] - a["ml"]),
                       err_mouth_open=rel(float(jx["mouth_u_u2"][k] - jx["mouth_l_u2"][k]), a["mu"] - a["ml"]),
                       err_U_field=float(np.abs(Uj - a["U"]).max() / np.abs(a["U"]).max()),
                       abq_equilibrium=abs(a["rf2"] + a["rp2_rf2"]) / abs(a["rf2"]),
                       jax_equilibrium=abs(float(jx["rp1_rf2"][k] + jx["rp2_rf2"][k])) / abs(float(jx["rp1_rf2"][k])),
                       newton=int(jx["iterations"][k]), wall=float(jx["wall"][k]))
            rows.append(row)
            print(" ".join(f"{k_}={v:.3e}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))

# ABAQUS default vs tight controls (its own Newton tolerance)
for lam in sorted(set(r["lam"] for r in rows)):
    a, b = frame_values(*ref["s3a_el_default"], lam), frame_values(*ref["s3a_el_tight"], lam)
    rows.append(dict(solver="abaqus-default", abaqus="s3a_el_tight", lam=lam, RF2_abq=b["rf2"], RF2=a["rf2"],
                     err_RF2=rel(a["rf2"], b["rf2"]), err_UR3=rel(a["ur3"], b["ur3"]),
                     err_mouth_U=rel(a["mu"], b["mu"]), err_mouth_L=abs(a["ml"] - b["ml"]) / abs(b["mu"] - b["ml"]),
                     err_mouth_open=rel(a["mu"] - a["ml"], b["mu"] - b["ml"]),
                     err_U_field=float(np.abs(a["U"] - b["U"]).max() / np.abs(b["U"]).max()),
                     abq_equilibrium=np.nan, jax_equilibrium=np.nan, newton=-1, wall=np.nan))
    print("abaqus default vs tight lam", lam, rows[-1]["err_RF2"], rows[-1]["err_U_field"])

with open(os.path.join(OUT, "s3a_elastic.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
print("wrote", os.path.join(OUT, "s3a_elastic.csv"))
