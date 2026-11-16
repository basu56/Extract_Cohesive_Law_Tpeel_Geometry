"""s5b_analysis.py -- S5b tables: adjoint vs direct, adjoint vs FD of J (replay), p-scaling, cost/memory.
Light.

Units: mm, N, s, MB.
    python scripts/fe/s5b_analysis.py <tslid_runs root> [outdir=results/solver/S5_sensitivity]

Inputs (scripts/fe/run_s5b.sh -> s5b_run.py, solver_S5b/):
  c_s40_t10   coarse, rtol 1e-10, alpha law, adjoint + direct p = 4.  Its states must be those of S5a's b1
              (deterministic: P compared bit for bit), so the S5a FD replay runs (solver_S5a/c_s40_fd_*,
              theta_j +/- h on b1's schedule) are perturbations of the same converged states.
  c_s40_pw30  coarse, rtol 1e-6, piecewise law with 30 segments (p = 31), adjoint + direct p = 4, 8, 16, 31
  c_s40_cost, sw_s40_cost   forward + stored states + adjoint only (rtol 1e-6)
J (D29): 1/2 w sum_k (y_k - y_k^ABAQUS)^2 over the bins of the pre-peak window and of the full window.
FD of J: J(theta +/- h) from the replay runs with the same observation matrix (same delta_n through the
last bin), (J+ - J-)/2h; a step is valid only if both runs keep b1's increments through the last bin.
Writes s5b_adjoint_vs_direct.csv, s5b_adjoint_vs_fd.csv, s5b_scaling.csv, s5b_cost.csv.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.fe.observation import DELTA_BIN, observation_matrix, observe, windows  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S5_sensitivity"
B = os.path.join(ROOT, "solver_S5b")
A = os.path.join(ROOT, "solver_S5a")
NAMES = ["ln_K0", "ln_sigc", "ln_Gc", "alpha"]


def js(name):
    p = os.path.join(B, name)
    return json.load(open(p)) if os.path.exists(p) else None


def write(rows, name):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    for r in rows:
        print(name, {k: (f"{v:.4g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def rd(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("time", "delta", "P")}


# ---------------------------------------------------------------- adjoint vs direct
avd = []
for tag in ("t10", "pw30", "cost"):
    for job in ("c_s40", "sw_s40"):
        r = js(f"{job}_{tag}_s5b.json")
        if r is None:
            continue
        for win, o in r["objectives"].items():
            row = dict(job=job, tag=tag, law=r["law"], rtol=r["rtol"], window=win, bins=o["bins"], J=o["J"])
            for k, v in o.items():
                if k.startswith("rel_err_p"):
                    row[k] = v
            avd.append(row)
if avd:
    write(avd, "s5b_adjoint_vs_direct.csv")

# ---------------------------------------------------------------- adjoint vs FD of J (replay of b1)
fdr = []
base = js("c_s40_t10_s5b.json")
if base is not None:
    z = np.load(os.path.join(B, "c_s40_t10_s5b.npz"))
    b1 = np.load(os.path.join(A, "c_s40_b1_sens.npz"))
    same = len(z["P"]) == len(b1["P"]) and np.array_equal(z["time"], b1["time"]) and np.array_equal(z["P"], b1["P"])
    print("t10 base bit-identical to S5a b1:", same)
    ref = load(os.path.join(ROOT, "solver_S3c"), "c_s40")
    W = windows(ref["d_end"], base["delta_peak_ref"], base["width"])
    rdd = np.r_[0.0, ref["d"]] if ref["d"][0] > 0 else ref["d"]
    rPP = np.r_[0.0, ref["P"]] if ref["d"][0] > 0 else ref["P"]
    for win, edges in W.items():
        O = observation_matrix(np.r_[0.0, z["delta"]], edges)[:, 1:]
        need = int(np.nonzero(O.any(axis=0))[0].max()) + 1                 # increments that enter J
        yd = observe(rdd, rPP, edges)

        def J(P):
            r = O @ P - yd
            return 0.5 * DELTA_BIN * float(r @ r)

        J0 = J(z["P"])
        gA = np.array(base["objectives"][win]["adjoint"])
        for j, nm in enumerate(NAMES):
            for h in ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7"):
                fp, fm = (os.path.join(A, f"c_s40_fd_{nm}_{h}_{s}_jax.csv") for s in ("p", "m"))
                if not (os.path.exists(fp) and os.path.exists(fm)):
                    continue
                p, m = rd(fp), rd(fm)
                ok = all(len(x["P"]) - 1 >= need and np.array_equal(x["time"][1:need + 1], z["time"][:need])
                         for x in (p, m))
                row = dict(window=win, param=nm, h=float(h), valid_replay=ok, increments_in_J=need, J=J0,
                           adjoint=float(gA[j]))
                if ok:
                    pad = np.zeros(len(z["P"]) - need)                       # columns beyond `need` are zero in O
                    fd = (J(np.r_[p["P"][1:need + 1], pad]) - J(np.r_[m["P"][1:need + 1], pad])) / (2 * float(h))
                    row.update(fd=float(fd), rel_err=float(abs(gA[j] - fd) / abs(gA[j])))
                fdr.append(row)
if fdr:
    write(fdr, "s5b_adjoint_vs_fd.csv")

# ---------------------------------------------------------------- p-scaling (piecewise law)
sc = []
r = js("c_s40_pw30_s5b.json")
if r is not None:
    fwd = r["wall_forward_run"] - sum(v["wall"] for v in r["direct"].values()) - r["wall_record"]
    for p, v in sorted(r["direct"].items(), key=lambda kv: int(kv[0])):
        sc.append(dict(method="direct", p=int(p), wall=v["wall"], over_forward=v["wall"] / fwd))
    for win, o in r["objectives"].items():
        sc.append(dict(method=f"adjoint ({win})", p=len(r["names"]), wall=o["wall_adjoint"],
                       over_forward=o["wall_adjoint"] / fwd))
    sc.append(dict(method="forward (net of direct, record)", p=0, wall=fwd, over_forward=1.0))
    write(sc, "s5b_scaling.csv")

# ---------------------------------------------------------------- cost / memory
co = []
for job in ("c_s40", "sw_s40"):
    r = js(f"{job}_cost_s5b.json")
    if r is None:
        continue
    for win, o in r["objectives"].items():
        co.append(dict(job=job, window=win, increments=r["increments"], forward_run=r["wall_forward_run"],
                       record=r["wall_record"], adjoint=o["wall_adjoint"],
                       adjoint_over_forward=o["wall_adjoint"] / r["wall_forward_run"],
                       adj_rhs=o["adjoint_timers"]["rhs"], adj_solve=o["adjoint_timers"]["solve"],
                       adj_back=o["adjoint_timers"]["back"], storage_MB=r["storage_MB"],
                       storage_MB_per_increment=r["storage_MB_per_increment"], rss_before_MB=r["rss_MB_before_run"],
                       rss_after_forward_MB=r["rss_MB_after_forward"], rss_after_adjoint_MB=o["rss_MB_after_adjoint"],
                       jit_adjoint=r["wall_jit_adjoint"]))
if co:
    write(co, "s5b_cost.csv")
print("done")
