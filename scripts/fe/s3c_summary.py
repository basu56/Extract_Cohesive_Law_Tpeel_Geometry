"""s3c_summary.py -- S3c tables: coarse-run gains, tolerance study, benchmarks.  Light.

Units: mm, N, s.
    python scripts/fe/s3c_summary.py <rundir> [outdir=results/solver/S3_tpeel]

Inputs in <rundir> (scripts/fe/run_s3c.sh, run_s3c_bench.sh; ABAQUS coarse run linked from
solver_S3b_coarse): c_s40_<tag>_jax.csv/.json, c_s40_response.csv, c_s40.msg, bench_*_t<n>.json.
Writes
    s3c_runs.csv      one row per tslid.fe run: options, increments, cutbacks, Newton iterations
                      (converged increments / incl. abandoned attempts), line-search halvings, wall
                      time and its split, P_peak and whole-curve differences vs ABAQUS and vs the
                      reference tslid.fe run (tag "elc2": rtol 1e-10), on a common delta grid
    s3c_bench.csv     micro-benchmarks (tangent, residual, linear solvers) per deck and core count
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
REF = "elc2"


def rd(p):
    return np.genfromtxt(p, delimiter=",", names=True)


ab = rd(os.path.join(RUN, "c_s40_response.csv"))
Ppk = ab["P"].max()
d_end = ab["delta"].max()
grid = np.linspace(0.0, d_end, 4000)
pa = np.interp(grid, ab["delta"], ab["P"])
msg = open(os.path.join(RUN, "c_s40.msg")).read()
num = lambda pat: int(re.findall(pat, msg)[-1])
rows = [dict(tag="abaqus", increments=num(r"TOTAL OF\s+(\d+)\s+INCREMENTS"), cutbacks=num(r"(\d+)\s+CUTBACKS"),
             iterations_all=num(r"(\d+)\s+ITERATIONS INCLUDING"),
             wall=float(re.findall(r"WALLCLOCK TIME \(SEC\) =\s+(\d+)", msg)[-1]), P_peak=Ppk, complete=True)]
ref = None
p = os.path.join(RUN, f"c_s40_{REF}_jax.csv")
if os.path.exists(p):
    d = rd(p)
    ref = np.interp(grid, d["delta"], d["P"])
for js in sorted(glob.glob(os.path.join(RUN, "c_s40_*_jax.json"))):
    s = json.load(open(js))
    d = rd(js[:-5] + ".csv")
    reach = d["delta"].max()
    g = grid <= reach
    P = np.interp(grid[g], d["delta"], d["P"])
    e = (P - pa[g]) / Ppk
    k = int(np.argmax(d["P"]))
    row = dict(tag=s["tag"], newton=s.get("newton", "residual"), assembly=s.get("assembly", "jacfwd"),
               condense=s.get("condense", False), linsolve=s.get("linsolve", "superlu"), rtol=s["rtol"],
               diverge_after=s.get("diverge_after", 0), schedule=os.path.basename(s.get("schedule") or ""),
               complete=s["complete"], t_reached=s.get("t_reached"),
               increments=s["increments"], cutbacks=s["cutbacks"], newton_total=s["newton_total"],
               iterations_all=s.get("iterations_all"), ls_total=s["ls_total"],
               wall=s["wall_run"], t_increments=s.get("t_increments"), failed_wall=s.get("failed_wall"),
               wall_first_call=s.get("wall_first_call"), **{"T_" + k: v for k, v in s.get("timers", {}).items()},
               P_peak=float(d["P"][k]), err_P_peak=float((d["P"][k] - Ppk) / Ppk),
               max_dP_abq=float(np.abs(e).max()), rms_dP_abq=float(np.sqrt(np.mean(e ** 2))))
    if ref is not None:
        er = (P - ref[g]) / Ppk
        row.update(max_dP_ref=float(np.abs(er).max()), rms_dP_ref=float(np.sqrt(np.mean(er ** 2))))
        if s.get("schedule"):                                  # same increments: compare at equal times
            r0 = {round(t, 10): pp for t, pp in zip(*[rd(os.path.join(RUN, os.path.basename(s["schedule"])))[c]
                                                    for c in ("time", "P")])}
            same = [abs(pp - r0[round(t, 10)]) for t, pp in zip(d["time"], d["P"]) if round(t, 10) in r0]
            row["max_dP_ref_same_t"] = float(max(same) / Ppk) if same else np.nan
    rows.append(row)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(OUT, "s3c_runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) and abs(v) > 1e-3 else v) for k, v in r.items()
           if not k.startswith("T_")})

brows = []
for js in sorted(glob.glob(os.path.join(RUN, "bench_*_t*.json"))):
    s = json.load(open(js))
    base = dict(deck=os.path.basename(s["deck"]), cpus=s["cpus"], n_free=s["n_free"], n_cond=s["n_sys_condensed"])
    for k in ("tangent_jacfwd", "tangent_element", "tangent_element_condensed", "residual_numpy_lift",
              "residual_jit_q", "residual_element"):
        brows.append(dict(base, item=k, mean_s=s[k][0], min_s=s[k][1],
                          check=s["dK_rel"] if k.startswith("tangent_element") and "cond" not in k else
                          (s["dr_rel"] if k == "residual_element" else np.nan)))
    for r in s["solvers"]:
        if "error" in r:
            brows.append(dict(base, item=f"solve_{r['system']}_{r['solver']}", note=r["error"][:120]))
        else:
            brows.append(dict(base, item=f"solve_{r['system']}_{r['solver']}", mean_s=r["mean"], min_s=r["min"],
                              first_s=r["first"], check=r["backward"], n=r["n"], nnz=r["nnz"]))
if brows:
    keys = []
    for r in brows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "s3c_bench.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in brows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
print("wrote", OUT)
