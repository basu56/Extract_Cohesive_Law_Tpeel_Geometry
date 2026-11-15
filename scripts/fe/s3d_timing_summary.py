"""s3d_timing_summary.py -- table of the D25 timing protocol job (scripts/fe/run_s3d_timing.sh).  Light.

Units: s, MB.
    python scripts/fe/s3d_timing_summary.py <timing rundir> [outdir=results/solver/S3_tpeel]

One row per run: ABAQUS default deck, ABAQUS with thinned field output, tslid.fe 1 thread, tslid.fe 4
threads.  wall = /usr/bin/time of the whole command (incl. ABAQUS pre/post-processing, Python start-up,
JIT); for ABAQUS also WALLCLOCK of the .msg (the analysis), increments / cutbacks / iterations and the
.odb size; for tslid.fe the run summary (increments, iterations incl. abandoned attempts, setup, first
call, output writing).  Ratios are relative to the ABAQUS default deck, from this one job (D25).
"""
import csv
import glob
import json
import os
import re
import sys

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
log = open(sorted(glob.glob(os.path.join(RUN, "slurm-timing-*.out")))[-1]).read()
wall = {m.group(1): float(m.group(2)) for m in re.finditer(r"TIMING (\S+) wall ([\d.]+) s", log)}
host = re.search(r"host (\S+)", log).group(1)
rows = []
for j in ("tm_abq_default", "tm_abq_thin"):
    msg = open(os.path.join(RUN, j + ".msg")).read()
    g = lambda pat: int(re.findall(pat, msg)[-1])
    rows.append(dict(run=j, host=host, threads=1, wall=wall.get(j),
                     analysis_wall=float(re.findall(r"WALLCLOCK TIME \(SEC\) =\s+(\d+)", msg)[-1]),
                     increments=g(r"TOTAL OF\s+(\d+)\s+INCREMENTS"), cutbacks=g(r"(\d+)\s+CUTBACKS"),
                     iterations_all=g(r"(\d+)\s+ITERATIONS INCLUDING"),
                     completed="COMPLETED SUCCESSFULLY" in open(os.path.join(RUN, j + ".sta")).read(),
                     output_MB=os.path.getsize(os.path.join(RUN, j + ".odb")) / 2 ** 20))
for t in (1, 4):
    s = json.load(open(glob.glob(os.path.join(RUN, f"*_tm_t{t}_jax.json"))[0]))
    outs = glob.glob(os.path.join(RUN, f"*_tm_t{t}_*")) + glob.glob(os.path.join(RUN, f"*_tm_t{t}.*"))
    rows.append(dict(run=f"tslid.fe_t{t}", host=host, threads=s.get("affinity"), wall=wall.get(f"tslid_t{t}"),
                     analysis_wall=s["wall_run"], increments=s["increments"], cutbacks=s["cutbacks"],
                     iterations_all=s["iterations_all"], completed=s["complete"],
                     output_MB=sum(os.path.getsize(f) for f in set(outs)) / 2 ** 20, setup=s["wall_setup"],
                     first_call=s["wall_first_call"], t_io=s["t_io"], rtol=s["rtol"], linsolve=s["linsolve"]))
ref = rows[0]
for r in rows:
    r["ratio_wall"] = r["wall"] / ref["wall"] if r["wall"] and ref["wall"] else None
    r["ratio_analysis"] = r["analysis_wall"] / ref["analysis_wall"]
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(OUT, "s3d_timing.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print(r)
