"""s3_run_tpeel.py -- run a build_tpeel_inp.py deck with tslid.fe (S3b/S3c driver).

Units: mm, N, MPa, s.  Coarse decks run anywhere; production decks via sbatch.
    python scripts/fe/s3_run_tpeel.py <deck.inp> <outdir> [--history monolithic|staggered]
        [--kinematics svk|hencky] [--frame global|midsurface] [--rtol 1e-10] [--tag name]
        [--max-increments N]

Time incrementation from the deck's *Static line (dt0, T, dtmin, dtmax), ABAQUS-like growth /
cutback (src/tslid/fe/incremental.py).  Writes
    <outdir>/<job>_<tag>_jax.csv   per converged increment: time, delta (RP1 U2 - RP2 U2), P (RF2 at
                                   RP1, N), UR3, mouth openings, Dmax, newton iterations, passes,
                                   cutbacks before it, line-search halvings, wall/tangent/solve/history s
    <outdir>/<job>_<tag>_jax.json  run summary (totals, timings, options, failure message, profile)
Both are written as the run goes (S3c): one CSV row per converged increment, flushed; the JSON every
50 increments with "complete": false, and at the end with "complete": true.  A run killed by the
SLURM time limit therefore leaves its data.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import tslid.fe  # noqa: F401
from tslid.fe.incremental import SolverOptions, run
from tslid.fe.tpeel import TPeelModel

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("outdir")
ap.add_argument("--history", default="monolithic")
ap.add_argument("--kinematics", default="svk")
ap.add_argument("--frame", default=None)
ap.add_argument("--rtol", type=float, default=1e-10)
ap.add_argument("--tag", default=None)
ap.add_argument("--max-increments", type=int, default=10 ** 6)
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--summary-every", type=int, default=50)
ap.add_argument("--cprofile", action="store_true", help="also write <job>_<tag>_cprofile.txt (top 40 by cumtime)")
a = ap.parse_args()
job = os.path.basename(a.deck)[:-4]
tag = a.tag or f"{a.history}_{a.kinematics}_{a.frame or 'auto'}"
os.makedirs(a.outdir, exist_ok=True)

t0 = time.perf_counter()
m = TPeelModel(a.deck, kinematics=a.kinematics, frame=a.frame, history=a.history)
t_setup = time.perf_counter() - t0
st = m.deck.step["static"]
print(f"{job}: n_q {m.n_q} free {len(m.free)} colours {m.n_colors} law {m.law_kind} {m.law_params} "
      f"mu {m.viscosity} setup {t_setup:.1f} s", flush=True)

# first call (JIT) cost, measured separately
t1 = time.perf_counter()
h = m.frozen_args(m.zero_history(), st["dt0"])
q0 = m.apply_bc(m.zero_q(), 0.0)
m.tangent(q0, h, return_residual=True)
m.residual(q0, h)
t_jit = time.perf_counter() - t1
m.timers.clear()
m.counts.clear()


def rec(m, t, q, hist, h, r):
    return dict(time=t, delta=q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]], P=r[m.out["rp1_u2"]],
                UR3=q[m.out["rp1_ur3"]], mouth_u=q[m.out["mouth_u_u2"]], mouth_l=q[m.out["mouth_l_u2"]],
                Dmax=float(np.max(np.asarray(hist["D"]))), n_failed=int(np.sum(np.asarray(hist["D"]) >= 1.0)))


opt = SolverOptions(rtol=a.rtol)
path = os.path.join(a.outdir, f"{job}_{tag}_jax.csv")
jpath = os.path.join(a.outdir, f"{job}_{tag}_jax.json")
LOGF = ("dt", "newton", "passes", "cutbacks", "ls", "wall", "t_tangent", "t_solve", "t_history", "t_ls")
state = dict(fh=None, w=None, rows=[], logs=[], t_io=0.0, t_start=time.perf_counter())


def summary(complete, res=None):
    L, rows = state["logs"], state["rows"]
    s = dict(job=job, tag=tag, deck=os.path.abspath(a.deck), n_q=m.n_q, n_free=int(len(m.free)), colours=m.n_colors,
             history=a.history, kinematics=a.kinematics, frame=m.frame, coupling=m.coupling, rtol=a.rtol,
             complete=complete, t_reached=float(rows[-1]["time"]) if rows else 0.0,
             increments=len(L), cutbacks=int(sum(l.cutbacks for l in L)),
             failed=bool(res.failed) if res else None, message=res.message if res else "",
             newton_total=int(sum(l.iterations for l in L)), newton_max=int(max((l.iterations for l in L), default=0)),
             passes_total=int(sum(l.passes for l in L)), passes_max=int(max((l.passes for l in L), default=0)),
             ls_total=int(sum(l.ls_steps for l in L)),
             wall_run=res.wall if res else time.perf_counter() - state["t_start"], wall_setup=t_setup,
             wall_first_call=t_jit,
             t_tangent=float(sum(l.t_tangent for l in L)), t_solve=float(sum(l.t_solve for l in L)),
             t_history=float(sum(l.t_history for l in L)), t_ls=float(sum(l.t_ls for l in L)),
             t_increments=float(sum(l.wall for l in L)), t_io=state["t_io"],
             timers={k: float(v) for k, v in m.timers.items()}, counts=dict(m.counts),
             jit_cache=m.jit_cache_sizes(),
             P_peak=float(max(r["P"] for r in rows)) if rows else None,
             delta_peak=float(max(rows, key=lambda r: r["P"])["delta"]) if rows else None,
             threads=os.environ.get("XLA_FLAGS", ""), omp=os.environ.get("OMP_NUM_THREADS", ""))
    with open(jpath + ".tmp", "w") as fh:
        json.dump(s, fh, indent=2)
    os.replace(jpath + ".tmp", jpath)
    return s


def on_inc(row, l):
    t0 = time.perf_counter()
    row = dict(row)
    if l is None:
        row.update({k: 0 for k in LOGF})
    else:
        row.update(dt=l.dt, newton=l.iterations, passes=l.passes, cutbacks=l.cutbacks, ls=l.ls_steps, wall=l.wall,
                   t_tangent=l.t_tangent, t_solve=l.t_solve, t_history=l.t_history, t_ls=l.t_ls)
        state["logs"].append(l)
    state["rows"].append(row)
    if state["w"] is None:
        state["fh"] = open(path, "w", newline="")
        state["w"] = csv.DictWriter(state["fh"], fieldnames=list(row.keys()))
        state["w"].writeheader()
    state["w"].writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in row.items()})
    state["fh"].flush()
    if l is not None and len(state["logs"]) % a.summary_every == 0:
        summary(False)
    state["t_io"] += time.perf_counter() - t0


if a.cprofile:
    import cProfile
    import io
    import pstats
    prof = cProfile.Profile()
    prof.enable()
res = run(m, st["T"], st["dt0"], st["dtmin"], st["dtmax"], opt, record=rec, verbose=a.verbose,
          max_increments=a.max_increments, on_increment=on_inc)
if a.cprofile:
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
    open(os.path.join(a.outdir, f"{job}_{tag}_cprofile.txt"), "w").write(buf.getvalue())
state["fh"].close()
print(json.dumps(summary(True, res), indent=1))
