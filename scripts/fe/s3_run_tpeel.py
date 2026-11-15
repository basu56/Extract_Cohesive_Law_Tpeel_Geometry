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
    <outdir>/<job>_<tag>_jax.json  run summary (totals, timings, options, failure message)
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
ap.add_argument("--frame", default="global")
ap.add_argument("--rtol", type=float, default=1e-10)
ap.add_argument("--tag", default=None)
ap.add_argument("--max-increments", type=int, default=10 ** 6)
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
job = os.path.basename(a.deck)[:-4]
tag = a.tag or f"{a.history}_{a.kinematics}_{a.frame}"
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


def rec(m, t, q, hist, h, r):
    return dict(time=t, delta=q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]], P=r[m.out["rp1_u2"]],
                UR3=q[m.out["rp1_ur3"]], mouth_u=q[m.out["mouth_u_u2"]], mouth_l=q[m.out["mouth_l_u2"]],
                Dmax=float(np.max(np.asarray(hist["D"]))), n_failed=int(np.sum(np.asarray(hist["D"]) >= 1.0)))


opt = SolverOptions(rtol=a.rtol)
res = run(m, st["T"], st["dt0"], st["dtmin"], st["dtmax"], opt, record=rec, verbose=a.verbose,
          max_increments=a.max_increments)
rows = res.records
for k, l in enumerate(res.log, start=1):
    rows[k].update(dt=l.dt, newton=l.iterations, passes=l.passes, cutbacks=l.cutbacks, ls=l.ls_steps,
                   wall=l.wall, t_tangent=l.t_tangent, t_solve=l.t_solve, t_history=l.t_history)
rows[0].update(dt=0.0, newton=0, passes=0, cutbacks=0, ls=0, wall=0.0, t_tangent=0.0, t_solve=0.0, t_history=0.0)
path = os.path.join(a.outdir, f"{job}_{tag}_jax.csv")
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})
L = res.log
summ = dict(job=job, tag=tag, deck=os.path.abspath(a.deck), n_q=m.n_q, n_free=int(len(m.free)), colours=m.n_colors,
            history=a.history, kinematics=a.kinematics, frame=a.frame, rtol=a.rtol,
            increments=len(L), cutbacks=res.cutbacks, failed=res.failed, message=res.message,
            newton_total=int(sum(l.iterations for l in L)), newton_max=int(max(l.iterations for l in L)),
            passes_total=int(sum(l.passes for l in L)), passes_max=int(max(l.passes for l in L)),
            ls_total=int(sum(l.ls_steps for l in L)),
            wall_run=res.wall, wall_setup=t_setup, wall_first_call=t_jit,
            t_tangent=float(sum(l.t_tangent for l in L)), t_solve=float(sum(l.t_solve for l in L)),
            t_history=float(sum(l.t_history for l in L)),
            P_peak=float(max(r["P"] for r in rows)),
            delta_peak=float(max(rows, key=lambda r: r["P"])["delta"]),
            threads=os.environ.get("XLA_FLAGS", ""), omp=os.environ.get("OMP_NUM_THREADS", ""))
json.dump(summ, open(os.path.join(a.outdir, f"{job}_{tag}_jax.json"), "w"), indent=2)
print(json.dumps(summ, indent=1))
