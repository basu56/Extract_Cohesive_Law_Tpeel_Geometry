"""s5b_run.py -- S5b: forward run with stored states, discrete adjoint of J, direct method for comparison.

Units: mm, N, s.  Heavy (sbatch: scripts/fe/run_s5b.sh).
    python scripts/fe/s5b_run.py <deck.inp> <outdir> <tag> --ref <abaqus rundir>/<job>
        [--law alpha | piecewise:<m>] [--rtol 1e-6] [--direct none|4|4,8,16,31] [--linsolve pardiso_sym]

Objective (D29): J = 1/2 w sum_k (y_k - y_k^data)^2, y = bin averages (observation.py, w = DELTA_BIN)
of P (model) and of the ABAQUS record of the same case (data), on two windows: "prepeak" (bins up to
delta_peak of the reference) and "full" (bins up to d_end, compare_ld window).  dJ/dP_n = g_n = O^T w r.
For each J: the adjoint gradient (adjoint.py) and, if direct sensitivities were run, g^T S (same converged
states, same run).  Parameters: alpha law theta = (ln K0, ln sigc, ln Gc, alpha) (alpha_log_basis);
piecewise:<m> = PiecewiseLawParams.from_alpha(deck law, m), all leaves (identity basis, p = m + 1).
--direct lists the p of the direct runs (subsets of the basis, first p directions), each its own
DirectSensitivity on the same forward run, timed separately.
Writes <outdir>/<job>_<tag>_s5b.json (J, gradients, errors, timings, storage, peak RSS) and
<outdir>/<job>_<tag>_s5b.npz (time, delta, P, direct S of the largest p).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

import tslid.fe  # noqa: F401
from tslid.fe.adjoint import Adjoint, StateRecorder
from tslid.fe.assembly import ElementSystem
from tslid.fe.incremental import SolverOptions, newton_energy, run
from tslid.fe.interface import PiecewiseLawParams
from tslid.fe.linsolve import SOLVERS, make_solver
from tslid.fe.observation import DELTA_BIN, observation_matrix, observe, windows
from tslid.fe.sensitivity import (ALPHA_LOG, DirectSensitivity, ElementProblem, alpha_log_basis, identity_basis,
                                  subset_basis)
from tslid.fe.tpeel import TPeelModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("outdir")
ap.add_argument("tag")
ap.add_argument("--ref", required=True, help="<ABAQUS rundir>/<job> of the same case (data)")
ap.add_argument("--law", default="alpha")
ap.add_argument("--rtol", type=float, default=1e-6)
ap.add_argument("--direct", default="4")
ap.add_argument("--linsolve", default="pardiso_sym", choices=sorted(SOLVERS))
ap.add_argument("--width", type=float, default=DELTA_BIN)
a = ap.parse_args()
job = os.path.basename(a.deck)[:-4]
os.makedirs(a.outdir, exist_ok=True)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # MB (Linux: KB)

m = TPeelModel(a.deck)
S = ElementSystem(m, condense=True)
if a.law.startswith("piecewise"):
    S.set_theta(PiecewiseLawParams.from_alpha(m.law, int(a.law.split(":")[1])))
    basis, names = identity_basis(S.theta)
else:
    basis, names = alpha_log_basis(S.theta), list(ALPHA_LOG)
solver = make_solver(a.linsolve)
st = m.deck.step["static"]
prob = ElementProblem(S, solver)
ps = [] if a.direct == "none" else [int(x) for x in a.direct.split(",")]
directs = [DirectSensitivity(prob, S.theta, basis=subset_basis(basis, np.arange(p)), names=names[:p]) for p in ps]
rec = StateRecorder()

# compile outside the timed run (as s3_run_tpeel.py): one Newton direction + each kernel once
t0 = time.perf_counter()
hz = m.zero_history()
h0 = m.frozen_args(hz, st["dt0"])
q0 = m.apply_bc(m.zero_q(), 0.0)
E0, r0 = S.value_and_grad(q0, h0)
S.direction(q0, h0, r0, solver)
for ds in directs:
    ds.step(q0, hz, h0, hz, st["dt0"])
    ds.Sh, ds.rows, ds.wall = None, [], 0.0
t_jit = time.perf_counter() - t0
rss_before = rss()

walls = dict(record=0.0)


def hook(q, ho, h, hn, dt):
    for ds in directs:
        ds.step(q, ho, h, hn, dt)
    t1 = time.perf_counter()
    rec(q, ho, h, hn, dt)
    walls["record"] += time.perf_counter() - t1


out = run(S, st["T"], st["dt0"], st["dtmin"], st["dtmax"],
          SolverOptions(rtol=a.rtol, linsolve=solver, diverge_after=4),
          record=lambda mm, t, q, hist, h, r: dict(time=t, delta=float(q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]]),
                                                   P=float(r[m.out["rp1_u2"]])),
          newton=newton_energy, on_converged=hook)
t_run = out.wall
rss_fwd = rss()
rows = out.records[1:]
tt = np.array([r["time"] for r in rows])
d = np.array([r["delta"] for r in rows])
P = np.array([r["P"] for r in rows])

# objectives
rd, rj = os.path.split(a.ref)
ref = load(rd, rj)
d_peak_ref = float(ref["d"][int(np.argmax(ref["P"]))])
W = windows(ref["d_end"], d_peak_ref, a.width)
res = dict(job=job, tag=a.tag, deck=os.path.abspath(a.deck), law=a.law, names=names, rtol=a.rtol,
           linsolve=solver.name, increments=len(rows), cutbacks=out.cutbacks, failed=out.failed,
           wall_forward_run=t_run, wall_jit=t_jit, wall_record=walls["record"],
           direct={str(p): dict(wall=ds.wall, timers=None) for p, ds in zip(ps, directs)},
           storage_MB=rec.nbytes / 2 ** 20, storage_MB_per_increment=rec.nbytes / 2 ** 20 / max(len(rows), 1),
           rss_MB_before_run=rss_before, rss_MB_after_forward=rss_fwd, width=a.width, d_end=ref["d_end"],
           delta_peak_ref=d_peak_ref, objectives={})
adj = Adjoint(prob, S.theta, m.out["rp1_u2"])
t1 = time.perf_counter()
adj.gradient(rec.states[:1], np.zeros(1), basis)                    # compile the adjoint kernels
res["wall_jit_adjoint"] = time.perf_counter() - t1
for name, edges in W.items():
    O = observation_matrix(np.r_[0.0, d], edges)[:, 1:]
    yd = observe(np.r_[0.0, ref["d"]] if ref["d"][0] > 0 else ref["d"],
                 np.r_[0.0, ref["P"]] if ref["d"][0] > 0 else ref["P"], edges)
    w = a.width
    r = O @ P - yd
    J = 0.5 * w * float(r @ r)
    g = O.T @ (w * r)
    for k in adj.timers:
        adj.timers[k] = 0.0
    t1 = time.perf_counter()
    gA = adj.gradient(rec.states, g, basis)
    t_adj = time.perf_counter() - t1
    o = dict(bins=len(edges) - 1, J=J, adjoint=gA.tolist(), wall_adjoint=t_adj, adjoint_timers=dict(adj.timers),
             rss_MB_after_adjoint=rss())
    for p, ds in zip(ps, directs):
        gD = g @ ds.as_arrays()["P"]
        o[f"direct_p{p}"] = gD.tolist()
        o[f"rel_err_p{p}"] = float(np.abs(gA[:p] - gD).max() / max(np.abs(gD).max(), 1e-300))
    res["objectives"][name] = o
    print(name, json.dumps({k: v for k, v in o.items() if k != "adjoint_timers"}), flush=True)
for p, ds in zip(ps, directs):
    res["direct"][str(p)]["timers"] = None
res["timers_problem"] = prob.timers
json.dump(res, open(os.path.join(a.outdir, f"{job}_{a.tag}_s5b.json"), "w"), indent=1)
np.savez(os.path.join(a.outdir, f"{job}_{a.tag}_s5b.npz"), time=tt, delta=d, P=P,
         S=directs[-1].as_arrays()["P"] if directs else np.zeros((len(P), 0)))
print(json.dumps({k: v for k, v in res.items() if k != "objectives"}, indent=1))
