"""s3c_bench.py -- S3c micro-benchmarks on one T-peel deck: tangent assembly and linear solvers.

Units: mm, N, s.  Run on a compute node (production deck: ~160k unknowns).
    python scripts/fe/s3c_bench.py <deck.inp> <out.json> [--lam 0.06] [--reps 5]
        [--solvers superlu,superlu_mmd,pardiso,pardiso_sym]

State: Newton-converged solution (newton_energy, condensed element system) of one step from 0 to
lam * T with the deck's law and zero history, then the history updated there (damage active near the
tip).  At that state:
  1. Tangent assembly, each timed over `reps` calls after a warm-up call:
       jacfwd   TPeelModel.tangent (tatva coloured sparse jacfwd on U, then J^T K J + G, then the
                free-free slice), split into jac / reduction / slice;
       element  ElementSystem.system, uncondensed (same free-free matrix) and condensed.
     Check: element (uncondensed) vs jacfwd matrix, max |dK| / max |K|  (gate 1e-12).
  2. Residual: TPeelModel.residual (numpy lift + J), TPeelModel.value_and_grad (jitted q-energy),
     ElementSystem.value_and_grad.
  3. Linear solvers on the uncondensed and the condensed system: first call (incl. symbolic analysis)
     and the mean of `reps` further calls with perturbed values (same pattern), backward error
     max |A x - b| / max |b|.
"""
import argparse
import json
import os
import time

import numpy as np

import tslid.fe  # noqa: F401
from tslid.fe.assembly import ElementSystem
from tslid.fe.incremental import SolverOptions, newton_energy
from tslid.fe.linsolve import SOLVERS
from tslid.fe.tpeel import TPeelModel

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("out")
ap.add_argument("--lam", type=float, default=0.06)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--solvers", default="superlu,superlu_mmd,pardiso,pardiso_sym")
a = ap.parse_args()
res = dict(deck=os.path.abspath(a.deck), lam=a.lam, reps=a.reps, cpus=len(os.sched_getaffinity(0)),
           mkl_threads=os.environ.get("MKL_NUM_THREADS"), xla=os.environ.get("XLA_FLAGS", ""))


def timeit(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.mean(ts)), float(np.min(ts))


t0 = time.perf_counter()
m = TPeelModel(a.deck)
res["setup_model"] = time.perf_counter() - t0
t0 = time.perf_counter()
S = ElementSystem(m)
res["setup_element"] = time.perf_counter() - t0
Sc = ElementSystem(m, condense=True)
res.update(n_q=m.n_q, n_free=int(len(m.free)), n_sys_condensed=Sc.n_sys, colours=m.n_colors, nnz_free=S.nnz,
           nnz_condensed=Sc.nnz)
print({k: res[k] for k in ("n_q", "n_free", "n_sys_condensed", "colours", "nnz_free", "nnz_condensed")}, flush=True)

# ---- state
st = m.deck.step["static"]
dt = a.lam * st["T"]
hist0 = m.zero_history()
h = m.frozen_args(hist0, dt)
q = m.apply_bc(m.zero_q(), a.lam)
t0 = time.perf_counter()
q, info = newton_energy(Sc, q, h, SolverOptions(max_iter=40))
res["state_newton"] = dict(converged=info["converged"], iterations=info["iterations"], wall=time.perf_counter() - t0,
                           P=float(info["r"][m.out["rp1_u2"]]))
hist = Sc.update_history(q, hist0, dt)
res["state_Dmax"] = float(np.max(np.asarray(hist["D"])))
h = m.frozen_args(hist, dt)
print("state", res["state_newton"], "Dmax", res["state_Dmax"], flush=True)

# ---- 1. tangent assembly
m.tangent(q, h)                                                  # compile
m.timers.clear()
res["tangent_jacfwd"] = timeit(lambda: m.tangent(q, h)[m.free][:, m.free], a.reps)
n = a.reps + 1
res["tangent_jacfwd_split"] = {k: m.timers[k] / n for k in ("lift", "jac", "to_csr", "grad", "reduce_K")}
_, r = S.value_and_grad(q, h)
res["tangent_element"] = timeit(lambda: S.system(q, h, r), a.reps)
res["tangent_element_condensed"] = timeit(lambda: Sc.system(q, h, r), a.reps)
K0 = m.tangent(q, h)[m.free][:, m.free]
K1 = S.system(q, h, r)[0]
res["dK_rel"] = float(abs(K1 - K0).max() / abs(K0).max())
print("tangent", res["tangent_jacfwd"], res["tangent_jacfwd_split"], res["tangent_element"],
      res["tangent_element_condensed"], "dK", res["dK_rel"], flush=True)

# ---- 2. residual
res["residual_numpy_lift"] = timeit(lambda: m.residual(q, h), a.reps)
res["residual_jit_q"] = timeit(lambda: m.value_and_grad(q, h), a.reps)
res["residual_element"] = timeit(lambda: S.value_and_grad(q, h), a.reps)
r0 = m.residual(q, h)
res["dr_rel"] = float(np.abs(S.value_and_grad(q, h)[1] - r0).max() / np.abs(r0).max())
print("residual", res["residual_numpy_lift"], res["residual_jit_q"], res["residual_element"], "dr", res["dr_rel"],
      flush=True)

# ---- 3. linear solvers
res["solvers"] = []
for name, A, b in (("free", K1, r[S.sys_dofs]), ("condensed",) + tuple(Sc.system(q, h, r)[:2])):
    for sn in a.solvers.split(","):
        try:
            s = SOLVERS[sn]()
        except Exception as e:                                   # optional dependency missing
            res["solvers"].append(dict(system=name, solver=sn, error=repr(e)))
            continue
        t = time.perf_counter()
        x = s(A, b)
        first = time.perf_counter() - t
        ts = []
        for k in range(a.reps):
            A2 = A.copy()
            A2.data = A2.data * (1.0 + 1e-3 * (k + 1))
            t = time.perf_counter()
            x = s(A2, b)
            ts.append(time.perf_counter() - t)
        row = dict(system=name, solver=sn, n=A.shape[0], nnz=int(A.nnz), first=first, mean=float(np.mean(ts)),
                   min=float(np.min(ts)), backward=float(np.abs(A2 @ x - b).max() / np.abs(b).max()),
                   n_analysis=getattr(s, "n_analysis", None), n_perturbed=getattr(s, "n_perturbed", None))
        res["solvers"].append(row)
        print(row, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
