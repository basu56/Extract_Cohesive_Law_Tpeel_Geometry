"""s3_run_tpeel.py -- run a build_tpeel_inp.py deck with tslid.fe (S3b/S3c driver).

Units: mm, N, MPa, s.  Coarse decks run anywhere; production decks via sbatch.
    python scripts/fe/s3_run_tpeel.py <deck.inp> <outdir> [--history monolithic|staggered]
        (defaults since S3d = D24: --newton energy --assembly element --condense --linsolve pardiso_sym
         --diverge-after 4 --rtol 1e-6; the S3b path is --newton residual --assembly jacfwd
         --linsolve superlu --diverge-after 0 --rtol 1e-10)
        [--kinematics svk|hencky] [--frame global|midsurface] [--rtol 1e-10] [--tag name]
        [--max-increments N]

Time incrementation from the deck's *Static line (dt0, T, dtmin, dtmax), ABAQUS-like growth /
cutback (src/tslid/fe/incremental.py).  Writes
    <outdir>/<job>_<tag>_jax.csv   per converged increment: time, delta (RP1 U2 - RP2 U2), P (RF2 at
                                   RP1, N), UR3, mouth openings, Dmax, newton iterations, passes,
                                   cutbacks before it, line-search halvings, wall/tangent/solve/history s
    <outdir>/<job>_<tag>_jax.json  run summary (totals, timings, options, failure message, profile)
    <outdir>/<job>_<tag>_sens.npz  (--sens, S5a) per converged increment: time, delta, P, and the direct
                                   sensitivities dP, dmouth_u, dmouth_l (n_inc, 4) with respect to
                                   theta = (ln K0, ln sigc, ln Gc, alpha), the kink counts n_switch /
                                   n_fail, the solve consistency res_free and the wall time per step
--perturb name:h runs the law theta_j + h (finite differences, S5a; implies --no-abaqus-out); with
--schedule it replays another run's increments.
Both CSV and JSON are written as the run goes (S3c): one CSV row per converged increment, flushed; the JSON every
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
from tslid.fe.incremental import SolverOptions, newton_energy, newton_ls, run
from tslid.fe.linsolve import SOLVERS, make_solver
from tslid.fe.tpeel import TPeelModel

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("outdir")
ap.add_argument("--history", default="monolithic")
ap.add_argument("--kinematics", default="svk")
ap.add_argument("--frame", default=None)
ap.add_argument("--rtol", type=float, default=1e-6)                 # D24
ap.add_argument("--tag", default=None)
ap.add_argument("--max-increments", type=int, default=10 ** 6)
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--summary-every", type=int, default=50)
ap.add_argument("--newton", default="energy", choices=["residual", "energy"],
                help="residual: S3b newton_ls; energy: S3c newton_energy (jitted q-energy, Armijo)")
ap.add_argument("--assembly", default="element", choices=["jacfwd", "element"])
ap.add_argument("--condense", action=argparse.BooleanOptionalAction, default=True,
                help="condense the incompatible modes (element assembly; D24 default on)")
ap.add_argument("--linsolve", default="pardiso_sym", choices=sorted(SOLVERS),
                help="falls back to superlu_mmd if pypardiso/MKL is unavailable (D24)")
ap.add_argument("--no-abaqus-out", action="store_true",
                help="skip <job>_<tag>_response.csv / _profiles.npz / .json / .sta (tpeel_output.py)")
ap.add_argument("--schedule", default=None, help="CSV (12 digits) or _sens.npz (exact) of a previous run: "
                                                    "reuse its increment end times")
ap.add_argument("--diverge-after", type=int, default=4, help="newton_energy divergence check (0 = off; D24: 4)")
ap.add_argument("--cprofile", action="store_true", help="also write <job>_<tag>_cprofile.txt (top 40 by cumtime)")
ap.add_argument("--sens", action="store_true", help="S5a direct sensitivities (theta = ln K0, ln sigc, ln Gc, alpha)")
ap.add_argument("--perturb", default=None, help="name:h -- run the law with theta_name + h (FD; S5a)")
a = ap.parse_args()
job = os.path.basename(a.deck)[:-4]
tag = a.tag or f"{a.history}_{a.kinematics}_{a.frame or 'auto'}"
os.makedirs(a.outdir, exist_ok=True)

t0 = time.perf_counter()
m = TPeelModel(a.deck, kinematics=a.kinematics, frame=a.frame, history=a.history)
M = m
if a.assembly == "element":
    from tslid.fe.assembly import ElementSystem
    M = ElementSystem(m, condense=a.condense)
elif a.condense:
    a.condense = False                                          # condensation needs element assembly
    if a.newton != "energy":
        raise SystemExit("--assembly element needs --newton energy")
solver = make_solver(a.linsolve)
if a.perturb:
    from tslid.fe.sensitivity import perturb_alpha_log
    if a.assembly != "element":
        raise SystemExit("--perturb needs --assembly element")
    pn, ph = a.perturb.split(":")
    M.set_theta(perturb_alpha_log(m.law, pn, float(ph)))
    a.no_abaqus_out = True                                      # the writer's energies use the deck law
sens = None
if a.sens:
    from tslid.fe.sensitivity import DirectSensitivity, ElementProblem
    if a.assembly != "element" or a.history != "monolithic":
        raise SystemExit("--sens needs --assembly element --history monolithic")
    sens = DirectSensitivity(ElementProblem(M, solver), M.theta)
t_setup = time.perf_counter() - t0
st = m.deck.step["static"]
print(f"{job}: n_q {m.n_q} free {len(m.free)} colours {m.n_colors} law {m.law_kind} {m.law_params} "
      f"mu {m.viscosity} setup {t_setup:.1f} s", flush=True)

# first call (JIT) cost, measured separately
t1 = time.perf_counter()
h = m.frozen_args(m.zero_history(), st["dt0"])
q0 = m.apply_bc(m.zero_q(), 0.0)
if a.newton == "energy":
    E0, r0 = M.value_and_grad(q0, h)
    M.direction(q0, h, r0, solver)
else:
    m.tangent(q0, h, return_residual=True)
    m.residual(q0, h)
t_jit = time.perf_counter() - t1
t_jit_sens = 0.0
if sens is not None:                                            # compile the sensitivity kernels outside the run
    t1 = time.perf_counter()
    hz = m.zero_history()
    sens.step(q0, hz, h, hz, st["dt0"])
    sens.Sh, sens.rows, sens.wall = None, [], 0.0
    sens.prob.timers = dict.fromkeys(sens.prob.timers, 0.0)
    t_jit_sens = time.perf_counter() - t1
for X in {id(m): m, id(M): M}.values():
    X.timers.clear()
    X.counts.clear()


writer = None
if not a.no_abaqus_out:
    from tslid.fe.assembly import ElementSystem
    from tslid.fe.tpeel_output import TPeelWriter
    writer = TPeelWriter(M if a.assembly == "element" else ElementSystem(m), a.deck[:-4] + ".json", a.outdir,
                         f"{job}_{tag}")


def rec(m, t, q, hist, h, r):
    if writer is not None:
        tw = time.perf_counter()
        writer.add(t, q, hist, float(r[m.out["rp1_u2"]]), float(q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]]))
        state["t_io"] += time.perf_counter() - tw
    return dict(time=t, delta=q[m.out["rp1_u2"]] - q[m.out["rp2_u2"]], P=r[m.out["rp1_u2"]],
                UR3=q[m.out["rp1_ur3"]], mouth_u=q[m.out["mouth_u_u2"]], mouth_l=q[m.out["mouth_l_u2"]],
                Dmax=float(np.max(np.asarray(hist["D"]))), n_failed=int(np.sum(np.asarray(hist["D"]) >= 1.0)))


opt = SolverOptions(rtol=a.rtol, linsolve=solver, diverge_after=a.diverge_after)
path = os.path.join(a.outdir, f"{job}_{tag}_jax.csv")
jpath = os.path.join(a.outdir, f"{job}_{tag}_jax.json")
LOGF = ("dt", "newton", "passes", "cutbacks", "ls", "wall", "t_tangent", "t_solve", "t_history", "t_ls",
        "failed_iters", "failed_wall", "diverged")
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
             failed_iters=int(sum(l.failed_iters for l in L)), failed_wall=float(sum(l.failed_wall for l in L)),
             diverged=int(sum(l.diverged for l in L)),
             iterations_all=int(sum(l.iterations + l.failed_iters for l in L)),
             diverge_after=a.diverge_after, schedule=a.schedule,
             wall_run=res.wall if res else time.perf_counter() - state["t_start"], wall_setup=t_setup,
             wall_first_call=t_jit, wall_first_call_sens=t_jit_sens, perturb=a.perturb,
             theta=[float(x) for x in jax_leaves(M.theta)] if a.assembly == "element" and M.theta is not None else None,
             sens=bool(sens), sens_wall=sens.wall if sens else None, sens_timers=sens.prob.timers if sens else None,
             sens_res_free_max=float(max((r["res_free"] for r in sens.rows), default=0.0)) if sens else None,
             sens_n_switch=int(sum(r["n_switch"] for r in sens.rows)) if sens else None,
             sens_n_fail=int(sum(r["n_fail"] for r in sens.rows)) if sens else None,
             sens_finite=bool(all(r["finite"] for r in sens.rows)) if sens else None,
             t_tangent=float(sum(l.t_tangent for l in L)), t_solve=float(sum(l.t_solve for l in L)),
             t_history=float(sum(l.t_history for l in L)), t_ls=float(sum(l.t_ls for l in L)),
             t_increments=float(sum(l.wall for l in L)), t_io=state["t_io"],
             newton=a.newton, assembly=a.assembly, condense=a.condense, linsolve=solver.name,
             linsolve_requested=a.linsolve,
             n_sys=int(getattr(M, "n_sys", len(m.free))),
             timers={k: float(v) for k, v in {**m.timers, **M.timers}.items()},
             counts={**m.counts, **M.counts},
             jit_cache={**m.jit_cache_sizes(), **(M.jit_cache_sizes() if M is not m else {})},
             P_peak=float(max(r["P"] for r in rows)) if rows else None,
             delta_peak=float(max(rows, key=lambda r: r["P"])["delta"]) if rows else None,
             threads=os.environ.get("XLA_FLAGS", ""), omp=os.environ.get("OMP_NUM_THREADS", ""),
             affinity=len(os.sched_getaffinity(0)),
             linsolve_analyses=getattr(solver, "n_analysis", None), linsolve_perturbed=getattr(solver, "n_perturbed", None))
    with open(jpath + ".tmp", "w") as fh:
        json.dump(s, fh, indent=2)
    os.replace(jpath + ".tmp", jpath)
    return s


def jax_leaves(t):
    import jax
    return jax.tree_util.tree_leaves(t)


def write_sens():
    if sens is None or not sens.rows:
        return
    S = sens.as_arrays()
    rows = state["rows"][1: 1 + len(sens.rows)]
    np.savez(os.path.join(a.outdir, f"{job}_{tag}_sens.npz"), names=np.array(sens.names),
             time=np.array([r["time"] for r in rows]), delta=np.array([r["delta"] for r in rows]),
             P=np.array([r["P"] for r in rows]), dP=S["P"], dmouth_u=S["mouth_u"], dmouth_l=S["mouth_l"],
             n_switch=S["n_switch"], n_fail=S["n_fail"], res_free=S["res_free"], finite=S["finite"],
             wall=S["wall"], theta=np.array([float(x) for x in jax_leaves(sens.law)]))


def on_inc(row, l):
    t0 = time.perf_counter()
    row = dict(row)
    if l is None:
        row.update({k: 0 for k in LOGF})
    else:
        row.update(dt=l.dt, newton=l.iterations, passes=l.passes, cutbacks=l.cutbacks, ls=l.ls_steps, wall=l.wall,
                   t_tangent=l.t_tangent, t_solve=l.t_solve, t_history=l.t_history, t_ls=l.t_ls,
                   failed_iters=l.failed_iters, failed_wall=l.failed_wall, diverged=l.diverged)
        state["logs"].append(l)
    state["rows"].append(row)
    if state["w"] is None:
        state["fh"] = open(path, "w", newline="")
        state["w"] = csv.DictWriter(state["fh"], fieldnames=list(row.keys()))
        state["w"].writeheader()
    state["w"].writerow({k: (f"{v:.17g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    state["fh"].flush()
    if l is not None and len(state["logs"]) % a.summary_every == 0:
        summary(False)
        write_sens()
        if writer is not None:
            writer.write_profiles()
    state["t_io"] += time.perf_counter() - t0


if a.cprofile:
    import cProfile
    import io
    import pstats
    prof = cProfile.Profile()
    prof.enable()
res = run(M, st["T"], st["dt0"], st["dtmin"], st["dtmax"], opt, record=rec, verbose=a.verbose,
          max_increments=a.max_increments, on_increment=on_inc,
          newton=newton_energy if a.newton == "energy" else newton_ls,
          schedule=None if a.schedule is None else
          ([float(x) for x in np.load(a.schedule)["time"] if x > 0] if a.schedule.endswith(".npz") else
           [float(r["time"]) for r in csv.DictReader(open(a.schedule)) if float(r["time"]) > 0]),
          on_converged=None if sens is None else sens.step)
if a.cprofile:
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
    open(os.path.join(a.outdir, f"{job}_{tag}_cprofile.txt"), "w").write(buf.getvalue())
state["fh"].close()
write_sens()
if writer is not None:
    writer.close(completed=not res.failed)
print(json.dumps(summary(True, res), indent=1))
