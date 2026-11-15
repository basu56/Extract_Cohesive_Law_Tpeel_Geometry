"""Incremental (load-stepping) driver for the T-peel: Newton with line search, cutback, regrowth.
Step S3b.

Units: mm, N, MPa, s.

One increment t_n -> t_n+1 = t_n + dt:
  1. history (dmax_n, Dv_n) frozen; prescribed dofs set to their value at t_n+1 (linear ramp of
     the step boundary condition, as ABAQUS *Static with the default amplitude);
  2. initial guess: linear extrapolation of the previous increment on the free dofs (ABAQUS
     default, *Static, extrapolation=LINEAR), or the previous solution;
  3. Newton on the incremental potential (history="monolithic") with a backtracking line search
     on ||r_free||_2 (the Newton direction is a descent direction of ||r||^2 even when the
     softening tangent is indefinite);
  4. converged -> history update (update_damage), record, maybe grow dt; not converged within
     max_iter -> discard, halve dt, retry (cutback); dt < dtmin -> abort and report.
Time incrementation mimics ABAQUS/Standard: start at dt0, cap at dtmax, grow by 1.5 after two
consecutive increments that converge in <= I_G = 4 iterations, cut back on failure, land on T.

history="staggered" (cross-check): repeat solve -> update until the damage stops changing
(change <= stagger_tol), counting the passes.

Convergence of one Newton solve: ||r_free||_inf <= rtol * f_ref, f_ref = max(||r_fixed||_inf
(reactions at the current iterate), f_floor) -- a force scale independent of the size of the
prescribed jump -- or a roundoff-level correction (||dq||_inf <= utol ||q||_inf).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla


@dataclass
class SolverOptions:
    rtol: float = 1e-10
    f_floor: float = 1.0          # N (or N*mm): residual scale floor
    utol: float = 1e-13
    max_iter: int = 16
    line_search: bool = True
    ls_max: int = 8
    ls_c: float = 1e-4
    extrapolate: bool = True
    grow_after: int = 2           # consecutive easy increments before growing dt
    easy_iters: int = 4           # I_G
    grow: float = 1.5
    cut: float = 0.5
    stagger_tol: float = 1e-12
    max_stagger: int = 100
    linsolve: object = None       # callable (A_csc, b) -> x; default SciPy spsolve (SuperLU)
    diverge_after: int = 0        # S3c: >0 -> abandon the attempt (cut back) once ||r_free||_inf has
                                  # increased in two consecutive iterations after this iteration
                                  # (newton_energy only; 0 = off, the S3b behaviour)


@dataclass
class IncrementLog:
    t: float
    dt: float
    iterations: int
    passes: int
    cutbacks: int
    ls_steps: int
    wall: float
    t_tangent: float
    t_solve: float
    t_history: float
    t_ls: float = 0.0             # residual evaluations of rejected/accepted line-search trials
    failed_iters: int = 0         # Newton iterations of the abandoned attempts before this increment
    failed_wall: float = 0.0      # wall time of those attempts (s)
    diverged: int = 0             # how many of them stopped on the divergence check


@dataclass
class RunResult:
    records: list = field(default_factory=list)
    log: list = field(default_factory=list)
    cutbacks: int = 0
    failed: bool = False
    message: str = ""
    wall: float = 0.0


def newton_ls(model, q, h, opt, log=None):
    """Newton with backtracking line search on ||r_free||_2.  Returns (q, info)."""
    free, fixed = model.free, model.fixed
    solve = opt.linsolve or (lambda A, b: spla.spsolve(A, b))
    q = np.array(q, dtype=float)
    res, ok, n_ls = [], False, 0
    t_tan = t_sol = t_ls = 0.0
    t0 = time.perf_counter()
    K, r = model.tangent(q, h, return_residual=True)
    t_tan += time.perf_counter() - t0
    for it in range(opt.max_iter + 1):
        rf = r[free]
        f_ref = max(np.abs(r[fixed]).max() if fixed.size else 0.0, opt.f_floor)
        res.append(float(np.abs(rf).max()))
        if log:
            log("    newton %d |r|=%.3e f_ref=%.3e" % (it, res[-1], f_ref))
        if res[-1] <= opt.rtol * f_ref:
            ok = True
            break
        if it == opt.max_iter or not np.isfinite(res[-1]):
            break
        t0 = time.perf_counter()
        dq = solve(K[free][:, free].tocsc(), rf)
        t_sol += time.perf_counter() - t0
        if not np.all(np.isfinite(dq)):
            break
        n0 = np.linalg.norm(rf)
        step = 1.0
        for k in range(opt.ls_max + 1):
            qt = q.copy()
            qt[free] -= step * dq
            t0 = time.perf_counter()
            if not opt.line_search:
                K, r = model.tangent(qt, h, return_residual=True)
                t_tan += time.perf_counter() - t0
                break
            rt = model.residual(qt, h)
            nt = np.linalg.norm(rt[free])
            t1 = time.perf_counter()
            t_ls += t1 - t0
            if nt <= (1.0 - opt.ls_c * step) * n0 or k == opt.ls_max:
                K, r = model.tangent(qt, h, return_residual=True)
                t_tan += time.perf_counter() - t1
                break
            step *= 0.5
            n_ls += 1
        q = qt
        if step == 1.0 and np.abs(dq).max() <= opt.utol * max(np.abs(q).max(), 1e-300):
            res.append(float(np.abs(r[free]).max()))
            ok = True
            break
    q[model.slave_dofs] = model.lift(q)[model.slave_dofs]
    return q, dict(iterations=it, residuals=res, converged=ok, ls_steps=n_ls, t_tangent=t_tan,
                   t_solve=t_sol, t_ls=t_ls, r=r)


def default_solve(A, b):
    """SciPy SuperLU on the system matrix (CSR or CSC)."""
    return spla.spsolve(A.tocsc(), b)


def newton_energy(model, q, h, opt, log=None):
    """Newton with an energy-based Armijo line search (S3c item 2).  Returns (q, info).

    The model provides value_and_grad(q, h) -> (E, r_q) (one fused jitted call) and
    direction(q, h, r, solve) -> dq with K_ff dq_f = r_f.  Each trial point costs one value_and_grad;
    the full step is tried first; the accepted trial's (E, r) are reused, so an iteration costs one
    tangent + one solve + one (E, r) when the full step is accepted.  A trial is accepted if
        E(q - s dq) <= E(q) - c s (r . dq)       (Armijo on the incremental potential; dq is a
                                                  descent direction when r . dq > 0), or
        ||r_f(q - s dq)||_2 <= (1 - c s) ||r_f(q)||_2   (residual decrease: the Newton direction is
                                                  always a descent direction of ||r||^2, also with
                                                  the indefinite softening tangent and at
                                                  roundoff-level energy differences).
    Convergence as newton_ls.
    """
    free, fixed = model.free, model.fixed
    solve = opt.linsolve or default_solve
    q = np.array(q, dtype=float)
    res, ok, n_ls, diverged = [], False, 0, False
    t_tan = t_sol = t_ls = 0.0
    t0 = time.perf_counter()
    E, r = model.value_and_grad(q, h)
    t_ls += time.perf_counter() - t0
    for it in range(opt.max_iter + 1):
        rf = r[free]
        f_ref = max(np.abs(r[fixed]).max() if fixed.size else 0.0, opt.f_floor)
        res.append(float(np.abs(rf).max()))
        if log:
            log("    newton %d |r|=%.3e f_ref=%.3e E=%.12e" % (it, res[-1], f_ref, E))
        if res[-1] <= opt.rtol * f_ref:
            ok = True
            break
        if it == opt.max_iter or not np.isfinite(res[-1]):
            break
        if opt.diverge_after and it > opt.diverge_after and res[-1] > res[-2] > res[-3]:
            diverged = True
            break
        t0, s0 = time.perf_counter(), model.timers["solve"]
        dq = model.direction(q, h, r, solve)
        ds = model.timers["solve"] - s0
        t_sol += ds
        t_tan += time.perf_counter() - t0 - ds
        if not np.all(np.isfinite(dq)):
            break
        slope = float(r @ dq)
        n0 = np.linalg.norm(rf)
        step = 1.0
        t0 = time.perf_counter()
        for k in range(opt.ls_max + 1):
            qt = q - step * dq
            Et, rt = model.value_and_grad(qt, h)
            if not opt.line_search or k == opt.ls_max:
                break
            if np.isfinite(Et) and np.all(np.isfinite(rt)):
                if (slope > 0 and Et <= E - opt.ls_c * step * slope) or \
                        np.linalg.norm(rt[free]) <= (1.0 - opt.ls_c * step) * n0:
                    break
            step *= 0.5
            n_ls += 1
        t_ls += time.perf_counter() - t0
        q, E, r = qt, Et, rt
        if step == 1.0 and np.abs(dq).max() <= opt.utol * max(np.abs(q).max(), 1e-300):
            res.append(float(np.abs(r[free]).max()))
            ok = True
            break
    q[model.slave_dofs] = model.lift(q)[model.slave_dofs]
    return q, dict(iterations=it, residuals=res, converged=ok, ls_steps=n_ls, t_tangent=t_tan,
                   t_solve=t_sol, t_ls=t_ls, r=r, E=E, diverged=diverged)


def run(model, T, dt0, dtmin, dtmax, opt=None, record=None, verbose=False, max_increments=10 ** 6,
        on_increment=None, newton=None):
    """Integrate the step 0 -> T.  ``record(model, t, q, hist, h_used, r)`` returns a dict stored per
    converged increment (t = 0 included).  ``on_increment(row, inc_log)`` is called right after each
    record (inc_log None for t = 0), so a driver can write results as the run goes (S3c).
    ``newton(model, q, h, opt, log)`` replaces newton_ls (same return contract).  Returns RunResult."""
    newton = newton or newton_ls
    opt = opt or SolverOptions()
    out = RunResult()
    say = print if verbose else None
    t_start = time.perf_counter()
    hist = model.zero_history()
    q = model.apply_bc(model.zero_q(), 0.0)
    h0 = model.frozen_args(hist, dt0)
    if record:
        out.records.append(record(model, 0.0, q, hist, h0, model.residual(q, h0)))
        if on_increment:
            on_increment(out.records[-1], None)
    t, dt, q_prev, dt_prev, easy, n_inc = 0.0, dt0, None, None, 0, 0
    cut_here, f_its, f_wall, f_div = 0, 0, 0.0, 0
    while t < T * (1 - 1e-12) and n_inc < max_increments:
        dt = min(dt, T - t)
        if T - t - dt < 1e-9 * T:
            dt = T - t
        tw = time.perf_counter()
        qt = q.copy()
        if opt.extrapolate and q_prev is not None:
            qt[model.free] += (dt / dt_prev) * (q - q_prev)[model.free]
        qt = model.apply_bc(qt, (t + dt) / T)
        passes, its, n_ls, t_tan, t_sol, t_ls = 0, 0, 0, 0.0, 0.0, 0.0
        if model.history == "monolithic":
            h = model.frozen_args(hist, dt)
            qn, info = newton(model, qt, h, opt, log=say)
            ok, passes = info["converged"], 1
            its, n_ls, t_tan, t_sol = info["iterations"], info["ls_steps"], info["t_tangent"], info["t_solve"]
            t_ls = info.get("t_ls", 0.0)
            t0 = time.perf_counter()
            new = model.update_history(qn, hist, dt) if ok else None
            t_hist = time.perf_counter() - t0
        else:
            hs, ok, t_hist, qn = hist, True, 0.0, qt
            for passes in range(1, opt.max_stagger + 1):
                trial = model.update_history(qn, hist, dt)
                h = model.frozen_args(trial, dt)
                qn, info = newton(model, qn, h, opt, log=say)
                its += info["iterations"]
                t_ls += info.get("t_ls", 0.0)
                n_ls += info["ls_steps"]
                t_tan += info["t_tangent"]
                t_sol += info["t_solve"]
                if not info["converged"]:
                    ok = False
                    break
                t0 = time.perf_counter()
                new = model.update_history(qn, hist, dt)
                t_hist += time.perf_counter() - t0
                change = float(np.max(np.abs(np.asarray(new["Dv"]) - np.asarray(trial["Dv"]))))
                if change <= opt.stagger_tol:
                    break
            else:
                ok = False
        if not ok:
            out.cutbacks += 1
            cut_here += 1
            f_its += its
            f_wall += time.perf_counter() - tw
            f_div += int(info.get("diverged", False))
            if say:
                say("  cutback at t=%.6g: dt %.3g -> %.3g" % (t, dt, dt * opt.cut))
            dt *= opt.cut
            easy = 0
            if dt < dtmin:
                out.failed, out.message = True, "dt %.3g < dtmin %.3g at t = %.6g" % (dt, dtmin, t)
                break
            continue
        q_prev, dt_prev, q, hist = q, dt, qn, new
        t += dt
        n_inc += 1
        out.log.append(IncrementLog(t, dt, its, passes, cut_here, n_ls, time.perf_counter() - tw,
                                    t_tan, t_sol, t_hist, t_ls, f_its, f_wall, f_div))
        cut_here, f_its, f_wall, f_div = 0, 0, 0.0, 0
        if record:
            out.records.append(record(model, t, q, hist, h, info["r"]))
            if on_increment:
                on_increment(out.records[-1], out.log[-1])
        if verbose:
            print("inc %d t=%.6g dt=%.4g its=%d passes=%d ls=%d" % (n_inc, t, dt, its, passes, n_ls), flush=True)
        easy = easy + 1 if its <= opt.easy_iters else 0
        if easy >= opt.grow_after:
            dt = min(dt * opt.grow, dtmax)
            easy = 0
        else:
            dt = min(dt, dtmax)
    out.wall = time.perf_counter() - t_start
    if not out.failed and t < T * (1 - 1e-12):
        out.failed, out.message = True, "stopped at t = %.6g (max_increments)" % t
    return out
