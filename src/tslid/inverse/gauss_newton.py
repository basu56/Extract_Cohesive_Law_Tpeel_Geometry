"""Levenberg-Marquardt for the parametric inversion (I1a, D31).

Units: theta = (ln sigc [ln MPa], ln Gc [ln N/mm], alpha [-]); loads in N.

Misfit  J(theta) = 1/2 sum_b (y_b(theta) - y_b^data)^2 / s^2  over the bins of one window (D29/D31);
residual r = (y - y_data)/s, Jacobian A = Sbar/s from the S5a direct sensitivities (one forward run + p
sensitivity right-hand sides per evaluation).

Step:  (A^T A + mu D) dtheta = -A^T r,  D = diag(A^T A) (Marquardt scaling, so mu is dimensionless and
the step is invariant to the scaling of each parameter), followed by projection onto the feasible set:

  * box bounds (``Bounds``), by clipping;
  * D10 admissibility (E2: phi_e = sigc^2 / (2 K0 Gc) <= 1/2, i.e. ln sigc <= (ln K0 + ln Gc)/2; E3
    then holds), by moving ln sigc down onto the constraint (and ln Gc up if that would leave the box).

Bounds are handled by an active set plus projection: a parameter on a box bound whose descent direction
points outward is held fixed and the damped system is solved for the free ones (otherwise the clipped
step can have a non-positive predicted reduction and good steps get rejected, seen in the I1a smoke run);
a clipped step whose prediction is still <= 0 is accepted if it lowers J.  Projection, not a smooth
log/logit map: the bounds are rarely active near the solution, and a
projected step keeps theta in physical coordinates, so the Jacobian is the S5a one without a chain
factor.  The gain ratio uses the projected step: rho = (J - J_trial) / pred, pred = -(dtheta_p . g)
- 1/2 dtheta_p^T A^T A dtheta_p.  Damping update (Nielsen 1999): accepted (rho > 1e-4) -> mu *= max(1/3,
1 - (2 rho - 1)^3), nu = 2; rejected -> mu *= nu, nu *= 2.  A failed forward run (dt < dtmin) is a
rejected step.
Stopping (D31): relative parameter step max_i |dtheta_i| / max(1, |theta_i|) < xtol = 1e-6 (log
parameters: the relative change of sigc and Gc), or the misfit decrease below ftol = 1e-8 J over two
consecutive accepted iterations, or max_iter = 30 evaluations; also mu > 1e12 (no descent possible).
"""
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bounds:
    lo: np.ndarray
    hi: np.ndarray
    names: tuple = ("ln_sigc", "ln_Gc", "alpha")
    lnK0: float = float(np.log(8500.0))

    def project(self, theta):
        th = np.clip(np.asarray(theta, float), self.lo, self.hi)
        if "ln_sigc" in self.names and "ln_Gc" in self.names:
            i, j = self.names.index("ln_sigc"), self.names.index("ln_Gc")
            cap = 0.5 * (self.lnK0 + th[j]) - 1e-9                       # E2: sigc <= sqrt(K0 Gc)
            if th[i] > cap:
                th[i] = cap
                if th[i] < self.lo[i]:                                   # make room by raising Gc
                    th[i] = self.lo[i]
                    th[j] = min(self.hi[j], 2 * (self.lo[i] + 1e-9) - self.lnK0)
        return th


def default_bounds(lnK0=float(np.log(8500.0))):
    """sigc in [10, 60] MPa (D12 lower bound at Gc = 0.1 is 10.5), Gc in [0.1, 0.8] N/mm (D9's [0.1, 0.4]
    extended upward so the reference Gc = 0.4 is interior), alpha in [-5, 10] (D31)."""
    return Bounds(lo=np.array([np.log(10.0), np.log(0.1), -5.0]), hi=np.array([np.log(60.0), np.log(0.8), 10.0]),
                  lnK0=lnK0)


@dataclass
class LMResult:
    theta: np.ndarray
    J: float
    iterations: int
    converged: bool
    reason: str
    log: list = field(default_factory=list)


def levenberg_marquardt(evaluate, y_data, s, theta0, bounds, mu0=1e-3, max_iter=30, xtol=1e-6, ftol=1e-8,
                        callback=None, max_step=None):
    """Minimise J.  evaluate(theta) -> (y, Sbar, info) or (None, None, info) on a failed forward run.
    callback(entry) is called after every evaluation (for logging as the run goes).
    max_step: per-parameter cap on |dtheta| before projection; the whole step is scaled by one factor,
    so its direction is kept (a trust-region safeguard added after the second I1a smoke run, where an
    undamped alpha step crossed the whole box and the clipped trial was re-evaluated unchanged).  A trial
    identical to an already rejected one is not evaluated again: the damping is raised instead."""
    y_data = np.asarray(y_data, float)
    theta = bounds.project(theta0)
    log = []

    def ev(th):
        t0 = time.perf_counter()
        y, Sb, info = evaluate(th)
        if y is None:
            return None, None, None, info, time.perf_counter() - t0
        r = (y - y_data) / s
        return r, Sb / s, 0.5 * float(r @ r), info, time.perf_counter() - t0

    r, A, J, info, w = ev(theta)
    if r is None:
        raise RuntimeError("forward run failed at the start point: %s" % info)
    entry = dict(k=0, theta=theta.tolist(), J=J, mu=mu0, accepted=True, wall=w, **_info(info))
    log.append(entry)
    callback and callback(entry)
    mu, nu, n_eval, small = mu0, 2.0, 1, 0
    cap = None if max_step is None else np.asarray(max_step, float)
    rejected = []
    reason, converged = "max_iter", False
    while n_eval < max_iter:
        g = A.T @ r
        H = A.T @ A
        D = np.diag(np.maximum(np.diag(H), 1e-300 * max(np.diag(H).max(), 1.0)))
        # active set: a parameter on a box bound whose descent direction points outward stays fixed
        eps = 1e-12
        act = ((theta <= bounds.lo + eps) & (g > 0)) | ((theta >= bounds.hi - eps) & (g < 0))
        fr = ~act
        step = np.zeros_like(theta)
        if fr.any():
            Hf, gf, Df = H[np.ix_(fr, fr)], g[fr], D[np.ix_(fr, fr)]
            try:
                step[fr] = np.linalg.solve(Hf + mu * Df, -gf)
            except np.linalg.LinAlgError:
                step[fr] = -gf / (mu * np.diag(Df))
        if cap is not None:
            f = float(np.max(np.abs(step) / cap))
            if f > 1.0:
                step = step / f
        trial = bounds.project(theta + step)
        if any(np.array_equal(trial, t) for t in rejected):   # same clipped point: damp more, no new run
            mu *= nu
            nu *= 2.0
            if mu > 1e12:
                reason = "mu > 1e12"
                break
            continue
        dp = trial - theta
        pred = -(dp @ g) - 0.5 * dp @ H @ dp
        rel = float(np.max(np.abs(dp) / np.maximum(1.0, np.abs(theta))))
        if rel < xtol and nu == 2.0:                              # only right after an accepted step
            reason, converged = "xtol (relative step %.2e)" % rel, True
            break
        rt, At, Jt, info, w = ev(trial)
        n_eval += 1
        if rt is None:
            rho = -np.inf
        elif pred > 0:
            rho = (J - Jt) / pred
        else:                     # clipped step with a non-positive model prediction: accept any decrease
            rho = 0.5 if Jt < J else -np.inf
        acc = rho > 1e-4
        entry = dict(k=n_eval - 1, theta=trial.tolist(), active=act.tolist(), J=Jt if rt is not None else None, mu=mu, rho=float(rho),
                     pred=float(pred), accepted=bool(acc), rel_step=rel, wall=w, **_info(info))
        log.append(entry)
        callback and callback(entry)
        if acc:
            rejected = []
            dJ = J - Jt
            theta, r, A, J = trial, rt, At, Jt
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            small = small + 1 if dJ < ftol * max(J, 1e-300) else 0
            if small >= 2:
                reason, converged = "ftol (dJ < %.0e J twice)" % ftol, True
                break
        else:
            rejected.append(trial.copy())
            mu *= nu
            nu *= 2.0
            if mu > 1e12:
                reason = "mu > 1e12"
                break
    return LMResult(theta=theta, J=J, iterations=n_eval, converged=converged, reason=reason, log=log)


def _info(info):
    keys = ("ok", "increments", "cutbacks", "wall_sens", "message")
    return {k: info.get(k) for k in keys} if isinstance(info, dict) else {}


def latin_hypercube(n, bounds, seed):
    """n points, one per stratum in every dimension of the box (before the admissibility projection)."""
    rng = np.random.default_rng(seed)
    d = len(bounds.lo)
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return bounds.lo + u * (bounds.hi - bounds.lo)
