"""Discrete adjoint of the incremental T-peel response.  Step S5b.

Units: mm, N, MPa, s.  Derivation: docs/solver/ADJOINT.md section 3, equations (A1)-(A3).

The forward run stores, per converged increment n, (q_n, dmax_{n-1}, Dv_{n-1}, a_n, b_n, dt_n)
(``StateRecorder``, used as incremental.run(on_converged=...)).  For an objective whose derivative with
respect to the recorded outputs is g_n = dJ/dP_n (P_n = RF2 at RP1, the component ``out_index`` of the
gradient of the incremental potential at a prescribed dof), the backward sweep n = N .. 1 with eta_N = 0:

    K_n mu_n   = grad_q [ g_n Y_n + eta_n . H_n ]_f                         (A1)  (K symmetric)
    eta_{n-1}  = grad_h Phi_n,   dJ/dtheta += grad_theta Phi_n               (A2), (A3)
    Phi_n(q, h, theta) = g_n Y_n(q, h, theta) - mu_n . R_n(q, h, theta) + eta_n . H_n(q, h, theta)

with Y_n = grad_q Pi_n[out_index], R_n = grad_q Pi_n (mu_n is zero on the fixed dofs, so only the free
residual enters) and H_n = (dmax_n, Dv_n) = update_damage.  Every partial is one reverse-mode pass
(jax.grad of a scalar) at the stored state; K_n is re-assembled from the stored state and solved with the
problem's condensed solve (sensitivity.ElementProblem / DenseProblem).  The Newton loop is never
differentiated.  Cost: one Hessian + factorization + one solve + two gradient passes per increment,
independent of the number of parameters.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np


class StateRecorder:
    """on_converged hook storing the per-increment state needed by the backward sweep (in memory)."""

    def __init__(self):
        self.states = []

    def __call__(self, q, hist_old, h, hist_new, dt):
        self.states.append((np.array(q, dtype=float), np.array(hist_old["dmax"]), np.array(hist_old["Dv"]),
                            float(h[2]), float(h[3]), float(dt)))

    @property
    def nbytes(self):
        return int(sum(s[0].nbytes + s[1].nbytes + s[2].nbytes + 24 for s in self.states))


def contract(grad_law, basis):
    """(p,) directional derivatives: sum over the leaves of grad_leaf * basis_leaf[j]."""
    gl = jax.tree_util.tree_leaves(grad_law)
    bl = jax.tree_util.tree_leaves(basis)
    p = bl[0].shape[0]
    return np.array([sum(float(np.sum(np.asarray(g) * np.asarray(b)[j])) for g, b in zip(gl, bl)) for j in range(p)])


class Adjoint:
    """Backward sweep for problems of the sensitivity.py protocol (grad, hist, solve_many, free, fixed)."""

    def __init__(self, prob, law, out_index):
        self.prob, self.law, self.ip = prob, law, int(out_index)
        grad, hist, ip = prob.grad, prob.hist, self.ip

        def phi(q, dmax, Dv, a, b, dt, lw, g, mu, ed, ev):
            h = (dmax, Dv, a, b)
            r = grad(q, h, lw)
            return g * r[ip] - jnp.dot(mu, r) + _eta_dot(hist(q, dmax, Dv, dt, lw), ed, ev)

        def rhs(q, dmax, Dv, a, b, dt, lw, g, ed, ev):
            def f(qq):
                return g * grad(qq, (dmax, Dv, a, b), lw)[ip] + _eta_dot(hist(qq, dmax, Dv, dt, lw), ed, ev)
            return jax.grad(f)(q)

        self._rhs = jax.jit(rhs)
        self._back = jax.jit(jax.grad(phi, argnums=(1, 2, 6)))
        self.timers = dict(rhs=0.0, solve=0.0, back=0.0)

    def gradient(self, states, g, basis):
        """dJ/dtheta (p,) for dJ/dP_n = g[n] (n aligned with ``states``), in the directions ``basis``."""
        if len(g) != len(states):
            raise ValueError("g and states differ in length")
        fixed = self.prob.fixed
        ed = ev = None
        total = None
        for n in range(len(states) - 1, -1, -1):
            q, dmax, Dv, a, b, dt = states[n]
            q, dmax, Dv = jnp.asarray(q), jnp.asarray(dmax), jnp.asarray(Dv)
            a, b = jnp.asarray(a), jnp.asarray(b)
            if ed is None:
                ed, ev = jnp.zeros_like(dmax), jnp.zeros_like(Dv)
            gn = float(g[n])
            t0 = time.perf_counter()
            r = np.asarray(self._rhs(q, dmax, Dv, a, b, dt, self.law, gn, ed, ev))
            t1 = time.perf_counter()
            mu = np.array(self.prob.solve_many(q, (dmax, Dv, a, b), self.law, r[None, :])[0])
            mu[fixed] = 0.0
            t2 = time.perf_counter()
            ed, ev, gl = self._back(q, dmax, Dv, a, b, dt, self.law, gn, jnp.asarray(mu), ed, ev)
            jax.block_until_ready(ev)
            t3 = time.perf_counter()
            total = gl if total is None else jax.tree_util.tree_map(lambda x, y: x + y, total, gl)
            self.timers["rhs"] += t1 - t0
            self.timers["solve"] += t2 - t1
            self.timers["back"] += t3 - t2
        return contract(total, basis)


def _eta_dot(H, ed, ev):
    """eta . H_n for H = (dmax, D, Dv): the history state is (dmax, Dv); D is derived."""
    dmax, _, Dv = H
    return jnp.sum(ed * dmax) + jnp.sum(ev * Dv)
