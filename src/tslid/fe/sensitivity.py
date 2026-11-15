"""Direct (forward) sensitivities of the discrete incremental response.  Step S5a.

Units: mm, N, MPa, s.  Derivation and notation: docs/solver/ADJOINT.md; plan: S5_INSTRUCTIONS.md (D28).

Per converged increment n (realized schedule, converged q_n, history h_{n-1} = (dmax, Dv) it was
solved with, frozen arguments (dmax, Dv, a_n, b_n)), for p parameter directions e_j of the law pytree
(theta = (ln K0, ln sigc, ln Gc, alpha) by default, ``alpha_log_basis``), with S^h_0 = 0:

    K_n S^q_n = -(dR/dh . S^h_{n-1} + dR/dtheta . e_j)          on the free dofs, S^q = 0 on fixed dofs
    S^h_n     = dH/dq . S^q_n + dH/dh . S^h_{n-1} + dH/dtheta . e_j
    dy_n      = dY/dq . S^q_n + dY/dh . S^h_{n-1} + dY/dtheta . e_j

R = gradient of the incremental potential in q at frozen history (the Newton residual), H = the history
update (interface.update_damage), Y = outputs: a component of R at a prescribed dof (the reaction, e.g.
P = RF2 at RP1) or a component of q (e.g. a mouth U2).  Each bracket is one jax.jvp of R, H (or Y) in
the direction (S^q, S^h, e_j), vmapped over j; the Newton loop and the driver are never differentiated.
The solve is the Newton system at the converged q_n (one Hessian + one factorization, p right-hand
sides, incompatible modes condensed as in Newton).  Prescribed loads lambda_n do not depend on theta.

Kinks: the history update uses the derivative of the active branch (dmax = dm on loading, dm > dmax_old;
a tie counts as unloading) and jnp.where elsewhere; per increment the number of quadrature points within
a relative 1e-8 of the loading switch (dm vs max(dmax_old, d0)) and of failure (dmax vs df) is recorded.

Problem protocol (``ElementProblem`` for the T-peel, ``DenseProblem`` for small test problems):
    grad(q, h, law) -> R (n_q,)            pure JAX; h = (dmax, Dv, a, b)
    hist(q, dmax, Dv, dt, law) -> (dmax, D, Dv)   pure JAX
    dm(q) -> effective separation at the history points (pure JAX)
    solve_many(q, h, law, B) -> X (p, n_q)  K X^T = B^T on the free dofs, 0 elsewhere (NumPy)
    free, fixed: index arrays;  outputs: {name: ("r" | "q", index)}
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps

from .interface import AlphaLawParams, effective_separation

ALPHA_LOG = ("ln_K0", "ln_sigc", "ln_Gc", "alpha")


def alpha_log_basis(law):
    """Tangent pytree with a leading axis p = 4: d/d ln K0, d/d ln sigc, d/d ln Gc, d/d alpha."""
    z = np.zeros(4)
    return AlphaLawParams(K0=jnp.asarray(z + np.eye(4)[0] * law.K0), sigc=jnp.asarray(z + np.eye(4)[1] * law.sigc),
                          Gc=jnp.asarray(z + np.eye(4)[2] * law.Gc), alpha=jnp.asarray(z + np.eye(4)[3]))


def identity_basis(law):
    """Tangent pytree with one direction per scalar of every leaf (p = total size), e.g. all knot
    tractions of a PiecewiseLawParams.  Returns (basis, names)."""
    leaves, tdef = jax.tree_util.tree_flatten(law)
    sizes = [int(np.size(x)) for x in leaves]
    p = sum(sizes)
    out, names, off = [], [], 0
    import dataclasses
    fields = [f.name for f in dataclasses.fields(law) if f.name != "u"]       # data fields, in leaf order
    for x, n, f in zip(leaves, sizes, fields):
        e = np.zeros((p,) + np.shape(x))
        e.reshape(p, -1)[off: off + n, :] = np.eye(n)
        out.append(jnp.asarray(e))
        names += [f] if n == 1 and np.ndim(x) == 0 else [f"{f}[{i}]" for i in range(n)]
        off += n
    return jax.tree_util.tree_unflatten(tdef, out), names


def subset_basis(basis, idx):
    """Rows idx of a basis pytree."""
    idx = np.asarray(idx)
    return jax.tree_util.tree_map(lambda x: x[idx], basis)


def perturb_alpha_log(law, name, h):
    """Law with theta_j -> theta_j + h in the ALPHA_LOG coordinates (finite differences)."""
    K0, sigc, Gc, alpha = law.K0, law.sigc, law.Gc, law.alpha
    if name == "ln_K0":
        K0 = K0 * float(np.exp(h))
    elif name == "ln_sigc":
        sigc = sigc * float(np.exp(h))
    elif name == "ln_Gc":
        Gc = Gc * float(np.exp(h))
    elif name == "alpha":
        alpha = alpha + h
    else:
        raise ValueError(name)
    return AlphaLawParams(K0, sigc, Gc, alpha)


class ElementProblem:
    """Sensitivity protocol of an assembly.ElementSystem (D24 production path).

    solver: a linsolve solver object (callable(A, B) with B (n, p)); the PARDISO solvers do one
    numerical factorization for all p columns."""

    def __init__(self, system, solver):
        S = self.S = system
        self.solver = solver
        self.free, self.fixed = S.free, S.fixed
        vg, upd, seps = S._raw["value_and_grad"], S._raw["update_history"], S._raw["separations"]
        self.grad = lambda q, h, law: vg(q, h, law)[1]
        self.hist = upd
        self.dm = lambda q: effective_separation(*seps(q))
        o = S.out
        self.outputs = {"P": ("r", o["rp1_u2"]), "mouth_u": ("q", o["mouth_u_u2"]), "mouth_l": ("q", o["mouth_l_u2"])}
        for name, (kind, i) in self.outputs.items():
            if kind == "q" and i in set(S.slave_dofs.tolist()):
                raise ValueError("output %s is a slave dof (would need the lift)" % name)
        self.timers = dict(hessian=0.0, solve=0.0, recover=0.0)

    def solve_many(self, q, h, law, B):
        S = self.S
        t0 = time.perf_counter()
        data, rhs, aux = S._hess_multi(jnp.asarray(q), h, jnp.asarray(B), law)
        jax.block_until_ready(data)
        A = sps.csr_matrix((np.asarray(data), S.indices, S.indptr), shape=(S.n_sys, S.n_sys))
        t1 = time.perf_counter()
        X = np.asarray(self.solver(A, np.asarray(rhs).T)).reshape(S.n_sys, -1)
        t2 = time.perf_counter()
        if S.condense:
            dq = np.asarray(S._recover_multi(jnp.asarray(X.T), aux))
        else:
            dq = np.zeros((B.shape[0], S.n_q))
            dq[:, S.sys_dofs] = X.T
        t3 = time.perf_counter()
        self.timers["hessian"] += t1 - t0
        self.timers["solve"] += t2 - t1
        self.timers["recover"] += t3 - t2
        return dq


class DenseProblem:
    """Sensitivity protocol for a small problem given by its incremental potential
    energy(q, h, law) (dense Hessian; tests: single cohesive element, two-block bar)."""

    def __init__(self, energy, hist, dm, n_q, free, outputs):
        self.grad = jax.grad(energy)
        self._K = jax.jit(jax.hessian(energy))
        self.hist, self.dm = hist, dm
        self.n_q = n_q
        self.free = np.asarray(free, dtype=np.int64)
        self.fixed = np.setdiff1d(np.arange(n_q), self.free)
        self.outputs = outputs

    def solve_many(self, q, h, law, B):
        X = np.zeros((B.shape[0], self.n_q))
        if self.free.size:
            K = np.asarray(self._K(jnp.asarray(q), h, law))
            f = self.free
            X[:, f] = np.linalg.solve(K[np.ix_(f, f)], np.asarray(B)[:, f].T).T
        return X


class DirectSensitivity:
    """Forward sensitivity recursion; call ``step`` after every converged increment, in order.

    Args:
        prob: problem protocol (module docstring).
        law: the law pytree at which derivatives are taken (AlphaLawParams).
        basis: tangent pytree with leading axis p (default alpha_log_basis(law)), names: p labels.
    Each step appends to ``rows`` a dict with dy (p,) per output and diagnostics:
        res_free  max |dR_free/dtheta (total)| / max(|B_free|, tiny): consistency of the solve
        n_switch  points within 1e-8 (relative) of the loading switch; n_fail: of failure (dmax vs df)
        finite    all sensitivities finite;  wall: seconds spent in step.
    """

    def __init__(self, prob, law, basis=None, names=ALPHA_LOG, switch_tol=1e-8):
        self.prob, self.law = prob, law
        self.basis = alpha_log_basis(law) if basis is None else basis
        self.names = tuple(names)
        self.p = len(self.names)
        self.switch_tol = switch_tol
        self.Sh = None
        self.rows = []
        self.wall = 0.0
        grad, hist, dm = prob.grad, prob.hist, prob.dm

        def partial_rhs(q, h, law, Sd, Sv, E):
            """dR/dh . S^h + dR/dtheta . e_j for all j: (p, n_q)."""
            def one(sd, sv, e):
                th = (sd, sv, jnp.zeros_like(h[2]), jnp.zeros_like(h[3]))
                return jax.jvp(lambda hh, l: grad(q, hh, l), (h, law), (th, e))[1]
            return jax.vmap(one)(Sd, Sv, E)

        def total(q, h, law, Sq, Sd, Sv, E, dt):
            """Total derivatives: dR (p, n_q) and the new history tangents (dmax, D, Dv) (p, ...)."""
            def one(sq, sd, sv, e):
                th = (sd, sv, jnp.zeros_like(h[2]), jnp.zeros_like(h[3]))
                dR = jax.jvp(lambda qq, hh, l: grad(qq, hh, l), (q, h, law), (sq, th, e))[1]
                dH = jax.jvp(lambda qq, a, b, l: hist(qq, a, b, dt, l), (q, h[0], h[1], law), (sq, sd, sv, e))[1]
                return dR, dH
            return jax.vmap(one)(Sq, Sd, Sv, E)

        def kinks(q, dmax_old, dmax_new, law):
            d0, df = law.d0, law.df
            m = dm(q)
            dstar = jnp.maximum(dmax_old, d0)
            n_sw = jnp.sum(jnp.abs(m - dstar) <= self.switch_tol * dstar)
            n_fail = jnp.sum(jnp.abs(dmax_new - df) <= self.switch_tol * df)
            return n_sw, n_fail

        self._rhs = jax.jit(partial_rhs)
        self._total = jax.jit(total)
        self._kinks = jax.jit(kinks)

    def step(self, q, hist_old, h, hist_new, dt):
        t0 = time.perf_counter()
        q = jnp.asarray(q)
        if self.Sh is None:
            z = jnp.zeros((self.p,) + tuple(np.shape(hist_old["dmax"])))
            self.Sh = (z, z)
        Sd, Sv = self.Sh
        B = np.asarray(self._rhs(q, h, self.law, Sd, Sv, self.basis))
        Sq = -self.prob.solve_many(q, h, self.law, B)
        Sq[:, self.prob.fixed] = 0.0
        dR, (sdm, sD, sDv) = self._total(q, h, self.law, jnp.asarray(Sq), Sd, Sv, self.basis, dt)
        dR = np.asarray(dR)
        f = self.prob.free
        scale = max(float(np.abs(B[:, f]).max()) if f.size else 0.0, 1e-300)
        row = dict(res_free=float(np.abs(dR[:, f]).max()) / scale if f.size else 0.0)
        for name, (kind, i) in self.prob.outputs.items():
            row[name] = dR[:, i].copy() if kind == "r" else Sq[:, i].copy()
        n_sw, n_fail = self._kinks(q, hist_old["dmax"], hist_new["dmax"], self.law)
        row.update(n_switch=int(n_sw), n_fail=int(n_fail),
                   finite=bool(np.all(np.isfinite(Sq)) and np.all(np.isfinite(np.asarray(sDv)))
                               and np.all(np.isfinite(dR))))
        self.Sh = (sdm, sDv)
        jax.block_until_ready(sDv)
        row["wall"] = time.perf_counter() - t0
        self.wall += row["wall"]
        self.rows.append(row)
        return row

    def as_arrays(self):
        """{output: (n_inc, p)} plus the diagnostics as (n_inc,) arrays."""
        out = {}
        for k in self.rows[0] if self.rows else ():
            out[k] = np.array([r[k] for r in self.rows])
        return out
