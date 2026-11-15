"""T-peel model assembled from an ABAQUS deck: bulk + cohesive row + rigid couplings.  Step S3a.

Units: mm, N, MPa, N/mm, s.  Formulation: docs/solver/TPEEL.md.

Unknowns.  The *full* vector U has the BulkModel layout (2 dofs per deck node, then 4 incompatible
mode amplitudes per element); the reference-point (RP) nodes are ordinary deck nodes that no
element uses, so their two translational dofs in U are the RP translations.  The *reduced*
vector q = [U, theta_1, ..., theta_m] adds one rotation per kinematic coupling.

Kinematic coupling (ABAQUS *Coupling, *Kinematic) by elimination: the slave nodes s of coupling c
follow the finite rigid motion of their RP,
    u_s = u_RP + (R(theta) - I) (X_s - X_RP),
so their dofs in q are dependent (listed as eliminated, never solved for) and U = lift(q).

Derivatives by the chain rule, with the AD tangent K of the full energy on the plain mesh pattern
(no RP rows: a dense RP row would force ~150 extra colours on tatva's sparse jacfwd):
    r_q = J^T r_U,    K_q = J^T K_U J + G,
J = dU/dq (identity on independent dofs; slave rows: 1 on the RP translation, R'(theta) dX on theta)
and G the theta-theta term sum_s r_U,s . R''(theta) dX = -sum_s r_U,s . R(theta) dX.
"""
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sps
import jax
import jax.numpy as jnp
from tatva.sparse import ColoredMatrix
from tatva.sparse import jacfwd as sparse_jacfwd

from .elements import BulkModel
from .inp_import import cohesive_law, read_inp
from .interface import AlphaLawParams, CohesiveInterface
from .solve import pattern_from_elem_dofs


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def drot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


class TPeelModel:
    """Plane-strain T-peel from a build_tpeel_inp.py deck.

    Args:
        deck: a Deck (inp_import.read_inp) or a path to the .inp.
        kinematics: bulk energy, "svk" (default, D21 update), "hencky" or "linear".
        kind: "Q4IM" (CPE4I) or "Q4".
        quadrature: "nodal" (COH2D4, S2 probe) or "gauss".
        frame: cohesive separation kinematics (interface.py).  None (default) = what COH2D4 does
            for the deck's NLGEOM setting [D, S3a probes]: "corotational" (YES) or
            "corotational_linear" (NO).
        coupling: "finite" (R(theta)) or "linear" (u_s = u_RP + theta e3 x dX).  None (default) =
            "finite" for NLGEOM = YES, "linear" for NO.
        history: "monolithic" (S3b default: incremental potential at the frozen history of the
            previous increment, one Newton solve per increment) or "staggered" (S2: stiffness
            damage frozen, solve / update history / repeat).

    The energy, residual and tangent take the frozen-history arguments ``h`` returned by
    ``frozen_args(hist, dt)``: (dmax_n, Dv_n, a, b) for monolithic, (D_stiffness,) for staggered.
    """

    def __init__(self, deck, kinematics="svk", kind="Q4IM", quadrature="nodal", frame=None,
                 coupling=None, history="monolithic"):
        self.deck = read_inp(deck) if isinstance(deck, str) else deck
        d = self.deck
        nl = d.step.get("nlgeom", True)
        frame = frame or ("corotational" if nl else "corotational_linear")
        self.coupling = coupling or ("finite" if nl else "linear")
        self.history = history
        self.frame = frame
        self.coords = d.coords
        self.n_nodes = len(d.coords)
        conn, self.elset_of = d.bulk_elements()
        sec = d.solid_section()
        E, nu = d.materials[sec["material"]]["elastic"]["values"][:2]
        self.bulk = BulkModel(d.coords, conn, E, nu, thickness=sec["thickness"], kind=kind,
                              kinematics=kinematics)
        cs = d.cohesive_section()
        if abs(cs["constitutive_thickness"] - 1.0) > 0:
            raise ValueError("constitutive thickness must be 1 (separation = nominal strain)")
        bot, top = d.cohesive_chain(cs["elset"])
        self.coh = CohesiveInterface(d.coords, bot, top, width=cs["width"], quadrature=quadrature, frame=frame)
        self.law_kind, lp = cohesive_law(d, cs["material"])
        self.K0 = lp["K0"]
        self.law = AlphaLawParams(lp["K0"], lp["sigc"], lp["Gc"], lp["alpha"]) if self.law_kind == "alpha" else None
        self.law_params = lp
        ctl = d.controls.get(cs["controls"], {}) if cs["controls"] else {}
        self.viscosity = float(ctl.get("viscosity", 0.0))

        # layouts
        self.n_full = self.bulk.n_dof
        self.couplings = []
        for k, c in enumerate(d.couplings):
            if c["kind"] != "kinematic":
                raise ValueError("only kinematic couplings are supported")
            self.couplings.append(dict(name=c["name"], ref=c["ref"], slaves=np.asarray(c["slaves"]),
                                       dX=d.coords[c["slaves"]] - d.coords[c["ref"]],
                                       theta=self.n_full + k))
        self.n_q = self.n_full + len(self.couplings)
        slave_dofs = np.concatenate([(2 * c["slaves"][:, None] + np.arange(2)).ravel() for c in self.couplings]) \
            if self.couplings else np.zeros(0, np.int64)
        self.slave_dofs = np.sort(slave_dofs)

        # boundary conditions: (q index, value at the end of the step)
        self.bc_initial = self._bc_dofs(d.boundary_initial)
        self.bc_step = self._bc_dofs(d.boundary_step)
        fixed = np.r_[self.slave_dofs, [i for i, _ in self.bc_initial], [i for i, _ in self.bc_step]]
        self.fixed = np.unique(fixed.astype(np.int64))
        self.free = np.setdiff1d(np.arange(self.n_q), self.fixed)

        # named outputs
        rp_top = next(c for c in self.couplings if c["name"].upper() == "C_TOP")
        rp_bot = next(c for c in self.couplings if c["name"].upper() == "C_BOT")
        self.out = dict(rp1_u1=2 * rp_top["ref"], rp1_u2=2 * rp_top["ref"] + 1, rp1_ur3=rp_top["theta"],
                        rp2_u1=2 * rp_bot["ref"], rp2_u2=2 * rp_bot["ref"] + 1, rp2_ur3=rp_bot["theta"],
                        mouth_u_u2=2 * int(d.nsets["MOUTH_U"][0]) + 1,
                        mouth_l_u2=2 * int(d.nsets["MOUTH_L"][0]) + 1)

        # AD gradient and sparse tangent of the full energy on the mesh pattern
        pattern = pattern_from_elem_dofs([self.bulk.elem_dofs, self.coh.elem_dofs], self.n_full)
        self.colored = ColoredMatrix.from_csr(pattern)
        self.n_colors = int(np.asarray(self.colored.colors).max()) + 1
        grad = jax.grad(self.energy_full)
        self._grad = jax.jit(grad)
        self._jac = jax.jit(sparse_jacfwd(grad, self.colored))
        self._energy = jax.jit(self.energy_full)
        self.timers = defaultdict(float)        # cumulative wall time per phase (s), see _tick
        self.counts = defaultdict(int)

    def _tick(self, key, t0):
        """Add the time since t0 to timers[key]; return now (S3c profiling)."""
        t = time.perf_counter()
        self.timers[key] += t - t0
        self.counts[key] += 1
        return t

    def jit_cache_sizes(self):
        """Number of compiled variants of each jitted function (1 each = no recompilation)."""
        return {k: f._cache_size() for k, f in (("grad", self._grad), ("jac", self._jac), ("energy", self._energy))}

    # ---------------------------------------------------------------- energies
    def energy_full(self, U, *h):
        """Total energy (N*mm) of the full vector U at frozen history h (see frozen_args)."""
        u = U[: 2 * self.n_nodes].reshape(-1, 2)
        if self.history == "monolithic":
            dmax_n, Dv_n, a, b = h
            coh = self.coh.energy_incremental(u, dmax_n, Dv_n, self.law, a, b, self.K0)
        else:
            coh = self.coh.energy(u, h[0], self.K0)
        return self.bulk.energy(U) + coh

    def zero_history(self):
        return self.coh.zero_history()

    def frozen_args(self, hist, dt):
        """Arguments h of energy/residual/tangent for an increment of length dt from history hist."""
        if self.history == "monolithic":
            mu = self.viscosity
            a, b = (0.0, 1.0) if mu == 0.0 else (mu / (mu + dt), dt / (mu + dt))
            return (hist["dmax"], hist["Dv"], jnp.asarray(a), jnp.asarray(b))
        return (hist["Dv"],)

    def update_history(self, q, hist, dt):
        """History after the converged increment (S2 update_damage; no-op for the elastic law)."""
        if self.law is None:
            return hist
        u = jnp.asarray(self.lift(q)[: 2 * self.n_nodes].reshape(-1, 2))
        return self.coh.update_history(u, hist, self.law, self.viscosity, dt)

    def _bc_dofs(self, bcs):
        out = []
        for nodes, a, b, v in bcs:
            for n in nodes:
                cpl = next((c for c in self.couplings if c["ref"] == n), None)
                for dof in range(a, b + 1):
                    if dof in (1, 2):
                        out.append((2 * int(n) + dof - 1, v))
                    elif dof == 6 and cpl is not None:
                        out.append((cpl["theta"], v))
                    elif dof in (3, 4, 5, 6):
                        continue                       # out-of-plane / no rotation dof: inactive in 2D
        return out

    # ---------------------------------------------------------------- reduction
    def lift(self, q):
        """Full vector U from reduced q (slave dofs from the rigid motion of their RP)."""
        U = np.array(q[: self.n_full], dtype=float)
        for c in self.couplings:
            th = q[c["theta"]]
            urp = U[2 * c["ref"]: 2 * c["ref"] + 2]
            A = rot(th) - np.eye(2) if self.coupling == "finite" else th * drot(0.0)
            us = urp[None, :] + c["dX"] @ A.T
            U[2 * c["slaves"]] = us[:, 0]
            U[2 * c["slaves"] + 1] = us[:, 1]
        return U

    def lift_jacobian(self, q):
        """J = dU/dq (n_full x n_q, CSR)."""
        indep = np.setdiff1d(np.arange(self.n_full), self.slave_dofs)
        rows, cols, vals = [indep], [indep], [np.ones(len(indep))]
        for c in self.couplings:
            th = q[c["theta"]] if self.coupling == "finite" else 0.0
            dR = c["dX"] @ drot(th).T                               # (n_s, 2): d u_s / d theta
            for a in range(2):
                rs = 2 * c["slaves"] + a
                rows += [rs, rs]
                cols += [np.full(len(rs), 2 * c["ref"] + a), np.full(len(rs), c["theta"])]
                vals += [np.ones(len(rs)), dR[:, a]]
        return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.n_full, self.n_q))

    def residual(self, q, h):
        """Reduced residual r_q = J^T dE/dU (internal force; equals the reaction at fixed dofs)."""
        t0 = time.perf_counter()
        U = self.lift(q)
        t0 = self._tick("lift", t0)
        rU = np.asarray(self._grad(jnp.asarray(U), *h))
        t0 = self._tick("grad", t0)
        r = self.lift_jacobian(q).T @ rU
        self._tick("reduce_r", t0)
        return r

    def tangent(self, q, h, return_residual=False):
        t0 = time.perf_counter()
        U = jnp.asarray(self.lift(q))
        t0 = self._tick("lift", t0)
        KU = self._jac(U, *h)
        jax.block_until_ready(KU.data)
        t0 = self._tick("jac", t0)
        KU = KU.to_csr()
        t0 = self._tick("to_csr", t0)
        rU = np.asarray(self._grad(U, *h))
        t0 = self._tick("grad", t0)
        J = self.lift_jacobian(q)
        Kq = (J.T @ KU @ J).tocsr()
        G = np.zeros(self.n_q)
        for c in self.couplings if self.coupling == "finite" else []:
            Rd = c["dX"] @ rot(q[c["theta"]]).T                     # R dX, (n_s, 2)
            rs = rU[(2 * c["slaves"][:, None] + np.arange(2))]
            G[c["theta"]] = -np.sum(rs * Rd)
        Kq = Kq + sps.diags(G)
        self._tick("reduce_K", t0)
        return (Kq.tocsr(), J.T @ rU) if return_residual else Kq.tocsr()

    def energy(self, q, h):
        return float(self._energy(jnp.asarray(self.lift(q)), *h))

    # ---------------------------------------------------------------- helpers
    def zero_q(self):
        return np.zeros(self.n_q)

    def apply_bc(self, q, load_factor):
        """Set initial BCs and the step BCs scaled by load_factor (displacement ramp)."""
        q = np.array(q, dtype=float)
        for i, v in self.bc_initial:
            q[i] = v
        for i, v in self.bc_step:
            q[i] = load_factor * v
        return q

    def outputs(self, q, h, r=None):
        """History quantities named as in the ABAQUS deck output (RP1/RP2 U, UR3, RF; mouth U2)."""
        r = self.residual(q, h) if r is None else r
        U = self.lift(q)
        o = {k: float(U[i]) if i < self.n_full else float(q[i]) for k, i in self.out.items()}
        o["rp1_rf1"], o["rp1_rf2"] = float(r[self.out["rp1_u1"]]), float(r[self.out["rp1_u2"]])
        o["rp1_rm3"] = float(r[self.out["rp1_ur3"]])
        o["rp2_rf1"], o["rp2_rf2"] = float(r[self.out["rp2_u1"]]), float(r[self.out["rp2_u2"]])
        o["rp2_rm3"] = float(r[self.out["rp2_ur3"]])
        return o


def newton(model, q, h, rtol=1e-10, atol=1e-10, utol=1e-13, max_iter=25, linsolve=None, log=None):
    """Plain Newton on the free reduced dofs (fixed dofs of q hold the prescribed values).

    Converged when ||r_free||_inf <= max(atol, rtol * f_ref), f_ref = max(reaction ||r_fixed||_inf,
    initial ||r_free||_inf), or when the correction is at roundoff (||dq|| <= utol ||q||).
    Returns (q, info) with info = dict(iterations, residuals, converged, t_tangent, t_solve).
    """
    import time
    import scipy.sparse.linalg as spla
    linsolve = linsolve or (lambda A, b: spla.spsolve(A, b))
    q = np.array(q, dtype=float)
    free, fixed = model.free, model.fixed
    hist, r0, ok = [], None, False
    t_tan = t_sol = 0.0
    for it in range(max_iter + 1):
        t0 = time.perf_counter()
        K, r = model.tangent(q, h, return_residual=True)
        t_tan += time.perf_counter() - t0
        rf = r[free]
        r0 = np.abs(rf).max() if r0 is None else r0
        f_ref = max(np.abs(r[fixed]).max(), r0)
        hist.append(float(np.abs(rf).max()))
        if log:
            log("    newton %d |r|=%.3e f_ref=%.3e" % (it, hist[-1], f_ref))
        if hist[-1] <= max(atol, rtol * f_ref):
            ok = True
            break
        if it == max_iter:
            break
        t0 = time.perf_counter()
        dq = linsolve(K[free][:, free].tocsc(), rf)
        t_sol += time.perf_counter() - t0
        q[free] -= dq
        if it > 0 and np.abs(dq).max() <= utol * max(np.abs(q).max(), 1e-300):
            hist.append(float(np.abs(model.residual(q, h)[free]).max()))
            ok = True
            break
    q[model.slave_dofs] = model.lift(q)[model.slave_dofs]       # store dependent values for output
    return q, dict(iterations=it, residuals=hist, converged=ok, t_tangent=t_tan, t_solve=t_sol)
