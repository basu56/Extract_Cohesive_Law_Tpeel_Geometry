"""Element-local assembly of the T-peel system in the reduced unknowns q.  Step S3c.

Units: mm, N, MPa, N/mm, s.  Formulation: docs/solver/TPEEL.md section 6.

``TPeelModel`` (tpeel.py) differentiates the *global* energy of the full vector U with tatva's
coloured sparse jacfwd (34 JVPs on the T-peel pattern) and reduces to q with J^T K J + G in SciPy.
``ElementSystem`` computes the same derivatives element by element:

    E(q) = sum_e E_e(v_e),   v_e = local unknowns of element e gathered from q,
    r_q  = sum_e P_e^T grad E_e,   K_q = sum_e P_e^T hess E_e P_e,

with batched (vmap) element gradients / Hessians and a scatter into a fixed CSR pattern of the
free dofs.  Elements that touch the slave nodes of a kinematic coupling carry the coupling inside
the element: their local vector is [8 node dofs, (4 IM), u_RP (2), theta (1)], and a slave node's
displacement is computed from (u_RP, theta) by the same rigid map as TPeelModel.lift.  The Hessian of
that element energy therefore already contains J^T K J and the theta-theta term G; no global
reduction is needed.

Incompatible modes (S3c item 4, ``condense=True``): the 4 IM amplitudes of a Q4IM element appear in
that element only, so its local Hessian splits as [[K_kk, K_ki], [K_ik, K_ii]] (k = node / RP dofs,
i = IM).  The condensed system  S = K_kk - K_ki K_ii^-1 K_ik,  r_c = r_k - K_ki K_ii^-1 r_i  is
assembled from batched 4 x 4 solves, solved for du, and da = K_ii^-1 (r_i - K_ik du) is recovered per
element.  This is exactly the Newton step of the full system (same dq to roundoff).

Same energy as TPeelModel.energy_full to roundoff [D, tests/fe/test_s3c_assembly.py].

S5.0: the law parameters (an AlphaLawParams pytree, ``theta``) are an explicit argument of every
jitted kernel (energy, value_and_grad, hessian, update_history), so jax.jvp / jax.vjp with respect to
them is available on the production path (sensitivity.py).  ``self.theta`` (default ``model.law``) is
what the public methods pass; replacing it (``set_theta``) changes the law without recompiling.  The
cohesive stiffness everywhere (penalty in compression included) is theta.K0.
"""
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from tatva import element

from .interface import QUADRATURES, effective_separation, psi, psi_incremental, update_damage
from .kinematics import lame

_Q4 = element.Quad4()


# ------------------------------------------------------------------ element kernels
def _bulk_energy(u, amp, c, psi_fn, lam, mu, W):
    """Q4 / Q4IM element energy (N*mm).  u (4, 2) nodal displacements, amp (2 modes, 2 comps) or None,
    c: dNdX (q, 2 [j], 4 [n]), wdet (q,), g (q, 2 [k], 2 [i]) enhanced-mode gradients."""
    H = jnp.einsum("na,qjn->qaj", u, c["dNdX"])
    if amp is not None:
        H = H + jnp.einsum("ka,qki->qai", amp, c["g"])
    return W * jnp.sum(c["wdet"] * psi_fn(H, lam, mu))


def _separations(u, c, frame, N):
    """(delta_n, delta_s) at the quadrature points of one COH2D4 segment, nodes (b0, b1, t1, t0)
    (the CohesiveInterface.separations formulas for one segment)."""
    j = jnp.stack([u[3] - u[0], u[2] - u[1]])                     # jumps at node pairs 0, 1
    jq = N @ j                                                     # (q, 2)
    t0, n0, Gq = c["t0"], c["n0"], c["Gq"]
    if frame == "global":
        return jq @ n0, jq @ t0
    if frame == "corotational_linear":
        um0, um1 = 0.5 * (u[0] + u[3]), 0.5 * (u[1] + u[2])
        omega = jnp.dot(um1 - um0, n0) / c["L"]
        JG = jnp.stack([-Gq[:, 1], Gq[:, 0]], axis=-1)
        v = jq - omega * JG
        return v @ n0, v @ t0
    x = c["X"] + u
    seg = 0.5 * (x[1] + x[2]) - 0.5 * (x[0] + x[3])
    t = seg / jnp.linalg.norm(seg)
    n = jnp.stack([-t[1], t[0]])
    if frame == "midsurface":
        return jq @ n, jq @ t
    d = Gq + jq
    return d @ n - Gq @ n0, d @ t - Gq @ t0


def inv4(m):
    """Batched closed-form inverse of 4 x 4 matrices (..., 4, 4) by 2 x 2 cofactor expansion
    (vectorised; XLA's batched LAPACK solve is slow for many tiny systems on CPU).  Accuracy is
    checked through the condensed Newton step (tests/fe/test_s3c_assembly.py)."""
    a = [[m[..., i, j] for j in range(4)] for i in range(4)]
    s0 = a[0][0] * a[1][1] - a[1][0] * a[0][1]
    s1 = a[0][0] * a[1][2] - a[1][0] * a[0][2]
    s2 = a[0][0] * a[1][3] - a[1][0] * a[0][3]
    s3 = a[0][1] * a[1][2] - a[1][1] * a[0][2]
    s4 = a[0][1] * a[1][3] - a[1][1] * a[0][3]
    s5 = a[0][2] * a[1][3] - a[1][2] * a[0][3]
    c5 = a[2][2] * a[3][3] - a[3][2] * a[2][3]
    c4 = a[2][1] * a[3][3] - a[3][1] * a[2][3]
    c3 = a[2][1] * a[3][2] - a[3][1] * a[2][2]
    c2 = a[2][0] * a[3][3] - a[3][0] * a[2][3]
    c1 = a[2][0] * a[3][2] - a[3][0] * a[2][2]
    c0 = a[2][0] * a[3][1] - a[3][0] * a[2][1]
    det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    b = [[a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3, -a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3,
          a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3, -a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3],
         [-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1, a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1,
          -a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1, a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1],
         [a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0, -a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0,
          a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0, -a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0],
         [-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0, a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0,
          -a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0, a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0]]
    inv = jnp.stack([jnp.stack(row, axis=-1) for row in b], axis=-2)
    return inv / det[..., None, None]


class _Group:
    """Elements of one kind with a common local layout.

    qidx (E, n_loc): q index of each local unknown (n_q = dummy, for unused slots).
    For coupled groups also mask (E, 4) (1 = independent node, 0 = slave) and dX (E, 4, 2)."""

    def __init__(self, kind, sel, qidx, consts, coupled=False, mask=None, dX=None):
        self.kind, self.sel, self.coupled = kind, np.asarray(sel), coupled
        self.qidx = np.asarray(qidx, dtype=np.int64)
        self.consts = {k: jnp.asarray(v) for k, v in consts.items()}
        if coupled:
            self.consts["mask"], self.consts["dX"] = jnp.asarray(mask, float), jnp.asarray(dX)
        self.n, self.n_loc = self.qidx.shape


class ElementSystem:
    """Reduced T-peel system of a TPeelModel, assembled element by element (see module docstring).

    Args:
        model: TPeelModel (deck, layouts, BCs, law, history type and kinematics are taken from it).
        condense: statically condense the incompatible modes in the linear solve.
        linsolve: callable(A_csr, b) -> x on the assembled free (or condensed) system; default SuperLU.
    """

    def __init__(self, model, condense=False, linsolve=None, theta=None):
        m = self.model = model
        self.theta = model.law if theta is None else theta
        self.condense = bool(condense) and model.bulk.n_modes > 0
        self.n_q, self.n_full, self.n_nodes = m.n_q, m.n_full, m.n_nodes
        self.free, self.fixed, self.slave_dofs, self.out = m.free, m.fixed, m.slave_dofs, m.out
        self.history = m.history
        self.timers, self.counts = defaultdict(float), defaultdict(int)
        self.linsolve = linsolve
        bulk, coh = m.bulk, m.coh
        lam, mu = lame(bulk.E, bulk.nu)
        self._mat = (bulk._psi, lam, mu, bulk.thickness)
        self._coh_frame, self._coh_N = coh.frame, coh.N
        self._coh_w = coh.w
        self._W_coh = coh.width

        slave_of = {}
        for ci, c in enumerate(m.couplings):
            for s in c["slaves"]:
                slave_of[int(s)] = ci
        coupling = m.coupling

        # ---- bulk constants
        X = m.coords[bulk.elems]                                                # (E, 4, 2)
        qp, qw = np.asarray(_Q4.quad_points), np.asarray(_Q4.quad_weights)
        dN = np.stack([np.asarray(_Q4.shape_function_derivative(jnp.asarray(xi))) for xi in qp])  # (q, 2, 4)
        J = np.einsum("qin,enj->eqij", dN, X)
        dNdX = np.einsum("eqji,qin->eqjn", np.linalg.inv(J), dN)
        bc = dict(dNdX=dNdX, wdet=np.linalg.det(J) * qw[None, :])
        if bulk.n_modes:
            bc["g"] = np.asarray(bulk._g)
        ndofs = (2 * bulk.elems[:, :, None] + np.arange(2)).reshape(-1, 8)
        extra = bulk.elem_dofs[:, 8:]                                           # IM dofs (E, 4) or (E, 0)

        # ---- cohesive constants
        cb, ct = coh.bot, coh.top
        cnodes = np.stack([cb[:-1], cb[1:], ct[1:], ct[:-1]], axis=1)
        cc = dict(t0=np.asarray(coh.t0), n0=np.asarray(coh.n0), Gq=np.asarray(coh.Gq), L=coh.L,
                  X=m.coords[cnodes])
        cdofs = (2 * cnodes[:, :, None] + np.arange(2)).reshape(-1, 8)

        self.groups = []
        for kind, nodes, dofs, ext, consts in (("bulk", bulk.elems, ndofs, extra, bc),
                                               ("coh", cnodes, cdofs, np.zeros((len(cnodes), 0), np.int64), cc)):
            has = np.array([[int(n) in slave_of for n in row] for row in nodes])
            for coupled, sel in ((False, np.where(~has.any(1))[0]), (True, np.where(has.any(1))[0])):
                if sel.size == 0:
                    continue
                cs = {k: v[sel] for k, v in consts.items()}
                if not coupled:
                    self.groups.append(_Group(kind, sel, np.hstack([dofs[sel], ext[sel]]), cs))
                    continue
                refs, q_rp = [], []
                for row in nodes[sel]:
                    cis = {slave_of[int(n)] for n in row if int(n) in slave_of}
                    if len(cis) != 1:
                        raise ValueError("element touching slaves of more than one coupling")
                    c = m.couplings[cis.pop()]
                    refs.append(c["ref"])
                    q_rp.append([2 * c["ref"], 2 * c["ref"] + 1, c["theta"]])
                mask = np.array([[0.0 if int(n) in slave_of else 1.0 for n in row] for row in nodes[sel]])
                dX = m.coords[nodes[sel]] - m.coords[np.asarray(refs)][:, None, :]
                q_rp = np.asarray(q_rp)
                qn = dofs[sel].copy()
                qn[np.repeat(mask, 2, axis=1) == 0] = self.n_q                   # slave own dofs unused
                self.groups.append(_Group(kind, sel, np.hstack([qn, ext[sel], q_rp]), cs, True, mask, dX))
        self._finite = coupling == "finite"
        self._build_patterns()
        self._compile()

    # ------------------------------------------------------------------ element functions
    def _local_u(self, v, g, c):
        """Node displacements (4, 2) and IM amplitudes of one element from its local vector."""
        u = v[:8].reshape(4, 2)
        n_im = 4 if (g.kind == "bulk" and self.model.bulk.n_modes) else 0
        amp = v[8: 8 + n_im].reshape(2, 2) if n_im else None
        if g.coupled:
            urp, th = v[8 + n_im: 10 + n_im], v[10 + n_im]
            if self._finite:
                A = jnp.array([[jnp.cos(th) - 1.0, -jnp.sin(th)], [jnp.sin(th), jnp.cos(th) - 1.0]])
            else:
                A = th * jnp.array([[0.0, -1.0], [1.0, 0.0]])
            ur = urp[None, :] + c["dX"] @ A.T
            m = c["mask"][:, None]
            u = m * u + (1.0 - m) * ur
        return u, amp

    def _elem_energy(self, g):
        psi_fn, lam, mu, W = self._mat
        frame, N, w, Wc = self._coh_frame, self._coh_N, self._coh_w, self._W_coh
        model = self.model

        def f(v, c, hist, glob, law):
            u, amp = self._local_u(v, g, c)
            if g.kind == "bulk":
                return _bulk_energy(u, amp, c, psi_fn, lam, mu, W)
            dn, ds = _separations(u, c, frame, N)
            K0 = model.K0 if law is None else law.K0
            if model.history == "monolithic":
                dmax_n, Dv_n = hist
                a, b = glob
                dens = psi_incremental(dn, ds, dmax_n, Dv_n, law, a, b, K0)
            else:
                dens = psi(dn, ds, hist[0], K0)
            return Wc * jnp.sum(dens * w * 0.5 * c["L"])
        return f

    def _elem_seps(self, g):
        frame, N = self._coh_frame, self._coh_N

        def f(v, c):
            u, _ = self._local_u(v, g, c)
            return _separations(u, c, frame, N)
        return f

    def _hist_split(self, g, h):
        """Per-element history arguments of group g from the frozen-history tuple h."""
        if g.kind != "coh":
            return (), ()
        sel = jnp.asarray(g.sel)
        if self.history == "monolithic":
            return (h[0][sel], h[1][sel]), (h[2], h[3])
        return (h[0][sel],), ()

    # ------------------------------------------------------------------ patterns
    def _build_patterns(self):
        n_q = self.n_q
        im = np.zeros(n_q + 1, bool)
        if self.model.bulk.n_modes:
            im[2 * self.n_nodes: self.n_full] = True
        self.im_mask = im[:n_q]
        sysdofs = self.free[~self.im_mask[self.free]] if self.condense else self.free
        self.sys_dofs = sysdofs
        pos = np.full(n_q + 1, -1, np.int64)
        pos[sysdofs] = np.arange(len(sysdofs))
        self.n_sys = len(sysdofs)
        rows, cols, per = [], [], []
        for g in self.groups:
            loc = g.qidx
            if self.condense and g.kind == "bulk" and self.model.bulk.n_modes:
                g.k_slots = np.r_[0:8, 12:g.n_loc]                      # node (+ RP) slots
                g.i_slots = np.arange(8, 12)
            else:
                g.k_slots, g.i_slots = np.arange(g.n_loc), np.arange(0)
            p = pos[loc[:, g.k_slots]]                                   # (E, nk); -1 = not in system
            g.sys_pos = p
            nk = len(g.k_slots)
            R = np.repeat(p, nk, axis=1)
            C = np.tile(p, (1, nk))
            ok = (R >= 0) & (C >= 0)
            rows.append(R[ok]); cols.append(C[ok]); per.append((R, C, ok))
        A = sps.coo_matrix((np.ones(sum(r.size for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.n_sys, self.n_sys)).tocsr()
        A.sum_duplicates()
        A.sort_indices()
        self.indptr, self.indices = A.indptr.astype(np.int32), A.indices.astype(np.int32)
        self.nnz = A.nnz
        key = lambda r, c: r.astype(np.int64) * self.n_sys + c
        keys = key(np.repeat(np.arange(self.n_sys), np.diff(A.indptr)), A.indices)
        for g, (R, C, ok) in zip(self.groups, per):
            s = np.full(R.shape, self.nnz, np.int64)
            s[ok] = np.searchsorted(keys, key(R[ok], C[ok]))
            g.scatter = jnp.asarray(s.ravel())
            g.rpos = jnp.asarray(np.where(g.sys_pos >= 0, g.sys_pos, self.n_sys))
            g.qidx_j = jnp.asarray(g.qidx)

    # ------------------------------------------------------------------ compiled kernels
    def _compile(self):
        groups = self.groups
        n_q, nnz, n_sys = self.n_q, self.nnz, self.n_sys
        efs = [self._elem_energy(g) for g in groups]

        def gather(q, g):
            return jnp.append(q, 0.0)[g.qidx_j]

        def value_and_grad(q, h, law):
            E, r = 0.0, jnp.zeros(n_q + 1)
            for g, f in zip(groups, efs):
                hist, glob = self._hist_split(g, h)
                V = gather(q, g)
                e, ge = jax.vmap(jax.value_and_grad(f), in_axes=(0, 0, 0, None, None))(V, g.consts, hist, glob, law)
                E = E + jnp.sum(e)
                r = r.at[g.qidx_j.ravel()].add(ge.ravel())
            return E, r[:n_q]

        def energy(q, h, law):
            E = 0.0
            for g, f in zip(groups, efs):
                hist, glob = self._hist_split(g, h)
                E = E + jnp.sum(jax.vmap(f, in_axes=(0, 0, 0, None, None))(gather(q, g), g.consts, hist, glob, law))
            return E

        def hessian(q, h, r, law):
            """CSR data of the free (or condensed) system, its RHS, and the IM recovery data."""
            data, rhs, aux = jnp.zeros(nnz + 1), jnp.zeros(n_sys + 1), []
            rr = jnp.append(r, 0.0)
            for g, f in zip(groups, efs):
                hist, glob = self._hist_split(g, h)
                Ke = jax.vmap(jax.hessian(f), in_axes=(0, 0, 0, None, None))(gather(q, g), g.consts, hist, glob, law)
                re = rr[g.qidx_j]                                        # assembled residual at local slots
                ks, is_ = g.k_slots, g.i_slots
                if is_.size:
                    nk = len(ks)
                    perm = np.r_[ks, is_]
                    if not np.array_equal(perm, np.arange(len(perm))):     # coupled group: IM slots last
                        Ke, re = Ke[:, perm][:, :, perm], re[:, perm]
                    Kkk, Kki = Ke[:, :nk, :nk], Ke[:, :nk, nk:]
                    Kik, Kii = Ke[:, nk:, :nk], Ke[:, nk:, nk:]
                    Ai = inv4(Kii)
                    XK = jnp.einsum("eij,ejk->eik", Ai, Kik)                    # K_ii^-1 K_ik
                    Xr = jnp.einsum("eij,ej->ei", Ai, re[:, nk:])                # K_ii^-1 r_i
                    Kc = Kkk - jnp.einsum("eij,ejk->eik", Kki, XK)
                    # condensed RHS: r_k - K_ki K_ii^-1 r_i, the r_k part enters once via the global r
                    rhs = rhs.at[g.rpos.ravel()].add((-jnp.einsum("eij,ej->ei", Kki, Xr)).ravel())
                    aux.append((XK, Xr))
                else:
                    Kc = Ke
                    aux.append(None)
                data = data.at[g.scatter].add(Kc.reshape(Kc.shape[0], -1).ravel())
            rhs = rhs[:n_sys] + r[jnp.asarray(self.sys_dofs)]
            return data[:nnz], rhs, aux

        def recover(du_sys, aux):
            """Full dq (n_q) from the system solution: system dofs + recovered IM amplitudes."""
            dq = jnp.zeros(n_q + 1).at[jnp.asarray(self.sys_dofs)].set(du_sys)
            for g, ax in zip(groups, aux):
                if ax is None:
                    continue
                XK, Xr = ax
                du_e = dq[jnp.asarray(g.qidx[:, g.k_slots])]             # fixed / dummy slots are 0
                da = Xr - jnp.einsum("eik,ek->ei", XK, du_e)
                dq = dq.at[jnp.asarray(g.qidx[:, g.i_slots]).ravel()].set(da.ravel())
            return dq[:n_q]

        def hessian_multi(q, h, R, law):
            """hessian() for p right-hand sides R (p, n_q) on one tangent (S5a): CSR data, the condensed
            RHS (p, n_sys) and the recovery data [(XK, Xr (p, E, 4)) or None per group]."""
            data, rhs, aux = jnp.zeros(nnz + 1), jnp.zeros((R.shape[0], n_sys + 1)), []
            RR = jnp.concatenate([R, jnp.zeros((R.shape[0], 1))], axis=1)
            for g, f in zip(groups, efs):
                hist, glob = self._hist_split(g, h)
                Ke = jax.vmap(jax.hessian(f), in_axes=(0, 0, 0, None, None))(gather(q, g), g.consts, hist, glob, law)
                re = RR[:, g.qidx_j]                                     # (p, E, n_loc)
                ks, is_ = g.k_slots, g.i_slots
                if is_.size:
                    nk = len(ks)
                    perm = np.r_[ks, is_]
                    if not np.array_equal(perm, np.arange(len(perm))):
                        Ke, re = Ke[:, perm][:, :, perm], re[:, :, perm]
                    Kkk, Kki = Ke[:, :nk, :nk], Ke[:, :nk, nk:]
                    Kik, Kii = Ke[:, nk:, :nk], Ke[:, nk:, nk:]
                    Ai = inv4(Kii)
                    XK = jnp.einsum("eij,ejk->eik", Ai, Kik)
                    Xr = jnp.einsum("eij,pej->pei", Ai, re[:, :, nk:])
                    Kc = Kkk - jnp.einsum("eij,ejk->eik", Kki, XK)
                    rhs = rhs.at[:, g.rpos.ravel()].add(
                        (-jnp.einsum("eij,pej->pei", Kki, Xr)).reshape(R.shape[0], -1))
                    aux.append((XK, Xr))
                else:
                    Kc = Ke
                    aux.append(None)
                data = data.at[g.scatter].add(Kc.reshape(Kc.shape[0], -1).ravel())
            rhs = rhs[:, :n_sys] + R[:, jnp.asarray(self.sys_dofs)]
            return data[:nnz], rhs, aux

        def recover_multi(X, aux):
            """recover() for p solutions X (p, n_sys) -> (p, n_q)."""
            def one(x, xr):
                return recover(x, [None if ax is None else (ax[0], r) for ax, r in zip(aux, xr)])
            return jax.vmap(one)(X, [None if ax is None else ax[1] for ax in aux])

        def bulk_energy(q):
            E = 0.0
            for g, f in zip(groups, efs):
                if g.kind == "bulk":
                    E = E + jnp.sum(jax.vmap(f, in_axes=(0, 0, 0, None, None))(gather(q, g), g.consts, (), (), None))
            return E

        self._bulkE = jax.jit(bulk_energy)
        cg = [g for g in groups if g.kind == "coh"]
        sfs = [self._elem_seps(g) for g in cg]

        def separations(q):
            dn = jnp.zeros((self.model.coh.n_seg, self.model.coh.n_q))
            ds = jnp.zeros_like(dn)
            for g, f in zip(cg, sfs):
                a, b = jax.vmap(f)(gather(q, g), g.consts)
                dn, ds = dn.at[jnp.asarray(g.sel)].set(a), ds.at[jnp.asarray(g.sel)].set(b)
            return dn, ds

        mu_v = self.model.viscosity

        def update_history(q, dmax, Dv, dt, law):
            """S2 update_damage at the cohesive points: (dmax, D, Dv) after the converged q."""
            dn, ds = separations(q)
            return update_damage(effective_separation(dn, ds), dmax, Dv, law, mu_v, dt)

        self._raw = dict(value_and_grad=value_and_grad, update_history=update_history, separations=separations,
                         hessian_multi=hessian_multi, recover_multi=recover_multi)
        self._vg = jax.jit(value_and_grad)
        self._energy = jax.jit(energy)
        self._hess = jax.jit(hessian)
        self._recover = jax.jit(recover)
        self._seps = jax.jit(separations)
        self._upd = jax.jit(update_history) if self.model.law is not None else None
        self._hess_multi = jax.jit(hessian_multi)
        self._recover_multi = jax.jit(recover_multi)

    # ------------------------------------------------------------------ public API (TPeelModel-like)
    def _tick(self, key, t0):
        t = time.perf_counter()
        self.timers[key] += t - t0
        self.counts[key] += 1
        return t

    def set_theta(self, theta):
        """Use the law ``theta`` (same pytree structure as model.law) in every following call.  Kernels
        take it as an argument, so this does not recompile (S5.0)."""
        self.theta = theta

    def jit_cache_sizes(self):
        return {k: getattr(self, "_" + k)._cache_size() for k in ("vg", "energy", "hess", "recover")}

    def zero_history(self):
        return self.model.zero_history()

    def frozen_args(self, hist, dt):
        return self.model.frozen_args(hist, dt)

    def apply_bc(self, q, lam):
        return self.model.apply_bc(q, lam)

    def zero_q(self):
        return self.model.zero_q()

    def lift(self, q):
        return self.model.lift(q)

    def value_and_grad(self, q, h):
        t0 = time.perf_counter()
        E, r = self._vg(jnp.asarray(q), h, self.theta)
        E, r = float(E), np.asarray(r)
        self._tick("value_and_grad", t0)
        return E, r

    def residual(self, q, h):
        return self.value_and_grad(q, h)[1]

    def energy(self, q, h):
        return float(self._energy(jnp.asarray(q), h, self.theta))

    def system(self, q, h, r):
        """(A (CSR, n_sys x n_sys), rhs, aux) of the Newton step at q (r = residual at q)."""
        t0 = time.perf_counter()
        data, rhs, aux = self._hess(jnp.asarray(q), h, jnp.asarray(r), self.theta)
        jax.block_until_ready(data)
        t0 = self._tick("hessian", t0)
        A = sps.csr_matrix((np.asarray(data), self.indices, self.indptr), shape=(self.n_sys, self.n_sys))
        self._tick("to_csr", t0)
        return A, np.asarray(rhs), aux

    def direction(self, q, h, r, solve):
        """Newton correction dq (n_q; zero at fixed dofs) with K dq = r on the free dofs."""
        A, rhs, aux = self.system(q, h, r)
        t0 = time.perf_counter()
        x = solve(A, rhs)
        t0 = self._tick("solve", t0)
        dq = np.asarray(self._recover(jnp.asarray(x), aux)) if self.condense else self._expand(x)
        self._tick("recover", t0)
        return dq

    def _expand(self, x):
        dq = np.zeros(self.n_q)
        dq[self.sys_dofs] = x
        return dq

    def tangent_free(self, q, h):
        """Uncondensed free-free tangent (CSR, in self.free order); for checks against TPeelModel."""
        if self.condense:
            raise ValueError("tangent_free needs condense=False")
        _, r = self.value_and_grad(q, h)
        return self.system(q, h, r)[0]

    def update_history(self, q, hist, dt):
        if self._upd is None:
            return hist
        t0 = time.perf_counter()
        dmax, D, Dv = self._upd(jnp.asarray(q), hist["dmax"], hist["Dv"], dt, self.theta)
        out = dict(dmax=dmax, D=D, Dv=Dv)
        jax.block_until_ready(Dv)
        self._tick("history", t0)
        return out

    def bulk_energy(self, q):
        """Strain energy of the adherends (N*mm)."""
        return float(self._bulkE(jnp.asarray(q)))

    def separations(self, q):
        return self._seps(jnp.asarray(q))
