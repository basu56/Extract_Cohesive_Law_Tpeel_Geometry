"""Cohesive interface: separations, energy with frozen damage, history update.  Step S2.

Units: mm, N, MPa, N/mm.  Formulation and ABAQUS correspondence: docs/solver/INTERFACE.md.

The bond line is a chain of matched node pairs: lower face nodes ``bot[k]`` and upper face nodes
``top[k]`` (k = 0..n), forming n segments (Line2 on each face; a COH2D4-like element per segment).
Constitutive thickness 1, so the separation equals the displacement jump [[u]] = u_top - u_bot
(the geometric gap, 0.4 mm in the T-peel, only sets the initial geometry).

Separations at a quadrature point (node pair k, gap vector G_k = X_top,k - X_bot,k, current gap
d_k = x_top,k - x_bot,k = G_k + [[u]]_k), selected by ``frame``:
    "corotational" (S3a default; COH2D4 under NLGEOM [D, probes s3a_cohprobe]):
        delta_n = d . n - G . n0,  delta_s = d . t - G . t0,
        (t, n) the deformed mid-surface frame of the segment, (t0, n0) the reference one.
        Objective: a rigid rotation of the element gives zero separation even with a geometric gap.
    "corotational_linear" (COH2D4 with NLGEOM = NO [D]): the linearisation,
        delta = R0^T ([[u]] - omega J G), omega = (du_mid . n0)/L the small mid-surface rotation,
        J the 90-degree rotation; e.g. delta_s = [[u]] . t0 + omega g for a gap g along n0.
    "global" (S2): delta = R0^T [[u]] (correct only for zero gap or zero rotation).
    "midsurface": delta = R^T [[u]] in the deformed frame, no gap correction (kept for comparison;
        not what COH2D4 does).

Energy density with damage frozen at the history value D (D_v when viscous):
    psi = (1 - D) K0/2 (<delta_n>_+^2 + delta_s^2) + K0/2 <-delta_n>_+^2
so the traction jax.grad(psi) is the ABAQUS secant law with no damage in compression.
History: delta_m,max = max over time of delta_m = sqrt(<delta_n>_+^2 + delta_s^2) (QUADS with equal
normal/shear strengths and stiffnesses initiates at K0 delta_m = sigc for every mode mix; evolution
is in delta_m only, no mixed-mode keyword).  Viscous regularisation (ABAQUS): dD_v/dt = (D - D_v)/mu,
backward Euler: D_v,n+1 = (dt D_n+1 + mu D_v,n) / (mu + dt).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .laws_jax import damage as damage_alpha
from .laws_jax import envelope_integral as envelope_alpha
from .laws_jax import law_params as law_params_alpha

GAUSS2 = (np.array([-1.0, 1.0]) / np.sqrt(3.0), np.array([1.0, 1.0]))
NODAL2 = (np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
QUADRATURES = {"gauss": GAUSS2, "nodal": NODAL2}


def pos(x):
    return jnp.maximum(x, 0.0)


def safe_norm(a, b):
    """sqrt(a^2 + b^2) with a finite (zero) gradient at the origin."""
    r2 = a * a + b * b
    nz = r2 > 0.0
    return jnp.where(nz, jnp.sqrt(jnp.where(nz, r2, 1.0)), 0.0)


def psi(dn, ds, D, K0):
    """Interface energy density (N/mm per unit area = N*mm/mm^2) with damage D frozen."""
    return (1.0 - D) * 0.5 * K0 * (pos(dn) ** 2 + ds ** 2) + 0.5 * K0 * pos(-dn) ** 2


def traction(dn, ds, D, K0):
    """(t_n, t_s) = grad of psi wrt (dn, ds) at frozen D (pointwise; psi is elementwise)."""
    return jax.grad(lambda a, b: jnp.sum(psi(a, b, D, K0)), argnums=(0, 1))(dn, ds)


def effective_separation(dn, ds):
    return safe_norm(pos(dn), ds)


@dataclass(frozen=True)
class AlphaLawParams:
    """Analytic alpha-family law (ABAQUS LINEAR / EXPONENTIAL).

    A JAX pytree with leaves (K0, sigc, Gc, alpha) (S5.0): it is passed as a differentiable argument
    of the jitted energy / residual / tangent / history kernels (assembly.py), never closed over."""
    K0: float
    sigc: float
    Gc: float
    alpha: float

    def damage(self, dmax):
        return damage_alpha(dmax, self.K0, self.sigc, self.Gc, self.alpha)

    @property
    def d0(self):
        return self.sigc / self.K0

    @property
    def df(self):
        return law_params_alpha(self.K0, self.sigc, self.Gc, self.alpha)[1]

    def envelope_integral(self, d):
        """int_{d0}^{d} T(s) ds, d >= d0 (closed form, laws_jax.envelope_integral)."""
        return envelope_alpha(d, self.K0, self.sigc, self.Gc, self.alpha)


jax.tree_util.register_dataclass(AlphaLawParams, data_fields=["K0", "sigc", "Gc", "alpha"], meta_fields=[])


@dataclass(frozen=True)
class TabularLawParams:
    """ABAQUS TABULAR representation: D interpolated linearly in u = delta_m - d0 from a
    (D, u) table (tslid.laws.AlphaLaw.tabular), D = 0 below d0 and 1 beyond the table."""
    K0: float
    d0: float
    D_tab: tuple
    u_tab: tuple

    @classmethod
    def from_alpha_law(cls, law, n=1000):
        tab = law.tabular(n)
        return cls(law.K, law.d0, tuple(t[0] for t in tab), tuple(t[1] for t in tab))

    def damage(self, dmax):
        u = dmax - self.d0
        D = jnp.interp(u, jnp.asarray(self.u_tab), jnp.asarray(self.D_tab), left=0.0, right=1.0)
        return jnp.where(u <= 0.0, 0.0, D)

    def envelope_integral(self, d):
        """int_{d0}^{d} T(s) ds, d >= d0, with T(s) = (1 - D(s)) K0 s and D linear in s on each
        table segment: T is quadratic per segment, integrated exactly (cumulative sums at the
        table nodes + the partial segment).  Zero traction beyond the table (D = 1)."""
        sk = self.d0 + np.asarray(self.u_tab, dtype=float)
        Dk = np.asarray(self.D_tab, dtype=float)
        m = np.diff(Dk) / np.diff(sk)                                     # dD/ds per segment
        c = 1.0 - Dk[:-1] + m * sk[:-1]                                   # T = K0 s (c - m s)

        def seg(k, a, b):
            return self.K0 * (c[k] * (b ** 2 - a ** 2) / 2 - m[k] * (b ** 3 - a ** 3) / 3)

        C = np.r_[0.0, np.cumsum(seg(np.arange(len(m)), sk[:-1], sk[1:]))]
        dc = jnp.clip(d, sk[0], sk[-1])
        k = jnp.clip(jnp.searchsorted(jnp.asarray(sk), dc, side="right") - 1, 0, len(m) - 1)
        ck, mk, s0 = jnp.asarray(c)[k], jnp.asarray(m)[k], jnp.asarray(sk)[k]
        return jnp.asarray(C)[k] + self.K0 * (ck * (dc ** 2 - s0 ** 2) / 2 - mk * (dc ** 3 - s0 ** 3) / 3)


@dataclass(frozen=True)
class PiecewiseLawParams:
    """Piecewise-linear traction envelope with the knot tractions as parameters (S5b many-parameter
    law; chosen over TabularLawParams, whose envelope integral is built in NumPy from the table and is
    not differentiable in the table values).

    Leaves (differentiable): K0 (MPa/mm), sigc (MPa), t (m-1 interior knot tractions, MPa).
    Static: u (m+1 knot offsets u_0 = 0 < ... < u_m from d0, mm).  Knots s_k = d0 + u_k, d0 = sigc/K0:
        T = K0 d (d <= d0);  linear through (s_0, sigc), (s_k, t_k), (s_m, 0);  0 beyond df = s_m.
    D(d) = 1 - T(d)/(K0 d); F(d) = int_{d0}^{d} T exactly (trapezoids + partial segment), so the
    incremental potential is closed-form and differentiable in every leaf.  G_c = sigc d0/2 + F(df)."""
    K0: object
    sigc: object
    t: object
    u: tuple

    @classmethod
    def from_alpha(cls, law, m):
        """m equal segments on [d0, df] of an AlphaLawParams, knot tractions from its envelope."""
        from .laws_jax import traction as traction_alpha
        d0, df = float(law.d0), float(law.df)
        u = tuple(float(x) for x in np.linspace(0.0, df - d0, m + 1))
        t = np.array([float(traction_alpha(d0 + x, law.K0, law.sigc, law.Gc, law.alpha)) for x in u[1:-1]])
        return cls(float(law.K0), float(law.sigc), jnp.asarray(t), u)

    @property
    def d0(self):
        return self.sigc / self.K0

    @property
    def df(self):
        return self.d0 + self.u[-1]

    def _knots(self):
        s = self.d0 + jnp.asarray(self.u)
        T = jnp.concatenate([jnp.reshape(self.sigc, (1,)), jnp.asarray(self.t), jnp.zeros(1)])
        return s, T

    def _segment(self, d):
        s, T = self._knots()
        dc = jnp.clip(d, s[0], s[-1])
        k = jnp.clip(jnp.searchsorted(s, dc, side="right") - 1, 0, len(self.u) - 2)
        Td = T[k] + (T[k + 1] - T[k]) * (dc - s[k]) / (s[k + 1] - s[k])
        return s, T, dc, k, Td

    def damage(self, dmax):
        s, T, dc, k, Td = self._segment(dmax)
        soft = (dmax > s[0]) & (dmax < s[-1])
        d_safe = jnp.where(soft, dmax, s[-1])
        Ds = 1.0 - jnp.where(soft, Td, 0.0) / (self.K0 * d_safe)
        return jnp.where(dmax <= s[0], 0.0, jnp.where(dmax >= s[-1], 1.0, Ds))

    def envelope_integral(self, d):
        s, T, dc, k, Td = self._segment(d)
        C = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (T[1:] + T[:-1]) * jnp.diff(s))])
        return C[k] + 0.5 * (T[k] + Td) * (dc - s[k])

    @property
    def Gc(self):
        return 0.5 * self.sigc * self.d0 + self.envelope_integral(self.df)


jax.tree_util.register_dataclass(PiecewiseLawParams, data_fields=["K0", "sigc", "t"], meta_fields=["u"])


def update_damage(dm, dmax_old, Dv_old, law, mu=0.0, dt=1.0):
    """History update at quadrature points.  Returns (dmax, D, Dv): D inviscid, Dv the regularised
    damage used for the stiffness (Dv = D when mu = 0).

    dmax = dm on loading (dm > dmax_old), else dmax_old (S5): written with where, not maximum, so the
    derivative is that of the active branch, with a tie counted as unloading (jnp.maximum would split
    a tie's derivative 1/2 : 1/2).  Same values as max(dmax_old, dm)."""
    dmax = jnp.where(dm > dmax_old, dm, dmax_old)
    D = law.damage(dmax)
    Dv = D if mu == 0.0 else (dt * D + mu * Dv_old) / (mu + dt)
    return dmax, D, Dv


def psi_incremental(dn, ds, dmax_n, Dv_n, law, a, b, K0):
    """Incremental potential of the cohesive law at frozen history (Step S3b; docs/solver/INTERFACE.md).

    With r = dt/mu, a = 1/(1 + r) = mu/(mu + dt) and b = r/(1 + r) = dt/(mu + dt) (inviscid: a = 0,
    b = 1):
        Phi = a (1 - Dv_n) K0/2 dm^2 + b Psi_env(dm) + K0/2 <-dn>_+^2,  dm^2 = <dn>_+^2 + ds^2,
        Psi_env = (1 - D_n) K0/2 dm^2                                   (dm <= d*)
                = (1 - D_n) K0/2 d*^2 + F(dm) - F(d*)                   (dm >  d*)
    with d* = max(dmax_n, d0), D_n = D(dmax_n) and F the envelope integral of the law.  Its gradient
    in (dn, ds) is exactly the traction of update_damage (dmax_n+1 = max(dmax_n, dm), D_n+1 = D(dmax_n+1),
    Dv_n+1 = a Dv_n + b D_n+1) evaluated at the new separation: a single Newton solve per increment
    replaces the staggered solve/update loop.  The quadratic branch is written in dm^2 (smooth at
    dm = 0); the other branch takes sqrt only of dm^2 > d*^2 > 0 (double-where).  law=None: elastic.
    """
    dm2 = pos(dn) ** 2 + ds ** 2
    comp = 0.5 * K0 * pos(-dn) ** 2
    if law is None:
        return 0.5 * K0 * dm2 + comp
    D_n = law.damage(dmax_n)
    dstar = jnp.maximum(dmax_n, law.d0)
    quad = dm2 <= dstar ** 2
    dm = jnp.sqrt(jnp.where(quad, dstar ** 2 + 1.0, dm2))
    dm = jnp.where(quad, dstar, dm)
    env_q = 0.5 * (1.0 - D_n) * K0 * dm2
    env_s = 0.5 * (1.0 - D_n) * K0 * dstar ** 2 + law.envelope_integral(dm) - law.envelope_integral(dstar)
    env = jnp.where(quad, env_q, env_s)
    return a * (1.0 - Dv_n) * 0.5 * K0 * dm2 + b * env + comp


def viscous_weights(mu, dt):
    """(a, b) of psi_incremental: a = mu/(mu + dt), b = dt/(mu + dt)."""
    return (0.0, 1.0) if mu == 0.0 else (mu / (mu + dt), dt / (mu + dt))


class CohesiveInterface:
    """Chain of cohesive segments between matched node pairs.

    Args:
        coords: (n_nodes, 2) reference coordinates (mm).
        bot, top: (n + 1,) node indices of the lower / upper face, paired and ordered along the bond.
        width: out-of-plane thickness W (mm); energies are totals (N*mm).
        quadrature: "nodal" (2-point Newton-Cotes, what COH2D4 uses: S2 probe [D]) or "gauss".
        frame: "global" (S2 default), "corotational" (COH2D4 + NLGEOM), "corotational_linear"
            (COH2D4, NLGEOM = NO) or "midsurface" (see the module docstring).
    """

    def __init__(self, coords, bot, top, width=1.0, quadrature="nodal", frame="global"):
        self.coords = np.asarray(coords, dtype=float)
        self.bot, self.top = np.asarray(bot), np.asarray(top)
        self.width, self.frame = float(width), frame
        xi, w = QUADRATURES[quadrature]
        self.xi, self.w = jnp.asarray(xi), jnp.asarray(w)
        self.n_seg = len(self.bot) - 1
        self.n_q = len(xi)
        Xm = 0.5 * (self.coords[self.bot] + self.coords[self.top])          # reference mid-surface
        seg = Xm[1:] - Xm[:-1]
        self.L = np.linalg.norm(seg, axis=1)                                 # reference length
        t = seg / self.L[:, None]
        self.t0, self.n0 = jnp.asarray(t), jnp.asarray(np.c_[-t[:, 1], t[:, 0]])
        self.N = jnp.stack([0.5 * (1 - self.xi), 0.5 * (1 + self.xi)], axis=1)   # (q, 2)
        G = self.coords[self.top] - self.coords[self.bot]                   # reference gap per pair
        self.Gq = jnp.asarray(np.einsum("qa,sai->sqi", np.asarray(self.N), np.stack([G[:-1], G[1:]], axis=1)))
        b, tp = self.bot, self.top
        nodes = np.stack([b[:-1], b[1:], tp[1:], tp[:-1]], axis=1)          # COH2D4 order
        self.elem_dofs = (2 * nodes[:, :, None] + np.arange(2)).reshape(self.n_seg, 8)

    def zero_history(self):
        z = jnp.zeros((self.n_seg, self.n_q))
        return dict(dmax=z, D=z, Dv=z)

    def separations(self, u):
        """(delta_n, delta_s) at the quadrature points, each (n_seg, n_q).  u: (n_nodes, 2)."""
        jump = u[self.top] - u[self.bot]                                     # (n+1, 2)
        jq = jnp.einsum("qa,sai->sqi", self.N, jnp.stack([jump[:-1], jump[1:]], axis=1))
        t0, n0 = self.t0[:, None, :], self.n0[:, None, :]
        if self.frame == "global":
            return jnp.sum(jq * n0, -1), jnp.sum(jq * t0, -1)
        Gq = self.Gq                                                        # (s, q, 2) reference gap
        if self.frame == "corotational_linear":
            um = 0.5 * (u[self.bot] + u[self.top])
            omega = jnp.sum((um[1:] - um[:-1]) * self.n0, -1) / jnp.asarray(self.L)       # (s,)
            JG = jnp.stack([-Gq[..., 1], Gq[..., 0]], axis=-1)
            v = jq - omega[:, None, None] * JG
            return jnp.sum(v * n0, -1), jnp.sum(v * t0, -1)
        x = jnp.asarray(self.coords) + u
        xm = 0.5 * (x[self.bot] + x[self.top])
        seg = xm[1:] - xm[:-1]
        t = seg / jnp.linalg.norm(seg, axis=1, keepdims=True)
        n = jnp.stack([-t[:, 1], t[:, 0]], axis=1)
        t, n = t[:, None, :], n[:, None, :]
        if self.frame == "midsurface":
            return jnp.sum(jq * n, -1), jnp.sum(jq * t, -1)
        if self.frame != "corotational":
            raise ValueError("unknown frame %r" % self.frame)
        d = Gq + jq                                                         # current gap vector
        return jnp.sum(d * n, -1) - jnp.sum(Gq * n0, -1), jnp.sum(d * t, -1) - jnp.sum(Gq * t0, -1)

    def energy(self, u, Dstiff, K0):
        """Interface energy (N*mm) with the stiffness damage Dstiff (n_seg, n_q) frozen."""
        dn, ds = self.separations(u)
        dens = psi(dn, ds, Dstiff, K0)
        return self.width * jnp.sum(dens * self.w[None, :] * (0.5 * jnp.asarray(self.L))[:, None])

    def energy_incremental(self, u, dmax_n, Dv_n, law, a, b, K0):
        """Interface incremental potential (N*mm) at frozen history (dmax_n, Dv_n), psi_incremental."""
        dn, ds = self.separations(u)
        dens = psi_incremental(dn, ds, dmax_n, Dv_n, law, a, b, K0)
        return self.width * jnp.sum(dens * self.w[None, :] * (0.5 * jnp.asarray(self.L))[:, None])

    def update_history(self, u, hist, law, mu=0.0, dt=1.0):
        """New history after a converged (or trial) displacement u."""
        dn, ds = self.separations(u)
        dmax, D, Dv = update_damage(effective_separation(dn, ds), hist["dmax"], hist["Dv"], law, mu, dt)
        return dict(dmax=dmax, D=D, Dv=Dv)

    def tractions(self, u, hist, K0):
        dn, ds = self.separations(u)
        return traction(dn, ds, hist["Dv"], K0)
