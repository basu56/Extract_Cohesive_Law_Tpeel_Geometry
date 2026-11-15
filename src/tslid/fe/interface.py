"""Cohesive interface: separations, energy with frozen damage, history update.  Step S2.

Units: mm, N, MPa, N/mm.  Formulation and ABAQUS correspondence: docs/solver/INTERFACE.md.

The bond line is a chain of matched node pairs: lower face nodes ``bot[k]`` and upper face nodes
``top[k]`` (k = 0..n), forming n segments (Line2 on each face; a COH2D4-like element per segment).
Constitutive thickness 1, so the separation equals the displacement jump [[u]] = u_top - u_bot
(the geometric gap, 0.4 mm in the T-peel, only sets the initial geometry).

Separations at a quadrature point: delta_n = [[u]] . n, delta_s = [[u]] . t, with (t, n) the
reference (global) frame of the segment (``frame="global"``, S2), or the deformed mid-surface
frame (``frame="midsurface"``; what COH2D4 does under NLGEOM [H]; to be tested in S3).

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
    """Analytic alpha-family law (ABAQUS LINEAR / EXPONENTIAL)."""
    K0: float
    sigc: float
    Gc: float
    alpha: float

    def damage(self, dmax):
        return damage_alpha(dmax, self.K0, self.sigc, self.Gc, self.alpha)


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


def update_damage(dm, dmax_old, Dv_old, law, mu=0.0, dt=1.0):
    """History update at quadrature points.  Returns (dmax, D, Dv): D inviscid, Dv the regularised
    damage used for the stiffness (Dv = D when mu = 0)."""
    dmax = jnp.maximum(dmax_old, dm)
    D = law.damage(dmax)
    Dv = D if mu == 0.0 else (dt * D + mu * Dv_old) / (mu + dt)
    return dmax, D, Dv


class CohesiveInterface:
    """Chain of cohesive segments between matched node pairs.

    Args:
        coords: (n_nodes, 2) reference coordinates (mm).
        bot, top: (n + 1,) node indices of the lower / upper face, paired and ordered along the bond.
        width: out-of-plane thickness W (mm); energies are totals (N*mm).
        quadrature: "gauss" (2-point) or "nodal" (2-point Newton-Cotes), see INTERFACE.md.
        frame: "global" (reference segment frame) or "midsurface" (deformed mid-surface frame).
    """

    def __init__(self, coords, bot, top, width=1.0, quadrature="gauss", frame="global"):
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
        if self.frame == "global":
            t, n = self.t0[:, None, :], self.n0[:, None, :]
        else:
            x = jnp.asarray(self.coords) + u
            xm = 0.5 * (x[self.bot] + x[self.top])
            seg = xm[1:] - xm[:-1]
            t = seg / jnp.linalg.norm(seg, axis=1, keepdims=True)
            n = jnp.stack([-t[:, 1], t[:, 0]], axis=1)
            t, n = t[:, None, :], n[:, None, :]
        return jnp.sum(jq * n, -1), jnp.sum(jq * t, -1)

    def energy(self, u, Dstiff, K0):
        """Interface energy (N*mm) with the stiffness damage Dstiff (n_seg, n_q) frozen."""
        dn, ds = self.separations(u)
        dens = psi(dn, ds, Dstiff, K0)
        return self.width * jnp.sum(dens * self.w[None, :] * (0.5 * jnp.asarray(self.L))[:, None])

    def update_history(self, u, hist, law, mu=0.0, dt=1.0):
        """New history after a converged (or trial) displacement u."""
        dn, ds = self.separations(u)
        dmax, D, Dv = update_damage(effective_separation(dn, ds), hist["dmax"], hist["Dv"], law, mu, dt)
        return dict(dmax=dmax, D=D, Dv=Dv)

    def tractions(self, u, hist, K0):
        dn, ds = self.separations(u)
        return traction(dn, ds, hist["Dv"], K0)
