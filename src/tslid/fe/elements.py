"""Plane-strain bulk elements on tatva's Quad4.  Step S1; formulation in docs/solver/ELEMENTS.md.

Units: mm, N, MPa, N/mm.

Two elements, both with 2x2 Gauss integration (tatva ``element.Quad4`` quadrature):

- ``"Q4"``   -- plain bilinear quadrilateral (tatva Quad4).  Baseline; shear-locks in bending.
- ``"Q4IM"`` -- Q4 enhanced with 4 incompatible modes per element (Wilson modes
  (1 - xi^2), (1 - eta^2) for each displacement component), Taylor's modification
  (gradients mapped with the element-centre Jacobian J0 and scaled by j0/j), applied
  additively to the displacement gradient: H = grad u_c + H_enh.  For finite rotation this
  is the Simo-Armero (1992) enhanced deformation gradient F = F_c + H_enh.
  The 4 amplitudes per element are extra global unknowns appended after the nodal dofs; each
  couples only to its own element, so the tangent stays sparse and the total energy is
  minimised jointly (no static condensation code; AD gives the tangent).

Unknown vector layout (``BulkModel``): ``U = [u_x0, u_y0, u_x1, u_y1, ..., a_0(4), a_1(4), ...]``,
i.e. 2 * n_nodes nodal dofs followed by 4 * n_elements mode amplitudes (Q4IM only).
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from tatva import Mesh, Operator, element

from .kinematics import ENERGIES, lame

_Q4 = element.Quad4()


def _centroid_geometry(coords, elems):
    """Per element: inverse centre Jacobian J0^{-1} (n_el, 2, 2), j0 = det J0 (n_el,) and
    det J at the quadrature points (n_el, n_q).  tatva convention: J_ij = dX_j/dxi_i."""
    X = np.asarray(coords)[np.asarray(elems)]                       # (n_el, 4, 2)
    dN0 = np.asarray(_Q4.shape_function_derivative(jnp.zeros(2)))   # (2, 4)
    J0 = np.einsum("in,enj->eij", dN0, X)
    jq = np.stack([np.linalg.det(np.einsum("in,enj->eij",
                                           np.asarray(_Q4.shape_function_derivative(q)), X))
                   for q in np.asarray(_Q4.quad_points)], axis=1)
    return np.linalg.inv(J0), np.linalg.det(J0), jq


def _mode_derivatives():
    """d(phi_k)/d(xi_i) at the quadrature points for phi_1 = 1 - xi^2, phi_2 = 1 - eta^2.
    Shape (n_q, 2 [i], 2 [k])."""
    out = []
    for xi, eta in np.asarray(_Q4.quad_points):
        out.append([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])
    return np.array(out)


@dataclass
class BulkModel:
    """Plane-strain elastic body discretised with Q4 or Q4IM elements.

    Args:
        coords: (n_nodes, 2) reference coordinates, mm.
        elems: (n_el, 4) node indices, counter-clockwise.
        E, nu: Young's modulus (MPa) and Poisson ratio.
        thickness: out-of-plane thickness W, mm (energies are total, N*mm).
        kind: "Q4" or "Q4IM".
        kinematics: "linear" (small strain) or "svk" (finite rotation St. Venant-Kirchhoff).
    """
    coords: np.ndarray
    elems: np.ndarray
    E: float
    nu: float
    thickness: float = 1.0
    kind: str = "Q4IM"
    kinematics: str = "linear"
    op: Operator = field(init=False, repr=False)

    def __post_init__(self):
        if self.kind not in ("Q4", "Q4IM"):
            raise ValueError("kind must be 'Q4' or 'Q4IM'")
        self.coords = np.asarray(self.coords, dtype=float)
        self.elems = np.asarray(self.elems, dtype=np.int64)
        self.op = Operator(Mesh(jnp.asarray(self.coords), jnp.asarray(self.elems)), _Q4)
        self.n_nodes, self.n_el = len(self.coords), len(self.elems)
        self.n_modes = 4 if self.kind == "Q4IM" else 0
        self.n_dof = 2 * self.n_nodes + self.n_modes * self.n_el
        self._psi = ENERGIES[self.kinematics]
        if self.n_modes:
            invJ0, j0, jq = _centroid_geometry(self.coords, self.elems)
            dphi = _mode_derivatives()                                    # (q, i, k)
            # g[e, q, k, i] = (j0/j_q) * sum_j invJ0[e, i, j] dphi[q, j, k]
            g = np.einsum("eij,qjk->eqki", invJ0, dphi) * (j0[:, None] / jq)[:, :, None, None]
            self._g = jnp.asarray(g)
        d = (2 * self.elems[:, :, None] + np.arange(2)).reshape(self.n_el, 8)
        if self.n_modes:
            d = np.hstack([d, 2 * self.n_nodes + 4 * np.arange(self.n_el)[:, None] + np.arange(4)])
        self.elem_dofs = d                                                # (n_el, 8 or 12)

    def split(self, U):
        """U -> (nodal displacements (n_nodes, 2), mode amplitudes (n_el, 2 modes, 2 comps))."""
        u = U[: 2 * self.n_nodes].reshape(-1, 2)
        a = U[2 * self.n_nodes:].reshape(-1, 2, 2) if self.n_modes else None
        return u, a

    def displacement_gradient(self, U):
        """H = grad_X u at the quadrature points, (n_el, n_q, 2, 2); H[..., a, i] = du_a/dX_i."""
        u, a = self.split(U)
        H = self.op.grad(u)
        if self.n_modes:
            H = H + jnp.einsum("eka,eqki->eqai", a, self._g)
        return H

    def energy(self, U, E=None, nu=None):
        """Total strain energy, N*mm.  E and nu may be passed as (differentiable) overrides."""
        lam, mu = lame(self.E if E is None else E, self.nu if nu is None else nu)
        return self.thickness * self.op.integrate(self._psi(self.displacement_gradient(U), lam, mu))
