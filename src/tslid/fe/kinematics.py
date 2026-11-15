"""Bulk strain-energy densities for plane strain.  Step S1.

Units: mm, N, MPa, N/mm.  Energies are per unit reference area (N*mm/mm^2 = MPa) and per unit
out-of-plane thickness; multiply the integral by the thickness W to get N*mm.

All functions take the displacement gradient H = grad_X u (shape (..., 2, 2), in-plane
components; the out-of-plane displacement is zero, plane strain) and return the energy density.

- ``psi_linear``: small-strain isotropic linear elasticity,
      eps = sym(H),  psi = lam/2 tr(eps)^2 + mu eps:eps.
- ``psi_svk``: finite-rotation St. Venant-Kirchhoff, total Lagrangian,
      F = I + H (F33 = 1),  E = (F^T F - I)/2 (E_i3 = 0),  psi = lam/2 tr(E)^2 + mu E:E.
  For small strains it reduces to ``psi_linear``; it is frame-invariant, so it captures large
  rotations exactly while its stress-strain law differs from ABAQUS's hypoelastic NLGEOM
  update only at second order in strain [H, quantified in S1 test 4].
"""
import jax.numpy as jnp


def lame(E, nu):
    """Lame constants (lam, mu) in MPa from Young's modulus E (MPa) and Poisson ratio nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _iso(eps, lam, mu):
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    return 0.5 * lam * tr ** 2 + mu * jnp.sum(eps * eps, axis=(-2, -1))


def psi_linear(H, lam, mu):
    """Small-strain plane-strain energy density (MPa) from the displacement gradient H."""
    eps = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    return _iso(eps, lam, mu)


def psi_svk(H, lam, mu):
    """St. Venant-Kirchhoff plane-strain energy density (MPa) from the displacement gradient H.

    Uses E = (H + H^T + H^T H)/2, which equals (F^T F - I)/2 without the cancellation of
    forming F^T F - I in floating point.
    """
    Ht = jnp.swapaxes(H, -1, -2)
    E = 0.5 * (H + Ht + Ht @ H)
    return _iso(E, lam, mu)


ENERGIES = {"linear": psi_linear, "svk": psi_svk}
