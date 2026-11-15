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


def psi_hencky(H, lam, mu):
    """Hencky (logarithmic-strain) plane-strain energy density (MPa), closed form.  Step S3a.

    psi = lam/2 tr(h)^2 + mu h:h, h = 1/2 ln(C) (in-plane 2x2 block; C33 = 1 so h33 = 0).
    For the symmetric positive-definite 2x2 C with eigenvalues m +/- s,
        ln C = a I + b (C - m I),  a = 1/2 ln det C,  b = atanh(s/m)/s = (1/m) f(z2),
        z2 = (s/m)^2 = [((C11 - C22)/2)^2 + C12^2] / m^2,  f(z2) = atanh(sqrt z2)/sqrt z2,
    and f is evaluated by its series 1 + z2/3 + z2^2/5 + ... for z2 < 1e-4 (double-where), so the
    energy depends on C only through (det C, m, z2): smooth and NaN-free at H = 0 and at equal
    eigenvalues, and valid for any strain (the series version below diverges once |C - I| ~ 1,
    which the first Newton iterate after a prescribed jump can reach -- S3a).
    """
    Ht = jnp.swapaxes(H, -1, -2)
    A = H + Ht + Ht @ H                                     # C - I (no cancellation)
    c11, c22, c12 = 1.0 + A[..., 0, 0], 1.0 + A[..., 1, 1], A[..., 0, 1]
    m = 0.5 * (c11 + c22)
    p = (0.5 * (A[..., 0, 0] - A[..., 1, 1])) ** 2 + c12 ** 2
    z2 = p / m ** 2
    small = z2 < 1e-4
    z2s = jnp.where(small, 0.5, z2)
    z = jnp.sqrt(z2s)
    f_big = jnp.arctanh(z) / z
    zz = jnp.where(small, z2, 0.0)
    f_sml = 1.0 + zz * (1.0 / 3 + zz * (1.0 / 5 + zz * (1.0 / 7 + zz * (1.0 / 9 + zz / 11))))
    b = jnp.where(small, f_sml, f_big) / m
    detC = c11 * c22 - c12 ** 2
    a = 0.5 * jnp.log(detC)
    # tr(ln C) = 2a; deviatoric part b (C - m I) has entries b*(A11-A22)/2, b*c12
    d = 0.5 * (A[..., 0, 0] - A[..., 1, 1])
    h11 = 0.5 * (a + b * d)
    h22 = 0.5 * (a - b * d)
    h12 = 0.5 * b * c12
    tr = h11 + h22
    return 0.5 * lam * tr ** 2 + mu * (h11 ** 2 + h22 ** 2 + 2.0 * h12 ** 2)


def psi_hencky_series(H, lam, mu, order=10):
    """Hencky (logarithmic-strain) plane-strain energy density (MPa), power-series ln C (S1).

    psi = lam/2 tr(h)^2 + mu h:h with h = ln(U) = 1/2 ln(C), C = F^T F.  Its Kirchhoff stress
    is linear in log strain, the closest hyperelastic counterpart of ABAQUS's hypoelastic NLGEOM
    update of ``*Elastic`` [I].  ln(C) = ln(I + A), A = C - I, by its power series truncated at
    ``order`` terms: no eigen-decomposition, so it is smooth and NaN-free at H = 0.  Truncation
    error ~ |A|^(order+1)/(order+1): < 1e-13 relative for strains up to ~5 %; not for large strain.
    """
    Ht = jnp.swapaxes(H, -1, -2)
    A = H + Ht + Ht @ H                                     # C - I
    term, lnC = A, A
    for k in range(2, order + 1):
        term = term @ A
        lnC = lnC + ((-1) ** (k + 1) / k) * term
    return _iso(0.5 * lnC, lam, mu)


ENERGIES = {"linear": psi_linear, "svk": psi_svk, "hencky": psi_hencky, "hencky_series": psi_hencky_series}
