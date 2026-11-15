"""JAX version of the alpha shape family of cohesive laws (mirrors tslid.laws).  Step S2.

Units: mm, N, MPa.  K0 [MPa/mm], sigc [MPa], Gc [N/mm], separations [mm].

    d0 = sigc / K0,   df = d0 + (Gc - sigc d0 / 2) / (sigc g(alpha))
    sigma(d) = K0 d                      (d <= d0)
             = sigc s(x; alpha),  x = (d - d0)/(df - d0)   (d0 < d < df)
             = 0                         (d >= df)
    s(x; a) = expm1(a (1 - x)) / expm1(a),   g(a) = 1/a - 1/expm1(a) = int_0^1 s dx
    D(d)    = 1 - sigma(d) / (K0 d)      (ABAQUS damage variable, secant law)

Every parameter (K0, sigc, Gc, alpha) is a differentiable input.  Gradient safety
(docs/solver/S0_S2_INSTRUCTIONS.md):
- Near alpha = 0 (|alpha| < ALPHA_SERIES) s and g are evaluated as ratios of truncated power
  series in which the 0/0 cancels analytically:
      s = [sum_{k=1}^{m} y^k a^(k-1)/k!] / [sum_{k=1}^{m} a^(k-1)/k!],   y = 1 - x
      g = [sum_{k=2}^{m} a^(k-2)/k!]     / [sum_{k=1}^{m} a^(k-1)/k!]
  (exact up to O(a^m / (m+1)!) ~ 1e-50 at m = 16, |a| < 1e-3), smooth in a through 0.
- Branches use the double-where pattern: the unused branch is evaluated at a safe argument,
  so its derivative is finite and is then discarded.
"""
import math

import jax.numpy as jnp

ALPHA_SERIES = 1e-3
_M = 16
_FACT = [math.factorial(k) for k in range(_M + 2)]


def _series(a, coeffs):
    """sum_j coeffs[j] a^j (Horner)."""
    out = jnp.zeros_like(a) + coeffs[-1]
    for c in coeffs[-2::-1]:
        out = out * a + c
    return out


def _den(a):
    return _series(a, [1.0 / _FACT[k] for k in range(1, _M + 1)])          # expm1(a)/a


def g_alpha(alpha):
    """int_0^1 s(x; alpha) dx;  g(0) = 1/2."""
    alpha = jnp.asarray(alpha, dtype=float)
    small = jnp.abs(alpha) < ALPHA_SERIES
    a_big = jnp.where(small, 1.0, alpha)
    a_sml = jnp.where(small, alpha, 0.0)
    g_big = 1.0 / a_big - 1.0 / jnp.expm1(a_big)
    g_sml = _series(a_sml, [1.0 / _FACT[k] for k in range(2, _M + 2)]) / _den(a_sml)
    return jnp.where(small, g_sml, g_big)


def s_shape(x, alpha):
    """Normalised softening shape s(x; alpha), clipped to x in [0, 1]: s(0) = 1, s(1) = 0."""
    alpha = jnp.asarray(alpha, dtype=float)
    y = 1.0 - jnp.clip(x, 0.0, 1.0)
    small = jnp.abs(alpha) < ALPHA_SERIES
    a_big = jnp.where(small, 1.0, alpha)
    a_sml = jnp.where(small, alpha, 0.0)
    s_big = jnp.expm1(a_big * y) / jnp.expm1(a_big)
    # numerator sum_{k=1}^{m} y^k a^(k-1)/k!  = y * sum_{j=0}^{m-1} (a y)^j / (j+1)!
    num = y * _series(a_sml * y, [1.0 / _FACT[k] for k in range(1, _M + 1)])
    s_sml = num / _den(a_sml)
    return jnp.where(small, s_sml, s_big)


def law_params(K0, sigc, Gc, alpha):
    """(d0, df) in mm.  Admissibility (Gc > sigc^2 / (2 K0)) is the caller's job (tslid.laws)."""
    d0 = sigc / K0
    df = d0 + (Gc - 0.5 * sigc * d0) / (sigc * g_alpha(alpha))
    return d0, df


def traction(d, K0, sigc, Gc, alpha):
    """Monotonic-loading traction sigma(d) [MPa] for an opening d >= 0 (0 for d <= 0)."""
    d0, df = law_params(K0, sigc, Gc, alpha)
    x = (d - d0) / (df - d0)
    soft = sigc * s_shape(x, alpha)
    out = jnp.where(d <= d0, K0 * d, jnp.where(d >= df, 0.0, soft))
    return jnp.where(d <= 0.0, 0.0, out)


def damage(dmax, K0, sigc, Gc, alpha):
    """ABAQUS damage variable D(d_max) = 1 - sigma(d_max)/(K0 d_max); 0 up to d0, 1 from df."""
    d0, df = law_params(K0, sigc, Gc, alpha)
    soft = (dmax > d0) & (dmax < df)
    d_safe = jnp.where(soft, dmax, df)                        # safe denominator, inside (d0, df]
    x = (d_safe - d0) / (df - d0)
    D_soft = 1.0 - sigc * s_shape(x, alpha) / (K0 * d_safe)
    return jnp.where(dmax <= d0, 0.0, jnp.where(dmax >= df, 1.0, D_soft))
