"""Observation operator: the load averaged over fixed delta bins (D29).  Step S5b/S5c.

Units: mm, N.

The observed quantity is not P at the increments of a run but its average over fixed bins of the grip
separation delta,

    y_k = (1/w) int_{e_k}^{e_k + w} P^(delta) d delta,     e_k = k w,  k = 0 .. K-1,

where P^ is the piecewise-linear interpolant of the run's record (delta_n, P_n), n = 0 .. N (delta_0 = 0,
P_0 = 0 included).  y = O P with O (K x (N+1)) the exact integrals of the hat functions over the bins:
linear in P and independent of theta under displacement control (the delta_n are prescribed), so the
same O maps the sensitivities, dy/dtheta = O dP/dtheta, and data on another delta grid are observed by
their own O (same bins).

Bin width (D29): one bond-element passage, crack advance h_coh = 0.1 mm, converted with the tip-advance
rate dx_tip/d delta of the crack-tip record (energy_checks.zone_fronts).  For sw_s40 (ABAQUS and
tslid.fe identical to 3 digits) dx_tip/d delta falls from 58 (just after the peak) to 27 mm/mm (end of
the window); the linear fit over 0.3 mm <= delta <= d_end gives 33.5 mm/mm, i.e. w = 0.1 / 33.5 =
0.00298 mm.  DELTA_BIN = 0.003 mm (local passage 0.0017 - 0.0038 mm).  Documented in SOLVER_LOG S5b.

Windows: "full" = bins inside [0, d_end] (compare_ld window); "prepeak" = bins with upper edge <=
delta_peak of the reference record.
"""
import numpy as np

DELTA_BIN = 0.003          # mm, D29: one 0.1 mm bond-element passage of sw_s40 (see module docstring)


def bin_edges(d_end, width=DELTA_BIN, d_hi=None):
    """Edges 0, w, 2w, ... of the complete bins inside [0, d_end] (and <= d_hi if given)."""
    top = d_end if d_hi is None else min(d_end, d_hi)
    K = int(np.floor(top / width + 1e-9))
    return width * np.arange(K + 1)


def observation_matrix(delta, edges):
    """O (K x len(delta)): y = O P is the average over [edges[k], edges[k+1]] of the piecewise-linear
    interpolant of (delta, P).  delta must be increasing and cover the bins."""
    delta = np.asarray(delta, dtype=float)
    if np.any(np.diff(delta) <= 0):
        raise ValueError("delta must be strictly increasing")
    if edges[0] < delta[0] - 1e-12 or edges[-1] > delta[-1] + 1e-12:
        raise ValueError("bins outside the record")
    K, N = len(edges) - 1, len(delta)
    O = np.zeros((K, N))
    for k in range(K):
        a, b = edges[k], edges[k + 1]
        i0 = max(int(np.searchsorted(delta, a, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(delta, b, side="left")), N - 1)
        for i in range(i0, i1):                       # segment [delta_i, delta_i+1]
            x0, x1 = delta[i], delta[i + 1]
            lo, hi = max(a, x0), min(b, x1)
            if hi <= lo:
                continue
            L = x1 - x0
            # int_lo^hi of the hats: phi_i = (x1 - s)/L, phi_i+1 = (s - x0)/L
            O[k, i] += ((x1 - lo) ** 2 - (x1 - hi) ** 2) / (2 * L)
            O[k, i + 1] += ((hi - x0) ** 2 - (lo - x0) ** 2) / (2 * L)
        O[k] /= (b - a)
    return O


def observe(delta, values, edges):
    """Bin averages of a record: values (N,) or (N, p) -> (K,) or (K, p)."""
    return observation_matrix(delta, edges) @ np.asarray(values)


def windows(d_end, delta_peak_ref, width=DELTA_BIN):
    """{"full": edges, "prepeak": edges} (D29)."""
    return dict(full=bin_edges(d_end, width), prepeak=bin_edges(d_end, width, d_hi=delta_peak_ref))
