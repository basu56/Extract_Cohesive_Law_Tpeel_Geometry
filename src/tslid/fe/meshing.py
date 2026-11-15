"""Structured quadrilateral meshes.  Step S1 (cantilever, patch); the T-peel mesh comes in S3.

Units: mm, N, MPa, N/mm.

Through-thickness grading reuses ``graded``/``uniform`` from ``tslid.build_tpeel_inp`` so the
JAX meshes and the ABAQUS decks place nodes identically (that module is stdlib-only).
Node numbering of ``beam``: node (i, j) -> i * (ny + 1) + j, i along x, j along y; the ABAQUS
cantilever builder (scripts/fe/build_cantilever_inp.py) uses the same numbering plus one.
"""
import numpy as np

from tslid.build_tpeel_inp import graded, uniform


def beam(L, H, hx, ny, h_y0=None, skew=0.0):
    """Rectangular beam [0, L] x [0, H] with Q4 elements.

    Args:
        L, H: length and thickness, mm.
        hx: element length along x, mm (rounded to fit L).
        ny: elements through the thickness.
        h_y0: first element height at y = 0 (geometric grading as in the T-peel builder);
            None for uniform spacing.
        skew: optional distortion, mm: interior nodes are shifted in x by +skew / -skew on
            alternate rows (used for the distorted-mesh check).
    Returns:
        coords (n, 2), elems (n_el, 4) counter-clockwise, and a dict of node index sets:
        ``left``, ``right``, ``tip_mid`` (right-edge node nearest y = H/2), ``right_y`` (y of
        the right-edge nodes, increasing).
    """
    xs = np.array(uniform(0.0, L, hx))
    ys = np.array(graded(H, ny, h_y0) if h_y0 is not None else uniform(0.0, H, H / ny))
    nx = len(xs) - 1
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if skew:
        sgn = np.where(np.arange(ny + 1) % 2 == 0, 1.0, -1.0)
        interior = np.zeros_like(X, dtype=bool)
        interior[1:-1, :] = True
        X = X + interior * skew * sgn[None, :]
    coords = np.c_[X.ravel(), Y.ravel()]
    nid = lambda i, j: i * (ny + 1) + j
    elems = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for i in range(nx) for j in range(ny)])
    right = np.array([nid(nx, j) for j in range(ny + 1)])
    sets = dict(left=np.array([nid(0, j) for j in range(ny + 1)]), right=right,
                tip_mid=right[np.argmin(np.abs(ys - 0.5 * H))], right_y=ys)
    return coords, elems, sets


def edge_load(ys, total):
    """Consistent nodal forces of a uniform traction along an edge with node positions ys,
    summing to ``total`` (N).  Linear elements: each segment shares its load equally."""
    ys = np.asarray(ys)
    seg = np.diff(ys) / (ys[-1] - ys[0]) * total
    f = np.zeros(len(ys))
    f[:-1] += 0.5 * seg
    f[1:] += 0.5 * seg
    return f


def patch5():
    """Classic 5-element distorted patch in the unit square (MacNeal-Harder type).
    Returns coords (8, 2), elems (5, 4), boundary node indices (the 4 corners)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],          # corners
                       [0.25, 0.2], [0.75, 0.25], [0.8, 0.7], [0.3, 0.75]])     # distorted interior quad
    elems = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
    return coords, elems, np.array([0, 1, 2, 3])
