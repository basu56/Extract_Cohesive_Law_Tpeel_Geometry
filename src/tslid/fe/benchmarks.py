"""Verification problems shared by tests and comparison scripts.  Step S1: the cantilever.

Units: mm, N, MPa, N/mm.

``cantilever`` mirrors scripts/fe/build_cantilever_inp.py exactly (same mesh, nodal loads,
boundary conditions), so its output can be compared node for node with the ABAQUS decks.
"""
import time

import jax.numpy as jnp
import numpy as np

from .elements import BulkModel
from .meshing import beam, edge_load
from .solve import SparseNewton

AL = dict(E=70000.0, nu=0.33)              # adherend Al, MPa (D11)
CANT = dict(L=60.0, H=6.324, W=15.0)       # mm


def timoshenko_tip(P, L=CANT["L"], H=CANT["H"], W=CANT["W"], E=AL["E"], nu=AL["nu"], kappa=5.0 / 6.0):
    """Tip deflection (mm) of a plane-strain Timoshenko cantilever under total end load P (N)."""
    Ep, G, I = E / (1.0 - nu ** 2), E / (2.0 * (1.0 + nu)), H ** 3 / 12.0
    return P / W * L ** 3 / (3.0 * Ep * I) + P / W * L / (kappa * G * H)


def cantilever(ny, kind="Q4IM", kinematics="linear", load="force", P=230.0, v=0.67,
               hx=0.1, h_y0=0.1, skew=0.0, n_inc=1, rtol=1e-10, verbose=False):
    """Clamped-free plane-strain cantilever (left edge clamped, right edge loaded).

    Args:
        load: "force" (uniform shear traction, total P, dead load) or "disp" (all right-edge
            nodes U2 = v, U1 free).
        n_inc: equal load increments (force control); results are reported after each.
        h_y0: first element height at y = 0 (T-peel grading); None for uniform.
    Returns:
        dict with per-increment arrays ``lam``, ``u_tip``, ``v_tip`` (right-edge node nearest
        y = H/2), ``theta`` (rotation of the right edge, rad, from its end nodes),
        ``R_tip`` (sum of reactions on the right edge, y; disp control), ``M_clamp``
        (clamp reaction moment about the origin, N*mm), plus ``U`` (last state, nodal
        (n_nodes, 2)), the model, Newton statistics and wall times.
    """
    coords, elems, s = beam(CANT["L"], CANT["H"], hx, ny, h_y0=h_y0, skew=skew)
    m = BulkModel(coords, elems, AL["E"], AL["nu"], CANT["W"], kind, kinematics)
    left, right = s["left"], s["right"]
    fixed = np.r_[2 * left, 2 * left + 1] if load == "force" else np.r_[2 * left, 2 * left + 1, 2 * right + 1]
    t0 = time.perf_counter()
    solver = SparseNewton(m.energy, m.n_dof, m.elem_dofs, fixed)
    f1 = np.zeros(m.n_dof)
    if load == "force":
        f1[2 * right + 1] = edge_load(s["right_y"], P)
    U = np.zeros(m.n_dof)
    out = {k: [] for k in ("lam", "u_tip", "v_tip", "theta", "R_tip", "M_clamp", "iters", "t_solve")}
    for k in range(1, n_inc + 1):
        lam = k / n_inc
        if load == "disp":
            U[2 * right + 1] = lam * v
        res = solver.solve(U, lam * f1, rtol=rtol, verbose=verbose)
        if not res.converged:
            raise RuntimeError(f"Newton did not converge at increment {k}: {res.residual_history}")
        U = res.U
        r = solver.residual(U, lam * f1)                 # reactions live on fixed dofs
        u = U[: 2 * m.n_nodes].reshape(-1, 2)
        top, bot = right[-1], right[0]
        d = (coords[top] + u[top]) - (coords[bot] + u[bot])
        Y = coords[left, 1] + u[left, 1]
        X = coords[left, 0] + u[left, 0]
        out["lam"].append(lam)
        out["u_tip"].append(u[s["tip_mid"], 0])
        out["v_tip"].append(u[s["tip_mid"], 1])
        out["theta"].append(-np.arctan2(d[0], d[1]))
        out["R_tip"].append(r[2 * right + 1].sum() if load == "disp" else np.nan)
        out["M_clamp"].append(np.sum(X * r[2 * left + 1] - Y * r[2 * left]))
        out["iters"].append(res.iterations)
        out["t_solve"].append(res.wall_time)
    out = {k: np.array(v) for k, v in out.items()}
    out.update(U=U[: 2 * m.n_nodes].reshape(-1, 2), model=m, sets=s, n_dof=m.n_dof,
               n_colors=solver.n_colors, t_total=time.perf_counter() - t0)
    return out
