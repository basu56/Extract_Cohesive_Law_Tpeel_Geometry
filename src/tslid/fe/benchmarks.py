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


# --------------------------------------------------------------------------------------------
# S2: single cohesive element (one COH2D4-like segment, 1 mm long, width 1) under a prescribed
# top-face displacement path.
# --------------------------------------------------------------------------------------------
def single_cohesive(law, un_path, us_path, times, mu=0.0, quadrature="nodal"):
    """Response of one cohesive segment whose top nodes follow (us_path, un_path) uniformly.

    Args:
        law: AlphaLawParams or TabularLawParams.
        un_path, us_path: (n+1,) prescribed normal / shear top displacement (mm), starting at 0.
        times: (n+1,) times (s); used only for viscous regularisation (mu > 0).
    Returns dict of (n+1,) arrays: tn, ts (tractions, MPa), D, Dv, dmax, F1, F2 (top nodal force
    sums, N), W (external work, N*mm, trapezoid), psi (stored energy), diss = W - psi.
    """
    import jax
    import jax.numpy as jnp
    from .interface import CohesiveInterface

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    itf = CohesiveInterface(coords, [0, 1], [3, 2], 1.0, quadrature)
    K0 = law.K0
    top = jnp.array([2, 3])

    def u_of(un, us):
        return jnp.zeros((4, 2)).at[top, 0].set(us).at[top, 1].set(un)

    def step(carry, inp):
        hist, W, F_old, u_old = carry
        un, us, dt = inp
        u = u_of(un, us)
        hist = itf.update_history(u, hist, law, mu, dt)
        F = jax.grad(itf.energy)(u, hist["Dv"], K0)[top].sum(axis=0)
        W = W + 0.5 * jnp.sum((F + F_old) * (u[2] - u_old[2]))
        tn, ts = itf.tractions(u, hist, K0)
        psi = itf.energy(u, hist["Dv"], K0)
        out = dict(tn=tn[0, 0], ts=ts[0, 0], D=hist["D"][0, 0], Dv=hist["Dv"][0, 0],
                   dmax=hist["dmax"][0, 0], F1=F[0], F2=F[1], W=W, psi=psi)
        return (hist, W, F, u), out

    un_path, us_path, times = (jnp.asarray(np.asarray(a, dtype=float)) for a in (un_path, us_path, times))
    dts = jnp.diff(times)
    init = (itf.zero_history(), 0.0, jnp.zeros(2), u_of(un_path[0], us_path[0]))
    _, out = jax.lax.scan(step, init, (un_path[1:], us_path[1:], dts))
    out = {k: np.r_[0.0, np.asarray(v)] for k, v in out.items()}
    out["diss"] = out["W"] - out["psi"]
    return out


def two_block_bar(law, deltas, kind="Q4IM", gap=0.4, W=15.0, max_stagger=50):
    """S2 test 6: two 1 x 1 mm Q4/Q4IM blocks joined by one cohesive segment (gap ``gap``),
    bottom face fixed in y, top face pulled to y-displacement Delta (each value of ``deltas``),
    with one node fixed in x per block (lateral contraction free).  Staggered history:
    solve with damage frozen, update history, repeat until it no longer changes.

    Returns dict of arrays per Delta: R (top reaction, N), delta_n (interface opening),
    t_n (interface traction), D, n_stagger.
    """
    import jax.numpy as jnp
    from .elements import BulkModel
    from .interface import CohesiveInterface

    y0 = [0.0, 1.0, 1.0 + gap, 2.0 + gap]
    coords = np.array([[0.0, y0[0]], [1.0, y0[0]], [1.0, y0[1]], [0.0, y0[1]],
                       [0.0, y0[2]], [1.0, y0[2]], [1.0, y0[3]], [0.0, y0[3]]])
    elems = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    bulk = BulkModel(coords, elems, AL["E"], AL["nu"], W, kind, "linear")
    itf = CohesiveInterface(coords, [3, 2], [4, 5], W)
    nn = len(coords)

    def energy(U, Dv):
        return bulk.energy(U) + itf.energy(U[: 2 * nn].reshape(-1, 2), Dv, law.K0)

    fixed = np.array([2 * 0, 2 * 0 + 1, 2 * 1 + 1, 2 * 7, 2 * 6 + 1, 2 * 7 + 1])
    top_y = np.array([2 * 6 + 1, 2 * 7 + 1])
    solver = SparseNewton(energy, bulk.n_dof, [bulk.elem_dofs, itf.elem_dofs], fixed)
    U = np.zeros(bulk.n_dof)
    hist = itf.zero_history()
    out = {k: [] for k in ("R", "delta_n", "t_n", "D", "n_stagger")}
    for Dl in deltas:
        U[top_y] = Dl
        for it in range(max_stagger):
            res = solver.solve(U, None, hist["Dv"], rtol=1e-12, atol=1e-14)
            assert res.converged, res.residual_history
            U = res.U
            new = itf.update_history(jnp.asarray(U[: 2 * nn].reshape(-1, 2)), hist, law)
            change = float(jnp.max(jnp.abs(new["D"] - hist["D"])))
            hist = new
            if change < 1e-14:
                break
        r = solver.residual(U, np.zeros(bulk.n_dof), hist["Dv"])
        dn, _ = itf.separations(jnp.asarray(U[: 2 * nn].reshape(-1, 2)))
        tn, _ = itf.tractions(jnp.asarray(U[: 2 * nn].reshape(-1, 2)), hist, law.K0)
        out["R"].append(r[top_y].sum())
        out["delta_n"].append(float(dn.mean()))
        out["t_n"].append(float(tn.mean()))
        out["D"].append(float(hist["D"].mean()))
        out["n_stagger"].append(it + 1)
    out = {k: np.array(v) for k, v in out.items()}
    out["area"] = 1.0 * W
    return out


def se_case(meta):
    """Rebuild an ABAQUS single-COH2D4 run (Phase 0 src/tslid/single_element.py or S2
    scripts/fe/build_se_deck.py) from its JSON metadata.

    Returns (law, times, un, us, mu): the law object (AlphaLawParams or TabularLawParams, matching
    the ABAQUS representation), increment times (s), prescribed top normal / shear displacement
    (mm) at every increment (*Static, direct: t_k = k T / n_inc), and the viscosity (s).
    """
    from tslid.laws import AlphaLaw
    from .interface import AlphaLawParams, TabularLawParams

    p = meta["params"]
    K = p.get("K", 8500.0)
    lw = AlphaLaw(K, p["sigc"], p["Gc"], p["alpha"])
    soft = meta["representation"]["softening"]
    law = (TabularLawParams.from_alpha_law(lw, p.get("n_tab", 1000)) if soft == "tabular"
           else AlphaLawParams(K, p["sigc"], p["Gc"], p["alpha"]))
    n = meta.get("n_inc", p.get("n_inc"))
    T = meta.get("step_time", 1.0)
    t = np.arange(n + 1) / n
    if "U_top" in meta:                                   # S2 decks: linear ramp of (U1, U2)
        us_max, un_max = meta["U_top"]["3"]
        amp = t
        un, us = un_max * amp, us_max * amp
    else:                                                 # Phase 0 decks: mode I, optional cycle
        dmax = meta["dmax"]
        if meta.get("d_unload") is not None:
            du = meta["d_unload"]
            t1, t2 = du / (2 * du + dmax), 2 * du / (2 * du + dmax)
            amp = np.interp(t, [0.0, t1, t2, 1.0], [0.0, du / dmax, 0.0, 1.0])
        else:
            amp = t
        un, us = dmax * amp, 0.0 * amp
    return law, t * T, un, us, p.get("viscosity", 0.0)
