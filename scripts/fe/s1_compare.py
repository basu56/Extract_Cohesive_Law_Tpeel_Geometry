"""s1_compare.py -- S1 tests 2-5: tslid.fe cantilevers vs ABAQUS CPE4I and Timoshenko; timing.

Units: mm, N, MPa.  Run in the uv env on a compute node (scripts/fe/run_s1_compare.sh):
    python scripts/fe/s1_compare.py <abaqus rundir> <outdir=results/solver/S1_bulk>

Reads <rundir>/<job>_nodes.csv (scripts/fe/extract_cantilever_odb.py) and
<rundir>/tpeel_sw_s40_hist_2000001.csv.  Writes to outdir:
    abaqus_reference.csv   ABAQUS tip quantities per case and frame (used by tests/fe)
    s1_linear.csv          test 2-3: Q4 / Q4IM vs ABAQUS (and Timoshenko), ny = 2, 4, 8, 20, skew
    s1_nlgeom.csv          test 4: SVK Q4IM vs ABAQUS NLGEOM per increment
    s1_timing.csv          test 5: wall time per Newton solve
    tpeel_rotation.csv     T-peel load-point rotation UR3 vs delta (sw_s40)
    s1_bulk.png/.pdf       figure
"""
import csv
import os
import platform
import sys
import time

import numpy as np

import tslid.fe  # noqa: F401  (x64)
from tslid.fe.benchmarks import CANT, cantilever, timoshenko_tip
from tslid.fe.meshing import beam

RUN, OUT = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "results/solver/S1_bulk")
os.makedirs(OUT, exist_ok=True)

CASES = {  # job -> (ny, mesh kwargs, load kwargs)
    **{f"lin_ny{n}_F": (n, dict(hx=0.1, h_y0=0.1), dict(load="force", P=230.0)) for n in (2, 4, 8, 20)},
    **{f"lin_ny{n}_D": (n, dict(hx=0.1, h_y0=0.1), dict(load="disp", v=0.67)) for n in (2, 4, 8, 20)},
    "skew_ny4_F": (4, dict(hx=1.0, h_y0=None, skew=0.3), dict(load="force", P=230.0)),
}
NL = ("nl_ny20_F", 20, dict(hx=0.1, h_y0=0.1), dict(load="force", P=1400.0), 20)


def abaqus_tip(job, ny, mesh_kw):
    """Tip quantities per frame from the ABAQUS nodal CSV (node label = index + 1)."""
    coords, _, s = beam(CANT["L"], CANT["H"], mesh_kw["hx"], ny, h_y0=mesh_kw.get("h_y0"),
                        skew=mesh_kw.get("skew", 0.0))
    d = np.genfromtxt(os.path.join(RUN, job + "_nodes.csv"), delimiter=",", names=True)
    rows = []
    for fr in np.unique(d["frame"]).astype(int):
        if fr == 0:
            continue
        f = d[d["frame"] == fr]
        idx = f["node"].astype(int) - 1
        U = np.zeros((len(coords), 2)); RF = np.zeros((len(coords), 2))
        U[idx] = np.c_[f["U1"], f["U2"]]; RF[idx] = np.c_[f["RF1"], f["RF2"]]
        L, R = s["left"], s["right"]
        dv = (coords[R[-1]] + U[R[-1]]) - (coords[R[0]] + U[R[0]])
        rows.append(dict(case=job, frame=fr, lam=f["lam"][0], u_tip=U[s["tip_mid"], 0],
                         v_tip=U[s["tip_mid"], 1], theta=-np.arctan2(dv[0], dv[1]),
                         R_tip=RF[R, 1].sum(),
                         M_clamp=np.sum((coords[L, 0] + U[L, 0]) * RF[L, 1] - (coords[L, 1] + U[L, 1]) * RF[L, 0]),
                         U=U))
    return rows


def rel(a, b):
    return (a - b) / abs(b)


def write(name, rows, keys):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


ref_rows, lin_rows, timing = [], [], []
for job, (ny, mkw, lkw) in CASES.items():
    ab = abaqus_tip(job, ny, mkw)[-1]
    ref_rows.append(ab)
    for kind in ("Q4", "Q4IM"):
        t0 = time.perf_counter()
        r = cantilever(ny, kind, "linear", **mkw, **lkw)
        t1 = time.perf_counter()
        r2 = cantilever(ny, kind, "linear", **mkw, **lkw) if job.endswith("_F") and job.startswith("lin") else None
        fld = np.abs(r["U"] - ab["U"]).max() / np.abs(ab["U"]).max()
        row = dict(case=job, ny=ny, element=kind, load=lkw["load"], n_dof=r["n_dof"],
                   jax_v_tip=r["v_tip"][-1], abq_v_tip=ab["v_tip"], err_v_tip=rel(r["v_tip"][-1], ab["v_tip"]),
                   jax_R_tip=r["R_tip"][-1], abq_R_tip=ab["R_tip"],
                   err_R_tip=rel(r["R_tip"][-1], ab["R_tip"]) if lkw["load"] == "disp" else np.nan,
                   err_field=fld,
                   err_v_vs_timoshenko=rel(r["v_tip"][-1], timoshenko_tip(lkw["P"])) if lkw["load"] == "force" else np.nan,
                   abq_err_v_vs_timoshenko=rel(ab["v_tip"], timoshenko_tip(lkw["P"])) if lkw["load"] == "force" else np.nan)
        lin_rows.append(row)
        if r2 is not None:
            timing.append(dict(case=job, element=kind, kinematics="linear", n_dof=r["n_dof"], n_colors=r["n_colors"],
                               newton_iters=int(r2["iters"][-1]), t_first_call=t1 - t0,
                               t_solve_after_jit=float(r2["t_solve"][-1]), host=platform.node()))
        print(f"{job:12s} {kind:5s} v_err={row['err_v_tip']:+.2e} R_err={row['err_R_tip']:+.2e} field={fld:.2e}", flush=True)

job, ny, mkw, lkw, ninc = NL
ab = abaqus_tip(job, ny, mkw)
ref_rows += ab
t0 = time.perf_counter()
r = cantilever(ny, "Q4IM", "svk", **mkw, **lkw, n_inc=ninc)
timing.append(dict(case=job, element="Q4IM", kinematics="svk", n_dof=r["n_dof"], n_colors=r["n_colors"],
                   newton_iters=int(np.sum(r["iters"])), t_first_call=time.perf_counter() - t0,
                   t_solve_after_jit=float(np.median(r["t_solve"][1:])), host=platform.node()))
rl = cantilever(ny, "Q4IM", "linear", **mkw, **lkw, n_inc=1)
nl_rows = []
for k, a in enumerate(ab):
    assert abs(a["lam"] - r["lam"][k]) < 1e-6
    nl_rows.append(dict(frame=a["frame"], lam=a["lam"], P=a["lam"] * lkw["P"], theta_abq=a["theta"], theta_jax=r["theta"][k],
                        u_tip_abq=a["u_tip"], u_tip_jax=r["u_tip"][k], err_u_tip=rel(r["u_tip"][k], a["u_tip"]),
                        err_u_tip_vs_v=(r["u_tip"][k] - a["u_tip"]) / abs(a["v_tip"]),
                        v_tip_abq=a["v_tip"], v_tip_jax=r["v_tip"][k], err_v_tip=rel(r["v_tip"][k], a["v_tip"]),
                        M_abq=a["M_clamp"], M_jax=r["M_clamp"][k], err_M=rel(r["M_clamp"][k], a["M_clamp"]),
                        err_theta=rel(r["theta"][k], a["theta"]),
                        nl_effect_v=rel(a["v_tip"], a["lam"] * rl["v_tip"][-1])))
    print(f"NL k={a['frame']:2d} theta={a['theta']:.4f} err_v={nl_rows[-1]['err_v_tip']:+.2e} "
          f"err_u={nl_rows[-1]['err_u_tip']:+.2e} err_M={nl_rows[-1]['err_M']:+.2e}", flush=True)

write("abaqus_reference.csv", ref_rows, ["case", "frame", "lam", "u_tip", "v_tip", "theta", "R_tip", "M_clamp"])
write("s1_linear.csv", lin_rows, list(lin_rows[0].keys()))
write("s1_nlgeom.csv", nl_rows, list(nl_rows[0].keys()))
write("s1_timing.csv", timing, list(timing[0].keys()))

# T-peel load-point rotation (sw_s40, RP1 = node 2000001)
hp = os.path.join(RUN, "tpeel_sw_s40_hist_2000001.csv")
if os.path.exists(hp):
    h = np.genfromtxt(hp, delimiter=",", names=True, dtype=None, encoding=None)
    write("tpeel_rotation.csv", [dict(time=a, U2=b, UR3=c) for a, b, c in zip(h["time"], h["U2"], h["UR3"])],
          ["time", "U2", "UR3"])
    print(f"T-peel sw_s40: max |UR3| = {np.abs(h['UR3']).max():.4f} rad at U2 = {h['U2'][np.argmax(np.abs(h['UR3']))]:.3f} mm")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for kind, mk in (("Q4", "s"), ("Q4IM", "o")):
    rr = [x for x in lin_rows if x["element"] == kind and x["case"].startswith("lin") and x["load"] == "force"]
    ax[0].semilogy([x["ny"] for x in rr], [abs(x["err_v_tip"]) for x in rr], mk + "-", mfc="w", label=f"{kind} vs ABAQUS CPE4I")
ax[0].axhline(5e-4, color="0.5", ls=":", label="threshold 0.05 %")
ax[0].set(xlabel="elements through thickness $n_y$", ylabel=r"|tip deflection error|", xticks=[2, 4, 8, 20])
ax[0].legend(frameon=False, fontsize=8)
th = np.array([x["theta_abq"] for x in nl_rows])
for key, lab in (("err_v_tip", "$v_{tip}$"), ("err_u_tip_vs_v", r"$u_{tip}$ (rel. to $|v_{tip}|$)"), ("err_M", "clamp moment")):
    ax[1].plot(th, [abs(x[key]) for x in nl_rows], "o-", mfc="w", ms=4, label=lab)
ax[1].axhline(1e-3, color="0.5", ls=":", label="threshold 0.1 %")
ax[1].set_yscale("log")
ax[1].set(xlabel="tip rotation (rad)", ylabel="|SVK Q4IM - ABAQUS| (relative)")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s1_bulk.{ext}"), dpi=220)
print("done")
