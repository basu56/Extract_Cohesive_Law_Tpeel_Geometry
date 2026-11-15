"""s5a_analysis.py -- S5.0 forward gate, S5a determinism / step study / tolerance / cost tables and figure.
Light.

Units: mm, N, s.
    python scripts/fe/s5a_analysis.py <tslid_runs root> [outdir=results/solver/S5_sensitivity]

Inputs (Negishi, scripts/fe/run_s5*.sh):
  solver_S5_gate/<job>_{old,new,new2}_jax.csv/json   S5.0 gate 2 (old = pre-S5 code 7e21d1a)
  solver_S5a/c_s40_{b1,b2,bslu,r0,t6,t6r}*            coarse: sensitivities rtol 1e-10 (twice, SuperLU),
                                                      replay, rtol 1e-6 (own schedule / replay)
  solver_S5a/c_s40_fd_<param>_<h>_{p,m}_jax.csv       FD step study (replay of b1, rtol 1e-10)
  solver_S5a_cost/{c_s40,sw_s40}_{fwd*,sens*}         cost (one job, one pinned thread)
Comparison window: 0 <= delta <= d_end of the ABAQUS reference (compare_ld: the sigma_c front reaches
L - 15 mm), solver_S3c/c_s40 (coarse) and phase1a_sigc_sweep/sw_s40 (production).
Writes s5_gate.csv, s5a_determinism.csv, s5a_stepstudy.csv, s5a_tolerance.csv, s5a_cost.csv,
s5a_sw_s40_S.npz (production sensitivities), s5a.png/.pdf.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S5_sensitivity"
os.makedirs(OUT, exist_ok=True)
NAMES = ("ln_K0", "ln_sigc", "ln_Gc", "alpha")
HS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1495b"]
GATE = os.path.join(ROOT, "solver_S5_gate")
CO = os.path.join(ROOT, "solver_S5a")
COST = os.path.join(ROOT, "solver_S5a_cost")


def rd(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("time", "delta", "P")}


def js(path):
    return json.load(open(path)) if os.path.exists(path) else None


def write(rows, name):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    for r in rows:
        print(name, {k: (f"{v:.4g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


# ---------------------------------------------------------------- S5.0 gate 2
gate = []
for job in ("sw_s40", "sw_s58", "sw_s21", "as_s21_am3", "as_s21_a7"):
    f = {t: os.path.join(GATE, f"{job}_{t}_jax") for t in ("old", "new", "new2")}
    if not all(os.path.exists(v + ".json") and js(v + ".json")["complete"] for v in f.values()):
        print("gate: missing / incomplete", job)
        continue
    R = {t: rd(v + ".csv") for t, v in f.items()}
    J = {t: js(v + ".json") for t, v in f.items()}
    Pk = np.abs(R["old"]["P"]).max()
    row = dict(job=job, increments_old=len(R["old"]["P"]) - 1, increments_new=len(R["new"]["P"]) - 1)
    for a, b in (("old", "new"), ("new", "new2")):
        same = len(R[a]["P"]) == len(R[b]["P"])
        row[f"same_schedule_{a}_{b}"] = bool(same and np.array_equal(R[a]["time"], R[b]["time"]))
        row[f"bitwise_P_{a}_{b}"] = bool(same and np.array_equal(R[a]["P"], R[b]["P"]))
        if same:
            row[f"max_dP_rel_{a}_{b}"] = float(np.abs(R[a]["P"] - R[b]["P"]).max() / Pk)
        else:                                                  # different increments: on the union of deltas
            du = np.union1d(R[a]["delta"], R[b]["delta"])
            row[f"max_dP_rel_{a}_{b}"] = float(np.abs(np.interp(du, R[a]["delta"], R[a]["P"]) -
                                                      np.interp(du, R[b]["delta"], R[b]["P"])).max() / Pk)
    for t in f:
        row[f"wall_{t}"] = J[t]["wall_run"]
        row[f"iters_{t}"] = J[t]["iterations_all"]
    row["wall_new_over_old"] = J["new"]["wall_run"] / J["old"]["wall_run"]
    row["wall_new2_over_old"] = J["new2"]["wall_run"] / J["old"]["wall_run"]
    row["host"] = J["new"].get("affinity")
    gate.append(row)
if gate:
    write(gate, "s5_gate.csv")

# ---------------------------------------------------------------- coarse: determinism, replay, tolerance
S = {t: np.load(os.path.join(CO, f"c_s40_{t}_sens.npz")) for t in ("b1", "b2", "bslu", "t6", "t6r")
     if os.path.exists(os.path.join(CO, f"c_s40_{t}_sens.npz"))}
ref = load(os.path.join(ROOT, "solver_S3c"), "c_s40")
d_end = ref["d_end"]
det, tol = [], []
if "b1" in S:
    b1 = S["b1"]
    win = b1["delta"] <= d_end
    if "b2" in S:
        b2 = S["b2"]
        det.append(dict(pair="b1 vs b2 (PARDISO, repeat)", n=len(b1["P"]),
                        bitwise_time=bool(np.array_equal(b1["time"], b2["time"])),
                        bitwise_P=bool(np.array_equal(b1["P"], b2["P"])),
                        bitwise_dP=bool(np.array_equal(b1["dP"], b2["dP"])),
                        max_dP_rel=float(np.abs(b1["P"] - b2["P"]).max() / np.abs(b1["P"]).max()),
                        max_dS_rel=float(np.abs(b1["dP"] - b2["dP"]).max() / np.abs(b1["dP"]).max())))
    if "bslu" in S and len(S["bslu"]["P"]) == len(b1["P"]):
        c = S["bslu"]
        det.append(dict(pair="b1 vs bslu (PARDISO vs SuperLU)", n=len(c["P"]),
                        bitwise_time=bool(np.array_equal(b1["time"], c["time"])),
                        max_dP_rel=float(np.abs(b1["P"] - c["P"]).max() / np.abs(b1["P"]).max()),
                        **{f"dS_rel_{nm}": float(np.linalg.norm((b1["dP"] - c["dP"])[win, j]) /
                                                 np.linalg.norm(b1["dP"][win, j])) for j, nm in enumerate(NAMES)}))
    r0 = os.path.join(CO, "c_s40_r0_jax.csv")
    if os.path.exists(r0):
        r = rd(r0)
        same = len(r["P"]) - 1 == len(b1["P"])
        det.append(dict(pair="b1 vs r0 (replay of b1's schedule)", n=len(r["P"]) - 1, same_increments=same,
                        max_dP_rel=float(np.abs(r["P"][1:] - b1["P"]).max() / np.abs(b1["P"]).max()) if same else None))
    for t, lab in (("t6r", "rtol 1e-6, b1 schedule"), ("t6", "rtol 1e-6, own schedule")):
        if t not in S:
            continue
        c = S[t]
        row = dict(run=lab, increments=len(c["P"]), increments_b1=len(b1["P"]))
        for j, nm in enumerate(NAMES):
            if t == "t6r" and len(c["P"]) == len(b1["P"]):
                d = (c["dP"] - b1["dP"])[win, j]
                row[f"rel_{nm}"] = float(np.linalg.norm(d) / np.linalg.norm(b1["dP"][win, j]))
            else:                                             # on b1's deltas in the window
                g = b1["delta"][win]
                d = np.interp(g, c["delta"], c["dP"][:, j]) - b1["dP"][win, j]
                row[f"rel_{nm}"] = float(np.linalg.norm(d) / np.linalg.norm(b1["dP"][win, j]))
        row["P_rel"] = float(np.abs(np.interp(b1["delta"][win], c["delta"], c["P"]) - b1["P"][win]).max()
                             / np.abs(b1["P"]).max())
        tol.append(row)
if det:
    write(det, "s5a_determinism.csv")
if tol:
    write(tol, "s5a_tolerance.csv")

# ---------------------------------------------------------------- FD step study
step, fdcurves = [], {}
if "b1" in S:
    b1 = S["b1"]
    win = b1["delta"] <= d_end
    kink = (b1["n_switch"] > 0) | (b1["n_fail"] > 0)
    for j, nm in enumerate(NAMES):
        for h in HS:
            fp, fm = (os.path.join(CO, f"c_s40_fd_{nm}_{h}_{s}_jax.csv") for s in ("p", "m"))
            if not (os.path.exists(fp) and os.path.exists(fm)):
                continue
            p, m = rd(fp), rd(fm)
            ok = len(p["P"]) - 1 == len(b1["P"]) and len(m["P"]) - 1 == len(b1["P"])
            row = dict(param=nm, h=float(h), replay_same_increments=ok)
            if ok:
                fd = (p["P"][1:] - m["P"][1:]) / (2 * float(h))
                e = b1["dP"][:, j] - fd
                row.update(curve_rel=float(np.linalg.norm(e[win]) / np.linalg.norm(fd[win])),
                           curve_rel_no_kink=float(np.linalg.norm(e[win & ~kink]) / np.linalg.norm(fd[win & ~kink])),
                           point_max_rel=float(np.abs(e[win]).max() / np.abs(fd[win]).max()),
                           delta_at_point_max=float(b1["delta"][win][np.argmax(np.abs(e[win]))]),
                           kink_increments_in_window=int((kink & win).sum()))
                fdcurves[(nm, h)] = fd
            else:
                row["increments_p_m"] = f"{len(p['P']) - 1}/{len(m['P']) - 1} vs {len(b1['P'])}"
            step.append(row)
if step:
    write(step, "s5a_stepstudy.csv")

# ---------------------------------------------------------------- cost
cost = []
for job, tags in (("c_s40", ("fwd1", "sens1", "fwd2", "sens2")), ("sw_s40", ("fwd", "sens"))):
    for t in tags:
        J = js(os.path.join(COST, f"{job}_{t}_jax.json"))
        if J is None or not J["complete"]:
            continue
        cost.append(dict(job=job, run=t, wall_run=J["wall_run"], increments=J["increments"],
                         iterations_all=J["iterations_all"], sens_wall=J.get("sens_wall"),
                         sens_hessian=(J.get("sens_timers") or {}).get("hessian"),
                         sens_solve=(J.get("sens_timers") or {}).get("solve"),
                         first_call=J["wall_first_call"], first_call_sens=J.get("wall_first_call_sens")))
for job in ("c_s40", "sw_s40"):
    fw = [r["wall_run"] for r in cost if r["job"] == job and r["run"].startswith("fwd")]
    for r in cost:
        if r["job"] == job and fw:
            r["over_fwd_mean"] = r["wall_run"] / np.mean(fw)
if cost:
    write(cost, "s5a_cost.csv")
prod = os.path.join(COST, "sw_s40_sens_sens.npz")
if os.path.exists(prod):
    z = dict(np.load(prod))
    z["d_end_window"] = load(os.path.join(ROOT, "phase1a_sigc_sweep"), "sw_s40")["d_end"]
    np.savez(os.path.join(OUT, "s5a_sw_s40_S.npz"), **z)

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
for j, nm in enumerate(NAMES):
    rr = [r for r in step if r["param"] == nm and "curve_rel" in r]
    if rr:
        ax[0].loglog([r["h"] for r in rr], [r["curve_rel"] for r in rr], "o-", color=C[j], mfc="w", label=nm)
ax[0].axhline(1e-5, color="0.5", ls=":", lw=1)
ax[0].text(1.2e-7, 1.3e-5, "gate 1e-5", fontsize=8, color="0.3")
ax[0].set(xlabel="relative FD step h", ylabel=r"$\|S_{dir} - S_{FD}\|_2 / \|S_{FD}\|_2$ (window)",
          title="(a) coarse T-peel: direct vs central FD\n(replay, rtol 1e-10, deterministic)")
ax[0].legend(frameon=False, fontsize=8)
if "b1" in S:
    b1 = S["b1"]
    for j, nm in enumerate(NAMES):
        ax[1].plot(b1["delta"], b1["dP"][:, j], color=C[j], lw=1.1, label=nm)
        fd = fdcurves.get((nm, "1e-4"))
        if fd is not None:
            sel = slice(None, None, max(1, len(fd) // 30))
            ax[1].plot(b1["delta"][sel], fd[sel], "o", color=C[j], mfc="w", ms=3.5)
    ax[1].axvline(d_end, color="0.6", ls="--", lw=0.8)
    ax[1].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$dP/d\theta$ (N)",
              title="(b) coarse: sensitivities (lines), FD h = 1e-4 (markers)")
    ax[1].legend(frameon=False, fontsize=8)
if os.path.exists(prod):
    z = np.load(prod)
    a2 = ax[2].twinx()
    a2.plot(z["delta"], z["P"], color="0.75", lw=2.0)
    a2.set_ylabel("P (N)", color="0.5")
    for j, nm in enumerate(NAMES):
        ax[2].plot(z["delta"], z["dP"][:, j], color=C[j], lw=1.1, label=nm)
    ax[2].axvspan(0.10, 0.22, color="0.92", zorder=0)
    ax[2].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$dP/d\theta$ (N)",
              title="(c) production sw_s40, rtol 1e-6\n(grey band: D18 pre-peak window; grey line P)")
    ax[2].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s5a.{ext}"), dpi=200)
print("done")
