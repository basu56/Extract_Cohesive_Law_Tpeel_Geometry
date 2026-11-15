"""s5c_fisher.py -- S5c: Fisher matrix of sw_s40 three ways, schedule stability, weak direction vs D18.
Light.

Units: mm, N.  theta = (ln K0, ln sigc, ln Gc, alpha); theta3 = (ln sigc, ln Gc, alpha).
    python scripts/fe/s5c_fisher.py <tslid_runs root> [outdir=results/solver/S5_sensitivity]

Sensitivities (S5a, production sw_s40): two schedules,
  A = solver_S5a_cost/sw_s40_sens_sens.npz  (rtol 1e-6, D24 production)
  B = solver_S5c/sw_s40_t10_sens.npz        (rtol 1e-10, its own schedule)
Fisher (D29), s = assumed load-noise level (not a measurement):
  (i)   prepeak : y = O_pre P  (bin averages, observation.py, bins up to delta_peak of ABAQUS sw_s40),
                  F = (O S)^T (O S) / s^2  (each bin one observation with noise s)
  (ii)  full    : the same with the bins up to d_end (compare_ld window)
  (iii) raw     : the increments themselves, F = sum_n w_n S_n S_n^T / s^2, w_n = trapezoid weight in
                  delta / bin width (the same observation density as (ii)), delta_n <= d_end
Reported per variant, run and parameter set: eigenvalues (s = 1e-3 and 1e-4 P_peak), eigenvectors,
condition number, Cramer-Rao standard deviations sqrt(diag F^-1); stability A vs B: relative change of
each eigenvalue, angle between the weak eigenvectors.
Trade-off direction d(ln sigc)/d alpha at (sigc 40, alpha 0), compared in the (ln sigc, alpha) plane:
  D18 pairs (secants, raw and corrected for the pair's residual dP_peak with the local dP_peak/d ln sigc),
  the local iso-peak direction from the sweeps (sigc 35/45, alpha -1.5/+1.5; ABAQUS and tslid.fe),
  S5a: dP_peak/dtheta (increment and bin-average peak), weak eigenvector of the (ln sigc, alpha) block of
  F (i) and (ii) (and with ln Gc, ln K0 included: its (ln sigc, alpha) part).
Cross-check (S5c item 4): central FD of the sweep curves around sigc 40, alpha 0 vs S5a dP/dtheta,
bin-averaged, per window.
Writes s5c_fisher.csv, s5c_stability.csv, s5c_directions.csv, s5c_fdcheck.csv, s5c.png/.pdf.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.fe.observation import DELTA_BIN, observation_matrix, observe, windows  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S5_sensitivity"
os.makedirs(OUT, exist_ok=True)
NAMES = ["ln_K0", "ln_sigc", "ln_Gc", "alpha"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1495b", "#6b6b6b"]
A1 = os.path.join(ROOT, "phase1a_sigc_sweep")
B1 = os.path.join(ROOT, "phase1b_alpha_sweep")
S4 = os.path.join(ROOT, "solver_S4a")
RUNS = {"A_rtol1e-6": os.path.join(ROOT, "solver_S5a_cost", "sw_s40_sens_sens.npz"),
        "B_rtol1e-10": os.path.join(ROOT, "solver_S5c", "sw_s40_t10_sens.npz")}
SETS = {"theta3": [1, 2, 3], "theta4": [0, 1, 2, 3]}

ref = load(A1, "sw_s40")
P_PEAK = float(ref["P"].max())
D_PK = float(ref["d"][int(np.argmax(ref["P"]))])
W = windows(ref["d_end"], D_PK)
Z = {k: dict(np.load(v)) for k, v in RUNS.items() if os.path.exists(v)}


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


def obs_S(z, kind):
    """(observations x 4) sensitivities and the per-observation weights."""
    d, S = z["delta"], z["dP"]
    if kind in W:
        O = observation_matrix(np.r_[0.0, d], W[kind])[:, 1:]
        return O @ S, np.ones(O.shape[0])
    m = d <= ref["d_end"]
    dd = np.r_[0.0, d[m]]                                               # node 0 (delta 0) carries S = 0
    left = np.diff(dd)
    right = np.r_[left[1:], 0.0]
    return S[m], 0.5 * (left + right) / DELTA_BIN                       # trapezoid weights of nodes n >= 1


def fisher(z, kind, cols):
    Y, w = obs_S(z, kind)
    Y = Y[:, cols]
    return (Y * w[:, None]).T @ Y                                        # x 1/s^2


def eig(F):
    lam, V = np.linalg.eigh(F)
    V = V * np.sign(V[-1:, :] + 1e-300)                                  # last component (alpha) >= 0
    return lam, V


KINDS = {"i_prepeak": "prepeak", "ii_full_filtered": "full", "iii_full_raw": "raw"}
frows, srows, E = [], [], {}
for run, z in Z.items():
    for kn, kind in KINDS.items():
        for sn, cols in SETS.items():
            F = fisher(z, kind, cols)
            lam, V = eig(F)
            E[(run, kn, sn)] = (lam, V, F)
            for s_rel in (1e-3, 1e-4):
                s2 = (s_rel * P_PEAK) ** 2
                crb = np.sqrt(np.diag(np.linalg.inv(F / s2)))
                r = dict(run=run, variant=kn, params=sn, s_over_Ppeak=s_rel, n_obs=obs_S(z, kind)[0].shape[0],
                         cond=float(lam[-1] / lam[0]))
                for i, l in enumerate(lam):
                    r[f"lambda{i}"] = float(l / s2)
                for i, c in enumerate(cols):
                    r[f"crb_{NAMES[c]}"] = float(crb[i])
                for i, c in enumerate(cols):
                    r[f"weak_{NAMES[c]}"] = float(V[i, 0])
                frows.append(r)
if frows:
    write(frows, "s5c_fisher.csv")


def angle(u, v):
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(min(1.0, abs(float(u @ v))))))


if len(Z) == 2:
    a, b = list(Z)
    for kn in KINDS:
        for sn in SETS:
            la, Va, _ = E[(a, kn, sn)]
            lb, Vb, _ = E[(b, kn, sn)]
            r = dict(variant=kn, params=sn)
            for i in range(len(la)):
                r[f"dlambda{i}_rel"] = float((lb[i] - la[i]) / la[i])
                r[f"angle_v{i}_deg"] = angle(Va[:, i], Vb[:, i])
            r["max_dlambda_rel"] = float(np.max(np.abs((lb - la) / la)))
            r["weak_angle_deg"] = r["angle_v0_deg"]
            srows.append(r)
    write(srows, "s5c_stability.csv")

# ---------------------------------------------------------------- trade-off directions, (ln sigc, alpha) plane
drows = []


def dir_row(source, dl, da, note=""):
    v = np.array([dl, da], dtype=float)
    v = v / np.linalg.norm(v) * np.sign(v[1] if v[1] != 0 else 1.0)
    drows.append(dict(source=source, dlnsigc_dalpha=float(v[0] / v[1]) if v[1] else np.inf, v_lnsigc=float(v[0]),
                      v_alpha=float(v[1]), note=note))
    return v


def peak(rd, job):
    x = load(rd, job)
    return float(x["P"].max())


ppk = {}
for solver, getter in (("abaqus", lambda j: peak(A1 if j.startswith("sw") else B1, j)),
                       ("tslid.fe", lambda j: peak(S4, j + "_default"))):
    try:
        ppk[solver] = {j: getter(j) for j in ("sw_s35", "sw_s45", "as_s40_am1p5", "as_s40_a1p5", "sw_s40",
                                              "sw_s25", "sw_s30", "sw_s21", "as_s21_am1p5", "as_s21_am3",
                                              "as_s40_a3", "as_s40_a5", "as_s40_a7")}
    except Exception as e:                                           # noqa: BLE001
        print("peaks", solver, "unavailable:", e)
for solver, pk in ppk.items():
    gs = (pk["sw_s45"] - pk["sw_s35"]) / np.log(45 / 35)
    ga = (pk["as_s40_a1p5"] - pk["as_s40_am1p5"]) / 3.0
    dir_row(f"sweeps iso-peak at (40, 0), {solver}", -ga, gs, f"dPpk/dlnsigc {gs:.4g} N, dPpk/dalpha {ga:.4g} N")
    if solver == "abaqus":
        for (r0, s0), (o, so, ao) in ((("sw_s25", 25), ("as_s21_am1p5", 21, -1.5)), (("sw_s25", 25), ("as_s40_a5", 40, 5)),
                                       (("sw_s25", 25), ("as_s40_a7", 40, 7)), (("sw_s30", 30), ("as_s21_am3", 21, -3)),
                                       (("sw_s30", 30), ("as_s40_a3", 40, 3))):
            dl, da = np.log(so / s0), ao
            dir_row(f"D18 pair {o} vs {r0} (secant)", dl, da, f"dPpk {100 * (pk[o] - pk[r0]) / pk[r0]:+.2f} %")
            # correct the residual peak mismatch along ln sigc with the local gradient at the pair's sigc
            dl_c = dl - (pk[o] - pk[r0]) / gs
            dir_row(f"D18 pair {o} vs {r0} (corrected)", dl_c, da, "residual dPpk moved along ln sigc")
for run, z in Z.items():
    i = int(np.argmax(z["P"]))
    g = z["dP"][i]
    dir_row(f"S5a dP_peak/dtheta (increment peak), {run}", -g[3], g[1])
    Y, _ = obs_S(z, "full")
    yP = observation_matrix(np.r_[0.0, z["delta"]], W["full"])[:, 1:] @ z["P"]
    k = int(np.argmax(yP))
    dir_row(f"S5a dP_peak/dtheta (bin-average peak), {run}", -Y[k, 3], Y[k, 1])
    for kn in ("i_prepeak", "ii_full_filtered", "iii_full_raw"):
        for sn, cols in SETS.items():
            lam, V, F = E[(run, kn, sn)]
            if sn == "theta3":
                Fb = F[np.ix_([0, 2], [0, 2])]                            # (ln sigc, alpha) block
                lb, Vb = np.linalg.eigh(Fb)
                dir_row(f"F {kn} (lnsigc, alpha) block weak, {run}", Vb[0, 0], Vb[1, 0])
            i_s, i_a = cols.index(1), cols.index(3)
            dir_row(f"F {kn} {sn} weak eigvec projected, {run}", V[i_s, 0], V[i_a, 0],
                    f"|(v_sigc, v_alpha)| = {np.hypot(V[i_s, 0], V[i_a, 0]):.3f}")
ref_dirs = [r for r in drows if r["source"].startswith("sweeps iso-peak at (40, 0), abaqus")]
if ref_dirs:
    v0 = np.array([ref_dirs[0]["v_lnsigc"], ref_dirs[0]["v_alpha"]])
    for r in drows:
        r["angle_to_sweep_isopeak_deg"] = angle(np.array([r["v_lnsigc"], r["v_alpha"]]), v0)
write(drows, "s5c_directions.csv")

# ---------------------------------------------------------------- S5c item 4: FD of the sweeps vs S5a
crow = []
if "A_rtol1e-6" in Z:
    z = Z["A_rtol1e-6"]
    for solver, get in (("abaqus", lambda j: load(A1 if j.startswith("sw") else B1, j)),
                        ("tslid.fe", lambda j: load(S4, j + "_default"))):
        try:
            for nm, (jp, jm, step, col) in {"ln_sigc": ("sw_s45", "sw_s35", np.log(45 / 35), 1),
                                             "alpha": ("as_s40_a1p5", "as_s40_am1p5", 3.0, 3)}.items():
                xp, xm = get(jp), get(jm)
                for kind in ("prepeak", "full"):
                    e = W[kind]
                    fd = (observe(np.r_[0.0, xp["d"]] if xp["d"][0] > 0 else xp["d"],
                                  np.r_[0.0, xp["P"]] if xp["d"][0] > 0 else xp["P"], e) -
                          observe(np.r_[0.0, xm["d"]] if xm["d"][0] > 0 else xm["d"],
                                  np.r_[0.0, xm["P"]] if xm["d"][0] > 0 else xm["P"], e)) / step
                    S = observation_matrix(np.r_[0.0, z["delta"]], e)[:, 1:] @ z["dP"][:, col]
                    crow.append(dict(solver=solver, param=nm, window=kind, fd_step=step,
                                     rel_l2=float(np.linalg.norm(S - fd) / np.linalg.norm(S)),
                                     corr=float(np.corrcoef(S, fd)[0, 1])))
        except Exception as ex:                                    # noqa: BLE001
            print("fd check", solver, "unavailable:", ex)
    write(crow, "s5c_fdcheck.csv")

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
if Z:
    run = next(iter(Z))
    z = Z[run]
    m = z["delta"] <= ref["d_end"]
    Y, _ = obs_S(z, "full")
    cen = 0.5 * (W["full"][1:] + W["full"][:-1])
    for j, nm in enumerate(NAMES):
        ax[0].plot(z["delta"][m], z["dP"][m, j], color=C[j], lw=0.4, alpha=0.35)
        ax[0].plot(cen, Y[:, j], color=C[j], lw=1.3, label=nm + " (bin average)")
    ax[0].axvspan(0.10, 0.22, color="0.92", zorder=0)
    ax[0].axvline(D_PK, color="0.5", ls=":", lw=0.8)
    lo, hi = np.percentile(Y, [0.5, 99.5])
    ax[0].set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    a2 = ax[0].twinx()
    a2.plot(z["delta"][m], z["P"][m], color="0.7", lw=1.8)
    a2.set_ylabel("P (N)", color="0.5")
    ax[0].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$dP/d\theta$ (N)",
              title=f"(a) sw_s40 sensitivities, {run}: raw (thin), bin average (thick)\n"
                    "grey band: D18 pre-peak window; dotted: ABAQUS peak")
    ax[0].legend(frameon=False, fontsize=7, loc="upper right")
for (run, z), mk in zip(Z.items(), ("o", "s")):
    for (kn, _), c in zip(KINDS.items(), C):
        lam = E[(run, kn, "theta4")][0] / (1e-3 * P_PEAK) ** 2
        ax[1].semilogy(np.arange(len(lam)) + (0.1 if mk == "s" else -0.1), lam, mk, color=c, mfc="w" if mk == "s" else c,
                       label=f"{kn}, {run}")
ax[1].set(xlabel="eigenvalue index (theta4)", ylabel=r"eigenvalue of F, s = 1e-3 $P_{peak}$",
          title="(b) Fisher spectra, two schedules")
ax[1].legend(frameon=False, fontsize=6)
for r, c in zip([r for r in drows if ("pair" not in r["source"] or "corrected" in r["source"])
                 and "projected" not in r["source"] and "B_" not in r["source"]], C * 3):
    v = np.array([r["v_lnsigc"], r["v_alpha"]])
    ax[2].plot([-v[1] * 0, v[1]], [0, v[0]], color=c, lw=1.2, label=r["source"][:48])
ax[2].set(xlabel=r"$\alpha$ component", ylabel=r"$\ln\sigma_c$ component",
          title="(c) trade-off directions in (ln sigma_c, alpha), unit vectors")
ax[2].legend(frameon=False, fontsize=6)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s5c.{ext}"), dpi=200)
print("done")
