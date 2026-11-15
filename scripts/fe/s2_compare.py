"""s2_compare.py -- S2 tests 2-4: tslid.fe single cohesive element vs ABAQUS COH2D4 histories.

Units: mm, N, MPa, s.  Light (seconds); runs on any machine with the uv env.
    python scripts/fe/s2_compare.py <dir with *_ip.csv + *.json> [outdir=results/solver/S2_interface]

Inputs: <job>_ip.csv (scripts/fe/extract_se_odb.py) and <job>.json for the 14 Phase 0 runs
(prefixed p0_) and the S2 runs (s2_).  Writes to outdir:
    s2_single_element.csv   per run: traction error vs ABAQUS (normal, shear), vs the analytic law,
                            damage error, dissipated energy (ours, ABAQUS) vs Gc, unloading error
    s2_probe.csv            COH2D4 integration-point probe (quadrature)
    s2_viscous.csv          mu -> 0 limit and viscous runs
    abaqus_se_reference.npz compact ABAQUS histories (IP 1) + metadata, read by tests/fe/test_s2_single_element.py
    s2_interface.png/.pdf
"""
import csv
import glob
import json
import os
import sys

import numpy as np

import tslid.fe  # noqa: F401
from tslid import laws
from tslid.fe.benchmarks import se_case, single_cohesive
from tslid.fe.interface import AlphaLawParams

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S2_interface"
os.makedirs(OUT, exist_ok=True)


def load(job):
    d = np.genfromtxt(os.path.join(RUN, job + "_ip.csv"), delimiter=",", names=True)
    meta = json.load(open(os.path.join(RUN, job + ".json")))
    return d, meta


def ip1(d):
    return d[d["ip"] == 1]


rows, ref = [], {}
jobs = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(RUN, "*_ip.csv")))
for job in jobs:
    if job == "s2_probe":
        continue
    d, meta = load(job)
    a = ip1(d)
    law, t, un, us, mu = se_case(meta)
    assert len(a) == len(t), (job, len(a), len(t))
    p = meta["params"]
    sigc, Gc = p["sigc"], p["Gc"]
    r = single_cohesive(law, un, us, t, mu=mu)
    L = laws.AlphaLaw(p.get("K", 8500.0), sigc, Gc, p["alpha"])
    dm = np.hypot(np.maximum(un, 0), us)
    analytic = np.array([L.traction(x) for x in dm])               # monotonic effective traction
    teff_abq = np.hypot(np.maximum(a["S22"], 0), a["S12"]) if "S12" in a.dtype.names else a["S22"]
    unl = np.r_[False, np.diff(un) < 0] | np.r_[False, (np.diff(un) == 0) & (np.diff(us) < 0)]
    secant_abq = np.abs(a["S22"] - (1 - a["SDEG"]) * law.K0 * un)  # ABAQUS's own secant consistency
    diss_abq = a["ALLDMD"][-1] + a["ALLVD"][-1]
    row = dict(job=job, rep=meta["representation"]["softening"], alpha=p["alpha"], sigc=sigc, Gc=Gc,
               mode=p.get("mode", "n"), viscosity=mu, frames=len(t),
               err_tn_vs_abq=np.abs(r["tn"] - a["S22"]).max() / sigc,
               err_ts_vs_abq=np.abs(r["ts"] - a["S12"]).max() / sigc if "S12" in a.dtype.names else np.nan,
               err_D_vs_abq=np.abs(r["Dv"] - a["SDEG"]).max(),
               err_abq_vs_analytic=(np.abs(teff_abq - analytic).max() / sigc) if mu == 0 and not unl.any() else np.nan,
               err_unload_vs_abq=(np.abs(r["tn"] - a["S22"])[unl].max() / sigc) if unl.any() else np.nan,
               abq_secant_unload=(secant_abq[unl].max() / sigc) if unl.any() else np.nan,
               diss_jax_rel=r["diss"][-1] / Gc - 1, diss_abq_rel=diss_abq / Gc - 1,
               diss_jax_vs_abq=(r["diss"][-1] - diss_abq) / Gc,
               work_jax_vs_abq=(r["W"][-1] - a["ALLWK"][-1]) / Gc)
    k1 = int(np.argmax(r["D"] >= 1.0)) if (r["D"] >= 1.0).any() else len(t)
    e = np.abs(r["tn"] - a["S22"]) / sigc
    row["err_tn_before_D1"] = e[:k1].max()
    row["err_tn_after_D1"] = e[k1:].max() if k1 < len(t) else np.nan
    rows.append(row)
    ref[job] = dict(meta=json.dumps(meta), S22=a["S22"].astype(np.float32),
                    S12=(a["S12"] if "S12" in a.dtype.names else 0 * a["S22"]).astype(np.float32),
                    SDEG=a["SDEG"].astype(np.float32), diss=np.float64(diss_abq),
                    ALLWK=np.float64(a["ALLWK"][-1]))
    print(f"{job:26s} tn {row['err_tn_vs_abq']:.2e} ts {row['err_ts_vs_abq']:.2e} D {row['err_D_vs_abq']:.2e} "
          f"diss jax {row['diss_jax_rel']:+.2e} abq {row['diss_abq_rel']:+.2e} unload {row['err_unload_vs_abq']:.2e} "
          f"preD1 {row['err_tn_before_D1']:.2e} W-ALLWK {row['work_jax_vs_abq']:+.2e}", flush=True)

# viscous limit: mu -> 0 recovers the inviscid response
law = AlphaLawParams(8500.0, 20.0, 0.4, 3.0)
L = laws.AlphaLaw(8500.0, 20.0, 0.4, 3.0)
t = np.linspace(0, 1, 2001)
un = 1.2 * L.df * t
r0 = single_cohesive(law, un, 0 * un, t)
visc = []
for mu in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2):
    rv = single_cohesive(law, un, 0 * un, t, mu=mu)
    visc.append(dict(mu=mu, mu_over_dt=mu / (t[1] - t[0]), max_dt_rel=np.abs(rv["tn"] - r0["tn"]).max() / 20.0,
                     diss_rel=rv["diss"][-1] / 0.4 - 1, peak=rv["tn"].max()))

# quadrature probe
d, meta = load("s2_probe")
f = d[d["frame"] == d["frame"].max()]
dp, K = meta["params"]["probe"], meta["params"]["K"]
gp = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2
probe = []
for ip in sorted(set(f["ip"].astype(int))):
    s = float(f[f["ip"] == ip]["S22"][0])
    probe.append(dict(ip=ip, S22=s, gauss_pred=K * dp * gp[ip - 1], nodal_pred=K * dp * [0.0, 1.0][ip - 1]))
    print("probe", probe[-1])


def write(name, rr):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
        w.writeheader()
        for x in rr:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in x.items()})


write("s2_single_element.csv", rows)
write("s2_viscous.csv", visc)
write("s2_probe.csv", probe)
np.savez_compressed(os.path.join(OUT, "abaqus_se_reference.npz"),
                    **{f"{j}__{k}": v for j, dd in ref.items() for k, v in dd.items()})

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for job, c in (("p0_se_lin_a0", "C0"), ("p0_se_exp_a3_cyc", "C1"), ("p0_se_exp_am2", "C2"), ("p0_se_exp_a7", "C3")):
    dd, meta = load(job)
    law, t, un, us, mu = se_case(meta)
    r = single_cohesive(law, un, us, t, mu=mu)
    a = ip1(dd)
    ax[0].plot(un, a["S22"], color="k", lw=2.4, alpha=0.35)
    ax[0].plot(un, r["tn"], color=c, lw=1.0, label=job.replace("p0_se_", ""))
ax[0].set(xlabel=r"opening $\delta_n$ (mm)", ylabel=r"$t_n$ (MPa)", title="(a) mode I: ABAQUS (grey) vs tslid.fe")
ax[0].legend(frameon=False, fontsize=8)
for job, c in (("s2_s_exp_a3", "C0"), ("s2_ns_exp_a3", "C1"), ("s2_ns_lin_a0", "C2")):
    dd, meta = load(job)
    law, t, un, us, mu = se_case(meta)
    r = single_cohesive(law, un, us, t, mu=mu)
    a = ip1(dd)
    dmv = np.hypot(un, us)
    ax[1].plot(dmv, np.hypot(a["S22"], a["S12"]), color="k", lw=2.4, alpha=0.35)
    ax[1].plot(dmv, np.hypot(r["tn"], r["ts"]), color=c, lw=1.0, label=job.replace("s2_", ""))
ax[1].set(xlabel=r"$\delta_m$ (mm)", ylabel=r"$\sqrt{t_n^2+t_s^2}$ (MPa)", title="(b) shear and mixed mode")
ax[1].legend(frameon=False, fontsize=8)
er = [x for x in rows]
pre = [max(x["err_tn_before_D1"], x["err_ts_vs_abq"] if np.isfinite(x["err_ts_vs_abq"]) else 0) for x in er]
ax[2].semilogy(range(len(er)), pre, "o", mfc="w", label="until inviscid D = 1")
post = [(i, x["err_tn_after_D1"]) for i, x in enumerate(er) if x["viscosity"] > 0]
ax[2].semilogy([i for i, _ in post], [v for _, v in post], "x", color="C3", label="viscous, after D = 1")
ax[2].axhline(1e-6, color="0.5", ls=":", label="1e-6 (lin/exp)")
ax[2].axhline(2e-5, color="0.5", ls="--", label="2e-5 (tabular)")
ax[2].set_xticks(range(len(er)))
ax[2].set_xticklabels([x["job"].replace("p0_se_", "").replace("s2_", "") for x in er], rotation=90, fontsize=6)
ax[2].set(ylabel=r"max $|t - t_{ABAQUS}|/\sigma_c$", title="(c) all runs")
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s2_interface.{ext}"), dpi=220)
print("done")
