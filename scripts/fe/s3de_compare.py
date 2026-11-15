"""s3de_compare.py -- S3d / S3e tables and figure from compare_solvers.py.  Light.

Units: mm, N.
    python scripts/fe/s3de_compare.py <tslid_runs root> [outdir=results/solver/S3_tpeel]

S3d (sw_s40): tslid.fe default (D24) vs ABAQUS, and the toggles (Hencky; cohesive frame "global" and
"midsurface") each vs the default and vs ABAQUS.  S3e: sw_s58, sw_s21, as_s21_am3, as_s21_a7, default
vs ABAQUS.  Runs: <root>/solver_S3d/sw_s40_<tag>, <root>/solver_S3e/<job>_default; ABAQUS references
in phase1a_sigc_sweep / phase1b_alpha_sweep.
Writes s3d_compare.csv, s3e_compare.csv (one row per comparison, all compare_solvers.py fields, gates),
s3de.png/.pdf (per case: (P - P_ABAQUS)/P_peak vs delta in the window; S3d toggles vs the default).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.analysis.compare_solvers import compare, fmt  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/solver/S3_tpeel"
REF = {"sw_s40": "phase1a_sigc_sweep", "sw_s58": "phase1a_sigc_sweep", "sw_s21": "phase1a_sigc_sweep",
       "as_s21_am3": "phase1b_alpha_sweep", "as_s21_a7": "phase1b_alpha_sweep"}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]
FLAT = ("gates",)


def flat(o, **extra):
    r = dict(extra)
    for k, v in o.items():
        if k == "gates":
            r.update({"gate_" + g: bool(p) for g, p in v.items()})
        elif isinstance(v, (float, np.floating)):
            r[k] = float(v)
        else:
            r[k] = v
    return r


def write(rows, name):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


s3d, s3e, curves = [], [], {}
D = os.path.join(ROOT, "solver_S3d")
abq = os.path.join(ROOT, REF["sw_s40"])
for tag in ("default", "hencky", "frame_global", "frame_midsurface"):
    job = f"sw_s40_{tag}"
    if not os.path.exists(os.path.join(D, job + "_response.csv")):
        print("missing", job)
        continue
    o = compare(abq, "sw_s40", D, job)
    print(fmt(o))
    s3d.append(flat(o, case="sw_s40", run=tag, reference="abaqus"))
    if tag != "default":
        o2 = compare(D, "sw_s40_default", D, job)
        print(fmt(o2))
        s3d.append(flat(o2, case="sw_s40", run=tag, reference="tslid.fe default"))
E = os.path.join(ROOT, "solver_S3e")
for case in ("sw_s58", "sw_s21", "as_s21_am3", "as_s21_a7"):
    job = f"{case}_default"
    if not os.path.exists(os.path.join(E, job + "_response.csv")):
        print("missing", job)
        continue
    o = compare(os.path.join(ROOT, REF[case]), case, E, job)
    print(fmt(o))
    s3e.append(flat(o, case=case, run="default", reference="abaqus"))
if s3d:
    write(s3d, "s3d_compare.csv")
if s3e:
    write(s3e, "s3e_compare.csv")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
cases = [("sw_s40", D, "sw_s40_default")] + [(c, E, f"{c}_default") for c in ("sw_s58", "sw_s21", "as_s21_am3",
                                                                               "as_s21_a7")]
cases = [c for c in cases if os.path.exists(os.path.join(c[1], c[2] + "_response.csv"))]
fig, ax = plt.subplots(1, len(cases) + 1, figsize=(3.6 * (len(cases) + 1), 3.6))
for axi, (case, rd, job) in zip(ax, cases):
    a, b = load(os.path.join(ROOT, REF[case]), case), load(rd, job)
    dw = min(a["d_end"], b["d_end"])
    du = np.union1d(a["d"][a["d"] <= dw], b["d"][b["d"] <= dw])
    e = (np.interp(du, b["d"], b["P"]) - np.interp(du, a["d"], a["P"])) / a["P"].max()
    axi.plot(du, 1e2 * e, color=C[0], lw=0.8)
    axi.axhline(0, color="0.6", lw=0.8)
    axi.set(xlabel=r"$\Delta$ (mm)", title=case, ylim=(-0.012, 0.012))
    axi.text(0.97, 0.04, "max %.1e, rms %.1e" % (np.abs(e).max(), np.sqrt(np.mean(np.interp(
        np.linspace(0, dw, 20000), du, e) ** 2))), transform=axi.transAxes, ha="right", fontsize=8, color="0.2")
ax[0].set_ylabel(r"$(P - P_{ABAQUS}) / P_{peak}$ (%)")
a = load(D, "sw_s40_default") if os.path.exists(os.path.join(D, "sw_s40_default_response.csv")) else None
if a is not None:
    for (tag, lab), c in zip((("hencky", "Hencky"), ("frame_global", "frame 'global'"),
                              ("frame_midsurface", "frame 'midsurface'")), C[1:]):
        p = os.path.join(D, f"sw_s40_{tag}_response.csv")
        if not os.path.exists(p):
            continue
        b = load(D, f"sw_s40_{tag}")
        dw = min(a["d_end"], b["d_end"])
        g = np.linspace(0, dw, 6000)
        ax[-1].plot(g, 1e2 * (np.interp(g, b["d"], b["P"]) - np.interp(g, a["d"], a["P"])) / a["P"].max(), color=c,
                    lw=1.0, label=lab)
    ax[-1].axhline(0, color="0.6", lw=0.8)
    ax[-1].set(xlabel=r"$\Delta$ (mm)", ylabel=r"$(P - P_{default}) / P_{peak}$ (%)", title="sw_s40 toggles vs default")
    ax[-1].legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s3de.{ext}"), dpi=200)
print("done")
