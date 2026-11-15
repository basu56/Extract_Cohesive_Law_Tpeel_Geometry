"""s1_plot.py -- S1 figure from the CSVs written by s1_compare.py.   python s1_plot.py <outdir>"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/solver/S1_bulk"
rd = lambda n: list(csv.DictReader(open(os.path.join(OUT, n))))
lin, nl = rd("s1_linear.csv"), rd("s1_nlgeom.csv")
f = lambda r, k: abs(float(r[k]))
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for kind, mk in (("Q4", "s"), ("Q4IM", "o")):
    rr = [x for x in lin if x["element"] == kind and x["case"].startswith("lin") and x["load"] == "force"]
    ax[0].semilogy([int(x["ny"]) for x in rr], [f(x, "err_v_tip") for x in rr], mk + "-", mfc="w",
                   label=f"{kind} vs ABAQUS CPE4I")
ax[0].axhline(5e-4, color="0.5", ls=":", label="S1 threshold 0.05 %")
ax[0].set(xlabel="elements through thickness $n_y$", ylabel="|tip deflection error| (linear)",
          xticks=[2, 4, 8, 20], title="(a) linear, P = 230 N")
ax[0].legend(frameon=False, fontsize=8)
for kin, ls in (("svk", "-"), ("hencky", "--")):
    rr = [x for x in nl if x["kinematics"] == kin]
    th = [float(x["theta_abq"]) for x in rr]
    for key, lab, c in (("err_v_tip", "$v_{tip}$", "C0"), ("err_u_tip", "$u_{tip}$", "C1"), ("err_M", "clamp moment", "C2")):
        ax[1].plot(th, [f(x, key) for x in rr], "o" + ls, color=c, mfc="w", ms=3.5, label=f"{lab}, {kin.upper()}")
ax[1].axhline(1e-3, color="0.5", ls=":", label="S1 threshold 0.1 %")
ax[1].axvline(0.0325, color="0.6", lw=0.8)
ax[1].text(0.0335, 3e-1, "T-peel max\nrotation", fontsize=7, color="0.4", va="top")
ax[1].set(yscale="log", ylim=(1e-8, 1), xlabel="tip rotation (rad)",
          ylabel="|Q4IM - CPE4I| / |CPE4I|", title="(b) NLGEOM, Q4IM $n_y$ = 20")
ax[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"s1_bulk.{ext}"), dpi=220)
