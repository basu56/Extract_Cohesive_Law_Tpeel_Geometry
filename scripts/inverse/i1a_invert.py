"""i1a_invert.py -- one parametric inversion (I1a/I1b/I1c/I1d, D31) or the truth evaluation.  Heavy (sbatch).

Units: mm, N, MPa.
    python scripts/inverse/i1a_invert.py --deck <model deck.inp> --ref <ABAQUS rundir>/<job> --window prepeak|full
        --out <dir> [--start k | --theta0 a,b,c | --truth] [--params ln_sigc,ln_Gc,alpha] [--fix alpha=0]
        [--s-rel 1e-3] [--noise-s-rel 1e-3 --noise-seed n] [--lhs-seed 20261116] [--tag name]

Model: tslid.fe production (D24) on --deck, adaptive schedule per evaluation (D31), S5a Jacobian.
Data: the ABAQUS record --ref, observed with the D29 bins (window from the record: prepeak = 0..delta_peak,
full = 0..d_end); optional Gaussian noise (s = noise-s-rel * P_peak, seed recorded).
Misfit J = 1/2 sum ((y - y_data)/s)^2, s = s-rel * P_peak (noise-free: s only scales J).
Start: --start k = k-th point of an 8-point Latin hypercube over the bounds (gauss_newton.default_bounds;
same seed for every case), projected onto the admissible set; or --theta0.
--truth: evaluate model + Jacobian at the deck's law on both windows (bias prediction, I1a.4); no LM.
--params / --fix (I1d): e.g. --params ln_sigc,ln_Gc --fix alpha=0 for the bilinear fit.
Writes <out>/<tag>.json (iterates as they come, final theta, J, timings) [and <tag>_truth.npz].
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import tslid.fe  # noqa: E402,F401
from tslid.inverse.data import abaqus_record, add_noise, observed  # noqa: E402
from tslid.inverse.forward import Forward  # noqa: E402
from tslid.inverse.gauss_newton import Bounds, default_bounds, latin_hypercube, levenberg_marquardt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--deck", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--window", default="full", choices=["prepeak", "full"])
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=None)
ap.add_argument("--theta0", default=None)
ap.add_argument("--truth", action="store_true")
ap.add_argument("--params", default="ln_sigc,ln_Gc,alpha")
ap.add_argument("--fix", default="", help="e.g. alpha=0 (value of a parameter not in --params)")
ap.add_argument("--s-rel", type=float, default=1e-3)
ap.add_argument("--noise-s-rel", type=float, default=0.0)
ap.add_argument("--noise-seed", type=int, default=None)
ap.add_argument("--lhs-seed", type=int, default=20261116)
ap.add_argument("--n-lhs", type=int, default=8)
ap.add_argument("--max-iter", type=int, default=30)
ap.add_argument("--max-step", default="0.5,0.5,3.0", help="cap on |dtheta| per parameter (ln, ln, alpha); '' = none")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t_start = time.perf_counter()
rec = abaqus_record(a.ref)
obs = observed(rec)
params = tuple(a.params.split(","))
F = Forward(a.deck, params=params)
for kv in filter(None, a.fix.split(",")):                     # fixed parameters (I1d): set on the deck law
    k, v = kv.split("=")
    lw = F.law_deck
    F.law_deck = type(lw)(lw.K0, lw.sigc, lw.Gc, float(v)) if k == "alpha" else lw
truth_full = dict(sigc=rec["params"]["sigc"], Gc=rec["params"]["Gc"], alpha=rec["params"]["alpha"])
tag = a.tag or (os.path.basename(a.ref) + "_" + a.window + ("_truth" if a.truth else
                                                             f"_s{a.start}" if a.start is not None else "_theta0"))
path = os.path.join(a.out, tag + ".json")
meta = dict(tag=tag, deck=os.path.abspath(a.deck), ref=a.ref, window=a.window, params=params, fix=a.fix,
            truth=truth_full, P_peak=rec["P_peak"], delta_peak=rec["delta_peak"], d_end=rec["d_end"],
            s_rel=a.s_rel, noise_s_rel=a.noise_s_rel, noise_seed=a.noise_seed, lhs_seed=a.lhs_seed,
            setup=F.t_setup, host=os.uname().nodename)

if a.truth:
    th = F.theta_true()
    r = F.evaluate(th, {w: e for w, (e, _) in obs.items()})
    np.savez(os.path.join(a.out, tag + ".npz"), theta=th, delta=r["delta"], P=r["P"], S=r["S"],
             **{f"{w}_edges": obs[w][0] for w in obs}, **{f"{w}_y": r["obs"][w][0] for w in obs},
             **{f"{w}_S": r["obs"][w][1] for w in obs}, **{f"{w}_ydata": obs[w][1] for w in obs})
    meta.update(theta_true=th.tolist(), wall=r["wall"], increments=r["increments"], ok=r["ok"])
    json.dump(meta, open(path, "w"), indent=1)
    print(json.dumps(meta, indent=1))
    sys.exit(0)

edges, y_data = obs[a.window]
if a.noise_s_rel > 0:
    y_data = add_noise(y_data, a.noise_s_rel * rec["P_peak"], a.noise_seed)
s = a.s_rel * rec["P_peak"]
b = default_bounds(np.log(F.K0))
idx = [["ln_sigc", "ln_Gc", "alpha"].index(p) for p in params]
b = Bounds(lo=b.lo[idx], hi=b.hi[idx], names=params, lnK0=b.lnK0)
if a.theta0:
    th0 = np.array([float(x) for x in a.theta0.split(",")])
else:
    th0 = latin_hypercube(a.n_lhs, default_bounds(np.log(F.K0)), a.lhs_seed)[a.start][idx]
th0 = b.project(th0)
meta.update(theta0=th0.tolist(), bounds_lo=b.lo.tolist(), bounds_hi=b.hi.tolist(), bins=len(edges) - 1)
log = []


def evaluate(th):
    r = F.evaluate(th, {a.window: edges})
    if "obs" not in r:
        return None, None, r
    y, Sb = r["obs"][a.window]
    return y, Sb, r


def cb(entry):
    log.append(entry)
    json.dump(dict(meta, log=log, done=False), open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    print("eval %d J %.6g mu %.3g acc %s theta %s wall %.0f s" % (entry["k"], entry["J"] if entry["J"] is not None else
          np.nan, entry["mu"], entry["accepted"], np.round(entry["theta"], 6), entry["wall"]), flush=True)


res = levenberg_marquardt(evaluate, y_data, s, th0, b, max_iter=a.max_iter, callback=cb,
                         max_step=[float(v) for v in a.max_step.split(",")] if a.max_step else None)
th = res.theta
law = F.law(th)
meta.update(log=log, done=True, theta_hat=th.tolist(), J=res.J, iterations=res.iterations, converged=res.converged,
            reason=res.reason, law_hat=dict(sigc=law.sigc, Gc=law.Gc, alpha=law.alpha), evaluations=F.n_eval,
            wall_total=time.perf_counter() - t_start)
json.dump(meta, open(path + ".tmp", "w"), indent=1)
os.replace(path + ".tmp", path)
print(json.dumps({k: v for k, v in meta.items() if k != "log"}, indent=1))
