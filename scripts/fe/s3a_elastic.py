"""s3a_elastic.py -- S3a elastic gate, tslid.fe side: solve an elastic T-peel deck.

Units: mm, N, MPa, s.  Production mesh = heavy: run through sbatch (scripts/fe/run_s3a.sh).
    python scripts/fe/s3a_elastic.py <deck.inp> <outdir> [--kinematics svk|hencky] [--n-inc N]
                                     [--frame global|midsurface] [--tag name]

Load: the step boundary condition (RP1 U2 = delta) ramped linearly in n_inc equal increments
(the elastic deck has dtmax = T/n_inc), one Newton solve per increment from the previous solution.
Writes <outdir>/s3a_jax_<tag>.csv (one row per increment: lam, RP1/RP2 U, UR3, RF, RM, mouth U2,
energy, Newton iterations, timings) and s3a_jax_<tag>_U.npz (full nodal U per increment).
"""
import argparse
import csv
import os
import time

import jax.numpy as jnp
import numpy as np

import tslid.fe  # noqa: F401
from tslid.fe.tpeel import TPeelModel, newton

ap = argparse.ArgumentParser()
ap.add_argument('deck')
ap.add_argument('outdir')
ap.add_argument('--kinematics', default='svk')
ap.add_argument('--frame', default='global')
ap.add_argument('--n-inc', type=int, default=None)
ap.add_argument('--tag', default=None)
a = ap.parse_args()
tag = a.tag or a.kinematics
os.makedirs(a.outdir, exist_ok=True)

t0 = time.perf_counter()
m = TPeelModel(a.deck, kinematics=a.kinematics, frame=a.frame)
t_setup = time.perf_counter() - t0
st = m.deck.step['static']
n_inc = a.n_inc or int(round(st['T'] / st['dtmax']))
print('deck %s: n_q %d free %d colours %d law %s, setup %.1f s, %d increments'
      % (a.deck, m.n_q, len(m.free), m.n_colors, m.law_kind, t_setup, n_inc), flush=True)
D = jnp.zeros((m.coh.n_seg, m.coh.n_q))
q = m.apply_bc(m.zero_q(), 0.0)
rows, Us = [], []
for k in range(1, n_inc + 1):
    lam = k / n_inc
    q = m.apply_bc(q, lam)
    t1 = time.perf_counter()
    q, info = newton(m, q, D, log=print)
    wall = time.perf_counter() - t1
    o = m.outputs(q, D)
    o.update(lam=lam, time=lam * st['T'], energy=m.energy(q, D), iterations=info['iterations'],
             converged=info['converged'], wall=wall, t_tangent=info['t_tangent'], t_solve=info['t_solve'])
    rows.append(o)
    Us.append(m.lift(q)[: 2 * m.n_nodes].reshape(-1, 2))
    print('inc %d lam %.3f RF2 %.9g UR3 %.9g mouth %.9g/%.9g its %d conv %s wall %.1f s (tangent %.1f, solve %.1f)'
          % (k, lam, o['rp1_rf2'], o['rp1_ur3'], o['mouth_u_u2'], o['mouth_l_u2'], info['iterations'],
             info['converged'], wall, info['t_tangent'], info['t_solve']), flush=True)
with open(os.path.join(a.outdir, 's3a_jax_%s.csv' % tag), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.12g' % v if isinstance(v, float) else v) for k, v in r.items()})
np.savez_compressed(os.path.join(a.outdir, 's3a_jax_%s_U.npz' % tag), labels=m.deck.labels,
                    lam=np.array([r['lam'] for r in rows]), U=np.array(Us))
print('total %.1f s (setup %.1f s)' % (time.perf_counter() - t0, t_setup))
