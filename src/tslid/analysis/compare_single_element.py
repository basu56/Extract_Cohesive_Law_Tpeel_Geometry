#!/usr/bin/env python3
"""
compare_single_element.py -- compare single-element ABAQUS responses with the analytic law.

    python3 compare_single_element.py <rundir> [--plot out.png]

For every <job>.json + <job>_se.csv in <rundir> reports:
    trac_err   max |sigma_FE - sigma_exact(d)| / sigc over the monotonic loading frames
    E_diss     final ALLDMD + ALLVD (N/mm, element area 1 mm^2) vs Gc        (E1 on ABAQUS)
    E_bal      |ALLWK - ALLSE - ALLDMD - ALLVD| / ALLWK at the last frame      (E4)
    unload     cyclic runs: max deviation from secant unloading, / sigc
Writes <rundir>/single_element_summary.csv.  Standard library only (plot needs matplotlib).
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tslid.laws import AlphaLaw  # noqa: E402


def load(rundir):
    for meta_path in sorted(glob.glob(os.path.join(rundir, 'se_*.json'))):
        job = os.path.basename(meta_path)[:-5]
        csv_path = os.path.join(rundir, job + '_se.csv')
        meta = json.load(open(meta_path))
        rows = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else None
        yield job, meta, rows


def analyse(job, meta, rows):
    lw = meta['law']
    law = AlphaLaw(lw['K'], lw['sigc'], lw['Gc'], lw['alpha'])
    out = dict(job=job, alpha=law.alpha, rep=meta['representation']['softening'],
               Gc=law.Gc, sigc=law.sigc, phi_e=law.phi_e)
    if not rows:
        out['status'] = 'NO RESULTS'
        return out
    d = [float(r['d']) for r in rows]
    s = [float(r['traction']) for r in rows]
    dmax_hist, err_mono, err_unl = 0.0, 0.0, 0.0
    for di, si in zip(d, s):
        if di >= dmax_hist - 1e-12:              # on the monotonic envelope
            dmax_hist = max(dmax_hist, di)
            err_mono = max(err_mono, abs(si - law.traction(di)))
        else:                                    # unloading / reloading: secant to origin
            sec = law.traction(dmax_hist) / dmax_hist if dmax_hist > law.d0 else law.K
            err_unl = max(err_unl, abs(si - sec * di))
    last = rows[-1]
    W, SE, DMD, VD = (float(last[k]) for k in ('ALLWK', 'ALLSE', 'ALLDMD', 'ALLVD'))
    out.update(status='ok', frames=len(rows), d_end=d[-1],
               trac_err=err_mono / law.sigc, unload_err=err_unl / law.sigc,
               E_diss=DMD + VD, E_diss_rel=(DMD + VD - law.Gc) / law.Gc,
               ALLVD_rel=VD / DMD if DMD else float('nan'),
               E_bal=abs(W - SE - DMD - VD) / W if W else float('nan'))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rundir')
    p.add_argument('--plot')
    o = p.parse_args()
    res = [(analyse(j, m, r), m, r) for j, m, r in load(o.rundir)]
    keys = ['job', 'rep', 'alpha', 'Gc', 'sigc', 'phi_e', 'status', 'frames', 'trac_err',
            'unload_err', 'E_diss', 'E_diss_rel', 'ALLVD_rel', 'E_bal']
    with open(os.path.join(o.rundir, 'single_element_summary.csv'), 'w') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        for r, _, _ in res:
            w.writerow(r)
    for r, _, _ in res:
        print(' '.join(('%s=%.3g' % (k, r[k]) if isinstance(r.get(k), float) else '%s=%s' % (k, r.get(k)))
                       for k in keys if k in r))
    if o.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r, m, rows in res:
            if not rows or 'cyc' in r['job'] or 'hi' in r['job']:
                continue
            lw = m['law']
            law = AlphaLaw(lw['K'], lw['sigc'], lw['Gc'], lw['alpha'])
            dd = [law.df * 1.1 * i / 400 for i in range(401)]
            if r['rep'] != 'tabular':
                ax.plot(dd, [law.traction(x) for x in dd], 'k-', lw=0.8)
            ax.plot([float(x['d']) for x in rows][::20], [float(x['traction']) for x in rows][::20],
                    'o' if r['rep'] == 'tabular' else 's', ms=3, mfc='none',
                    label='%s alpha=%g' % (r['rep'], r['alpha']))
        ax.set_xlabel('separation (mm)')
        ax.set_ylabel('traction (MPa)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(o.plot, dpi=200)


if __name__ == '__main__':
    main()
