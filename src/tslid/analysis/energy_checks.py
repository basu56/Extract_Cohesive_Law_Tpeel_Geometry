#!/usr/bin/env python3
"""
energy_checks.py -- simulation-level energy constraints (RESEARCH_PLAN.md section 4b)
and process-zone length for T-peel runs of build_tpeel_inp.py + extract_tpeel_odb.py.

    python3 energy_checks.py <rundir> [--pattern 'pz_*'] [--out summary.csv]

Per run (<job>.json + <job>_response.csv):
    E4  E_bal   = max over frames (W_ext > 5 % of final) of
                  |W_ext - ALLSE - ALLPD - ALLDMD - ALLVD| / W_ext            tol 1 %
    E5  vd_rel  = ALLVD / ALLDMD at the last frame                            tol 1 %
    E6  diss_rel = slope of (ALLDMD + ALLVD) vs crack length over steady propagation,
                  / (Gc W) - 1                                                tol 2 %
    l_pz        = mean (ell_sdeg0 - ell_sdeg1) over steady propagation (mm), l_pz/H,
                  and elements per process zone l_pz / h_coh
    P_peak, delta_peak, completed (from .sta)
Steady window: crack tip ell_sdeg1 in [a0 + 5 mm, L - 15 mm] (needs numpy).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

TOL = dict(E4=0.01, E5=0.01, E6=0.02)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan for r in rows])
            for k in rows[0]}


def completed(rundir, job):
    try:
        return 'COMPLETED SUCCESSFULLY' in open(os.path.join(rundir, job + '.sta')).read()
    except IOError:
        return False


def check(rundir, job):
    meta = json.load(open(os.path.join(rundir, job + '.json')))
    p, lw = meta['params'], meta['law']['TIP']
    out = dict(job=job, Gc=lw['G'], sigc=lw['sig'], alpha=lw.get('alpha', 0.0),
               phi_e=lw.get('phi_e', np.nan), h_coh=p['h_coh'], completed=completed(rundir, job))
    path = os.path.join(rundir, job + '_response.csv')
    if not os.path.exists(path):
        out['status'] = 'no response csv'
        return out
    r = read_csv(path)
    W, lam = p['W'], np.zeros_like(r['W_ext'])
    stored = r['ALLSE'] + r.get('ALLPD', lam) + r['ALLDMD'] + r['ALLVD']
    Wx = r['W_ext']
    m = Wx > 0.05 * Wx[-1]
    out['E_bal'] = float(np.nanmax(np.abs(Wx[m] - stored[m]) / Wx[m])) if m.any() else np.nan
    out['vd_rel'] = float(r['ALLVD'][-1] / r['ALLDMD'][-1]) if r['ALLDMD'][-1] > 0 else np.nan
    i = int(np.nanargmax(r['P']))
    out.update(P_peak=float(r['P'][i]), delta_peak=float(r['delta'][i]), delta_end=float(r['delta'][-1]))
    ell1, ell0 = r['ell_sdeg1'], r['ell_sdeg0']
    s = (ell1 >= p['a0'] + 5.0) & (ell1 <= p['L'] - 15.0) & np.isfinite(ell0)
    out['n_steady'] = int(s.sum())
    out['ell_end'] = float(np.nanmax(ell1)) if np.isfinite(ell1).any() else np.nan
    if s.sum() >= 5:
        D = r['ALLDMD'][s] + r['ALLVD'][s]
        slope = np.polyfit(ell1[s], D, 1)[0]
        out['diss_rel'] = float(slope / (lw['G'] * W) - 1.0)
        lpz = float(np.mean(ell0[s] - ell1[s]))
        out.update(l_pz=lpz, l_pz_H=lpz / p['H'], n_el_pz=lpz / p['h_coh'],
                   l_pz_std=float(np.std(ell0[s] - ell1[s])))
    out['pass_E4'] = bool(out['E_bal'] <= TOL['E4'])
    out['pass_E5'] = bool(out['vd_rel'] <= TOL['E5'])
    out['pass_E6'] = bool(abs(out.get('diss_rel', np.inf)) <= TOL['E6'])
    out['status'] = 'ok'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rundir')
    ap.add_argument('--pattern', default='*')
    ap.add_argument('--out', default='energy_summary.csv')
    o = ap.parse_args()
    jobs = sorted(os.path.basename(j)[:-5] for j in glob.glob(os.path.join(o.rundir, o.pattern + '.json')))
    res = [check(o.rundir, j) for j in jobs]
    keys = ['job', 'Gc', 'sigc', 'alpha', 'phi_e', 'h_coh', 'completed', 'status', 'P_peak', 'delta_peak',
            'delta_end', 'ell_end', 'n_steady', 'l_pz', 'l_pz_std', 'l_pz_H', 'n_el_pz', 'E_bal', 'vd_rel',
            'diss_rel', 'pass_E4', 'pass_E5', 'pass_E6']
    with open(os.path.join(o.rundir, o.out), 'w') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        for x in res:
            w.writerow(x)
    fmt = '%-22s %5s %5s %5s %5s %8s %7s %7s %6s %8s %8s %8s %s'
    print(fmt % ('job', 'Gc', 'sigc', 'alpha', 'phi_e', 'P_peak', 'l_pz/H', 'n_el', 'done', 'E_bal', 'vd_rel',
                 'diss_rel', 'status'))
    for x in res:
        g = lambda k, f='%.3g': (f % x[k]) if isinstance(x.get(k), (float, int)) and not isinstance(x.get(k), bool) else str(x.get(k, '-'))
        print(fmt % (x['job'], g('Gc'), g('sigc'), g('alpha'), g('phi_e', '%.2f'), g('P_peak', '%.1f'),
                     g('l_pz_H'), g('n_el_pz', '%.1f'), x['completed'], g('E_bal', '%.1e'),
                     g('vd_rel', '%.1e'), g('diss_rel', '%+.1e'), x['status']))


if __name__ == '__main__':
    main()
