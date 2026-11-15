#!/usr/bin/env python3
"""
energy_checks.py -- simulation-level energy constraints (RESEARCH_PLAN.md section 4b)
and process-zone length for T-peel runs of build_tpeel_inp.py + extract_tpeel_odb.py.

    python3 energy_checks.py <rundir> [--pattern 'pz_*'] [--out summary.csv] [--window tip|front]

Per run (<job>.json + <job>_response.csv):
    E4  E_bal   = max over the analysis window (W_ext > 5 % of final, crack tip <= L - 15 mm) of
                  |W_ext - ALLSE - ALLPD - ALLDMD - ALLVD - ALLCD| / W_ext    tol 1 %
                  (E_bal_full: same over the whole record, incl. the unstable end-of-ligament failure)
    E5  vd_rel  = (ALLVD + ALLCD) / ALLDMD at the end of the window           tol 1 %
                  Runs without ALLCD output (pilot, before 2026-11-14 pm): E5 is bounded by E_bal.
    E6  diss_rel = slope of (ALLDMD + ALLVD) vs crack length over steady propagation,
                  / (Gc W) - 1                                                tol 2 %
    l_cz        = cohesive-zone length (DECISIONS.md D14): distance from the point where the
                  opening equals delta_0 (traction = sigc) to the point where it equals delta_f
                  (traction-free), from the nodal opening profile in <job>_profiles.npz with
                  linear interpolation between nodes; mean and std over steady propagation;
                  l_cz/H, l_ch/H (l_ch = E' Gc / sigc^2) and elements per l_cz.
                  (SDEG-threshold fronts are NOT used: biased, D14. l_pz is undefined, D14.)
    P_peak, delta_peak, completed (from .sta)
Steady window: traction-free tip (delta = delta_f) in [a0 + 5 mm, L - 15 mm] (needs numpy).
--window front (D26, cross-solver comparisons): the E4/E5 window and the E6 steady range end at the
first frame where the sigma_c front passes L - 15 mm, the end of the compare_ld.py load window.
The default (tip) is unchanged.
"""
import argparse
import csv
import glob
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tslid.laws import AlphaLaw  # noqa: E402

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


def last_crossing(x, y, level):
    """Largest x (scanning from the crack mouth) at which y >= level; linear interpolation
    to the next node. NaN if y never reaches level."""
    i = np.nonzero(y >= level)[0]
    if len(i) == 0:
        return np.nan
    k = i[-1]
    if k + 1 >= len(x):
        return float(x[k])
    return float(x[k] + (y[k] - level) / (y[k] - y[k + 1]) * (x[k + 1] - x[k]))


def zone_fronts(rundir, job, p, law):
    """Per frame: x of the traction-free tip (opening = df) and of the sigma_c point
    (opening = d0), from the bond-face opening profile."""
    z = np.load(os.path.join(rundir, job + '_profiles.npz'))
    x = z['x_face']
    sel = x >= p['a0'] - 1e-9
    xs = x[sel]
    xf = np.array([last_crossing(xs, o[sel], law.df) for o in z['opening']])
    x0 = np.array([last_crossing(xs, o[sel], law.d0) for o in z['opening']])
    return xf, x0


def check(rundir, job, window='tip'):
    """Energy checks of one run.  window='tip' (default, unchanged since D10/D14): E4/E5 window ends
    when the traction-free tip passes L - 15 mm.  window='front' (D26, cross-solver comparisons): it
    ends where compare_ld.py ends the load comparison, at the first frame where the sigma_c front
    passes L - 15 mm; the E6 steady range is cut there too."""
    meta = json.load(open(os.path.join(rundir, job + '.json')))
    p, lw = meta['params'], meta['law']['TIP']
    out = dict(job=job, Gc=lw['G'], sigc=lw['sig'], alpha=lw.get('alpha', 0.0),
               phi_e=lw.get('phi_e', np.nan), h_coh=p['h_coh'], completed=completed(rundir, job))
    path = os.path.join(rundir, job + '_response.csv')
    if not os.path.exists(path):
        out['status'] = 'no response csv'
        return out
    r = read_csv(path)
    law = AlphaLaw(p['K0'], p['sigc'], p['Gc'], p['alpha'])
    Eprime = p['E_al'] / (1.0 - p['nu_al'] ** 2)
    out['l_ch_H'] = law.l_ch(Eprime) / p['H']
    xf, x0 = zone_fronts(rundir, job, p, law)       # traction-free tip, sigma_c point
    W, lam = p['W'], np.zeros_like(r['W_ext'])
    visc = r['ALLVD'] + r.get('ALLCD', lam)        # ALLCD: cohesive viscous regularization (newer runs)
    stored = r['ALLSE'] + r.get('ALLPD', lam) + r['ALLDMD'] + visc
    Wx = r['W_ext']
    res = np.abs(Wx - stored) / np.where(Wx > 0, Wx, np.nan)
    m = Wx > 0.05 * Wx[-1]
    # E4 is evaluated over the analysis window: until the traction-free tip passes L - 15 mm
    # (frames before any traction-free point exists are included). The last ~2H of ligament
    # fails unstably (end effect); its residual is reported separately as E_bal_full.
    if window == 'front':
        hit = np.nonzero(x0 > p['L'] - 15.0)[0]
        before = np.arange(len(Wx)) < (hit[0] if len(hit) else len(Wx))
        win = m & before
    elif window == 'tip':
        before = np.ones(len(Wx), bool)
        win = m & ~(xf > p['L'] - 15.0)
    else:
        raise ValueError("window must be 'tip' or 'front'")
    out['window'] = window
    out['E_bal'] = float(np.nanmax(res[win])) if win.any() else np.nan
    out['E_bal_full'] = float(np.nanmax(res[m])) if m.any() else np.nan
    out['has_ALLCD'] = 'ALLCD' in r
    out['vd_rel'] = float(visc[win][-1] / r['ALLDMD'][win][-1]) if win.any() and r['ALLDMD'][win][-1] > 0 else np.nan
    i = int(np.nanargmax(r['P']))
    out.update(P_peak=float(r['P'][i]), delta_peak=float(r['delta'][i]), delta_end=float(r['delta'][-1]))
    ell1 = xf
    s = (ell1 >= p['a0'] + 5.0) & (ell1 <= p['L'] - 15.0) & np.isfinite(x0) & before
    out['n_steady'] = int(s.sum())
    out['ell_end'] = float(np.nanmax(ell1)) if np.isfinite(ell1).any() else np.nan
    if s.sum() >= 5:
        D = r['ALLDMD'][s] + r['ALLVD'][s]
        slope = np.polyfit(ell1[s], D, 1)[0]
        out['diss_rel'] = float(slope / (lw['G'] * W) - 1.0)
        lcz = x0[s] - xf[s]
        out.update(l_cz=float(lcz.mean()), l_cz_std=float(lcz.std()), l_cz_H=float(lcz.mean()) / p['H'],
                   n_el_cz=float(lcz.mean()) / p['h_coh'])
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
    ap.add_argument('--window', default='tip', choices=['tip', 'front'],
                    help="E4-E6 window end: traction-free tip (default) or sigma_c front (D26) at L - 15 mm")
    o = ap.parse_args()
    jobs = sorted(os.path.basename(j)[:-5] for j in glob.glob(os.path.join(o.rundir, o.pattern + '.json')))
    res = [check(o.rundir, j, o.window) for j in jobs]
    keys = ['job', 'Gc', 'sigc', 'alpha', 'phi_e', 'h_coh', 'completed', 'status', 'P_peak', 'delta_peak',
            'delta_end', 'ell_end', 'n_steady', 'l_ch_H', 'l_cz', 'l_cz_std', 'l_cz_H', 'n_el_cz', 'E_bal', 'E_bal_full', 'has_ALLCD', 'vd_rel',
            'diss_rel', 'pass_E4', 'pass_E5', 'pass_E6']
    with open(os.path.join(o.rundir, o.out), 'w') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        for x in res:
            w.writerow(x)
    fmt = '%-22s %5s %5s %5s %5s %8s %7s %7s %6s %8s %8s %8s %s'
    print(fmt % ('job', 'Gc', 'sigc', 'alpha', 'phi_e', 'P_peak', 'l_cz/H', 'n_el', 'done', 'E_bal', 'vd_rel',
                 'diss_rel', 'status'))
    for x in res:
        g = lambda k, f='%.3g': (f % x[k]) if isinstance(x.get(k), (float, int)) and not isinstance(x.get(k), bool) else str(x.get(k, '-'))
        print(fmt % (x['job'], g('Gc'), g('sigc'), g('alpha'), g('phi_e', '%.2f'), g('P_peak', '%.1f'),
                     g('l_cz_H'), g('n_el_cz', '%.1f'), x['completed'], g('E_bal', '%.1e'),
                     g('vd_rel', '%.1e'), g('diss_rel', '%+.1e'), x['status']))


if __name__ == '__main__':
    main()
