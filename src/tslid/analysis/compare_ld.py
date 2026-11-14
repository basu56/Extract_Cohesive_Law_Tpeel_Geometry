#!/usr/bin/env python3
"""
compare_ld.py -- compare load-displacement records of runs that differ in one parameter.

    python3 compare_ld.py <rundir> --jobs a,b,c --ref b [--context x,y] [--out prefix]

For each job vs the reference (both from <job>_response.csv, <job>.json, <job>_profiles.npz):
    P_peak, delta_peak, and their change relative to the reference
    slope0      initial stiffness (secant to 20 % of P_peak), relative change
    dP_post_max max |P - P_ref| / P_peak,ref over the common post-peak window
    dP_post_rms rms of the same
The common window ends at the smallest delta (over all jobs) at which the traction-free tip
reaches L - 15 mm (end effect excluded, as in energy_checks.py). The post-peak window starts at
1.2 x the largest delta_peak. Writes <out>_summary.csv and <out>.png/.pdf
(a: P/W vs delta; b: (P - P_ref)/P_peak,ref vs delta). Needs numpy, matplotlib.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tslid.laws import AlphaLaw  # noqa: E402
from tslid.analysis.energy_checks import read_csv, zone_fronts  # noqa: E402


def load(rundir, job):
    p = json.load(open(os.path.join(rundir, job + '.json')))['params']
    r = read_csv(os.path.join(rundir, job + '_response.csv'))
    law = AlphaLaw(p['K0'], p['sigc'], p['Gc'], p['alpha'])
    xf, _ = zone_fronts(rundir, job, p, law)
    hit = np.nonzero(xf > p['L'] - 15.0)[0]
    d_end = r['delta'][hit[0]] if len(hit) else r['delta'][-1]
    return dict(job=job, p=p, d=r['delta'], P=r['P'], d_end=float(d_end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rundir')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--context', default='', help='jobs drawn thin grey (e.g. outside the bounds)')
    ap.add_argument('--label', default='sigc', help='parameter that varies (for labels)')
    ap.add_argument('--out', default='ld_compare')
    o = ap.parse_args()
    jobs = o.jobs.split(',')
    ctx = [j for j in o.context.split(',') if j]
    R = {j: load(o.rundir, j) for j in jobs + ctx}
    ref = R[o.ref]
    ipk = int(np.argmax(ref['P']))
    Ppk_ref = ref['P'][ipk]
    d_hi = min(R[j]['d_end'] for j in jobs)
    d_lo = 1.2 * max(R[j]['d'][int(np.argmax(R[j]['P']))] for j in jobs)
    grid = np.linspace(0.0, d_hi, 2000)
    Pref = np.interp(grid, ref['d'], ref['P'])
    post = grid >= d_lo
    rows = []
    for j in jobs:
        x = R[j]
        i = int(np.argmax(x['P']))
        Pg = np.interp(grid, x['d'], x['P'])
        k = np.nonzero(x['P'] >= 0.2 * x['P'][i])[0][0]
        slope = x['P'][k] / x['d'][k]
        kr = np.nonzero(ref['P'] >= 0.2 * Ppk_ref)[0][0]
        slope_ref = ref['P'][kr] / ref['d'][kr]
        dp = (Pg - Pref) / Ppk_ref
        rows.append(dict(job=j, param=x['p'][o.label], P_peak=float(x['P'][i]), delta_peak=float(x['d'][i]),
                         dP_peak_rel=float(x['P'][i] / Ppk_ref - 1), d_delta_peak_rel=float(x['d'][i] / ref['d'][ipk] - 1),
                         slope0_rel=float(slope / slope_ref - 1),
                         dP_post_max=float(np.max(np.abs(dp[post]))) if post.any() else np.nan,
                         dP_post_rms=float(np.sqrt(np.mean(dp[post] ** 2))) if post.any() else np.nan,
                         window=' %.3f-%.3f mm' % (d_lo, d_hi)))
    keys = list(rows[0].keys())
    with open(os.path.join(o.rundir, o.out + '_summary.csv'), 'w') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print('reference %s (P_peak %.1f N); post-peak window %.3f-%.3f mm' % (o.ref, Ppk_ref, d_lo, d_hi))
    print('%-14s %6s %8s %8s %8s %8s %9s %9s' % ('job', o.label, 'P_peak', 'dPpk', 'd_dpk', 'dslope0', 'post_max', 'post_rms'))
    for r in rows:
        print('%-14s %6g %8.1f %+7.2f%% %+7.2f%% %+7.2f%% %8.2f%% %8.2f%%' % (
            r['job'], r['param'], r['P_peak'], 100 * r['dP_peak_rel'], 100 * r['d_delta_peak_rel'],
            100 * r['slope0_rel'], 100 * r['dP_post_max'], 100 * r['dP_post_rms']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    W = ref['p']['W']
    fig, ax = plt.subplots(1, 2, figsize=(29 / 2.54, 11.8 / 2.54))
    vals = [R[j]['p'][o.label] for j in jobs]
    cmap = cm.get_cmap('Blues')
    col = {j: cmap(0.35 + 0.6 * (v - min(vals)) / (max(vals) - min(vals) or 1)) for j, v in zip(jobs, vals)}
    for j in ctx:
        ax[0].plot(R[j]['d'], R[j]['P'] / W, color='0.7', lw=0.8, ls='--')
    for j in jobs:
        x = R[j]
        lw = 2.4 if j == o.ref else 1.4
        ax[0].plot(x['d'], x['P'] / W, color=col[j], lw=lw, label=r'$\sigma_c = %g$ MPa' % x['p'][o.label])
        Pg = np.interp(grid, x['d'], x['P'])
        ax[1].plot(grid, 100 * (Pg - Pref) / Ppk_ref, color=col[j], lw=lw)
    ax[0].axvline(d_hi, color='0.55', lw=0.8)
    ax[0].set_xlim(0, 1.15 * d_hi)
    ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r'$\delta$ (mm)', fontsize=14)
    ax[0].set_ylabel(r'$P/W$ (N/mm)', fontsize=14)
    ax[0].legend(frameon=False, fontsize=11)
    ax[1].axhline(0, color='0.55', lw=0.8)
    ax[1].set_xlim(0, d_hi)
    ax[1].set_xlabel(r'$\delta$ (mm)', fontsize=14)
    ax[1].set_ylabel(r'$(P - P_{\rm ref})/P_{\rm peak,ref}$ (%)', fontsize=14)
    for a, t in zip(ax, '(a)(b)'.replace(')(', ') (').split()):
        a.tick_params(direction='in', labelsize=12)
        a.text(-0.16, 1.0, t, transform=a.transAxes, fontsize=16, va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(o.rundir, o.out + '.png'), dpi=220)
    fig.savefig(os.path.join(o.rundir, o.out + '.pdf'))


if __name__ == '__main__':
    main()
