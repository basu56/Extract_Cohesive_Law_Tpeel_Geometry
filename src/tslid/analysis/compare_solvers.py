#!/usr/bin/env python3
"""
compare_solvers.py -- one T-peel case computed by two solvers (ABAQUS reference vs tslid.fe), S3d.

    python3 compare_solvers.py <rundir_ref> <job_ref> <rundir_new> <job_new> [--out result.json]

Both runs must provide what extract_tpeel_odb.py writes (<job>.json, <job>_response.csv,
<job>_profiles.npz; <job>.sta for the 'completed' flag); tslid.fe writes the same files
(src/tslid/fe/tpeel_output.py).  Units: mm, N.

Reported (gates from docs/solver/S3_INSTRUCTIONS.md, S3d item 4):
  window       0 <= delta <= d_win, d_win = the smaller of the two compare_ld.py window ends
               (sigma_c front at L - 15 mm)
  max_dP       max |P_new - P_ref| / P_peak,ref over the window, evaluated at the union of both
               records' delta values in the window (each curve linearly interpolated), so that no
               recorded point is skipped; delta of the maximum            gate <= 0.5 %, target 0.1 %
  rms_dP       rms of the same on a uniform grid of 20000 points in the window
  dP_peak      (P_peak,new - P_peak,ref) / P_peak,ref                     gate |.| <= 0.1 %
  ddelta_peak  delta_peak,new - delta_peak,ref, and the increment there (larger of the two codes'
               delta steps around their peaks)                             gate |.| <= one increment
  slope        elastic slope: least-squares P/delta through the origin over the rising branch with
               P <= 0.2 P_peak;
               relative difference
  l_cz         energy_checks.py l_cz (mean over steady propagation), difference in mm and in
               elements (h_coh)                                            gate |.| <= 2 elements
  tip          traction-free crack tip x(delta) (energy_checks.zone_fronts): max |dx| over the
               window where both exist, mm and elements; same for the sigma_c front
  E4-E6        energy_checks.check for both runs                           gate: all pass, both
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tslid.laws import AlphaLaw  # noqa: E402
from tslid.analysis.compare_ld import load  # noqa: E402
from tslid.analysis.energy_checks import check, zone_fronts  # noqa: E402

GATES = dict(max_dP=0.005, dP_peak=0.001, l_cz_elements=2.0)
TARGET = dict(max_dP=0.001)


def peak(d, P):
    k = int(np.argmax(P))
    lo = d[k] - d[k - 1] if k > 0 else 0.0
    hi = d[k + 1] - d[k] if k + 1 < len(d) else 0.0
    return k, float(P[k]), float(d[k]), float(max(lo, hi))


def slope(d, P, Ppk):
    pre = np.arange(len(P)) < int(np.argmax(P))                  # rising branch only (not the tail)
    m = pre & (P <= 0.2 * Ppk) & (d > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.sum(P[m] * d[m]) / np.sum(d[m] ** 2))


def compare(rd_ref, job_ref, rd_new, job_new):
    a, b = load(rd_ref, job_ref), load(rd_new, job_new)
    p = a['p']
    h = float(p['h_coh'])
    d_win = min(a['d_end'], b['d_end'])
    ka, Pa, da, inc_a = peak(a['d'], a['P'])
    kb, Pb, db, inc_b = peak(b['d'], b['P'])
    out = dict(job_ref=job_ref, job_new=job_new, rundir_ref=os.path.abspath(rd_ref), rundir_new=os.path.abspath(rd_new),
               h_coh=h, d_window=float(d_win), d_window_ref=a['d_end'], d_window_new=b['d_end'],
               P_peak_ref=Pa, P_peak_new=Pb, delta_peak_ref=da, delta_peak_new=db)
    # whole curve: union of recorded deltas in the window
    du = np.union1d(a['d'][a['d'] <= d_win], b['d'][b['d'] <= d_win])
    e = (np.interp(du, b['d'], b['P']) - np.interp(du, a['d'], a['P'])) / Pa
    i = int(np.argmax(np.abs(e)))
    g = np.linspace(0.0, d_win, 20000)
    eg = (np.interp(g, b['d'], b['P']) - np.interp(g, a['d'], a['P'])) / Pa
    out.update(max_dP=float(np.abs(e[i])), max_dP_signed=float(e[i]), max_dP_delta=float(du[i]),
               rms_dP=float(np.sqrt(np.mean(eg ** 2))))
    pre = du <= da
    out['max_dP_prepeak'] = float(np.abs(e[pre]).max()) if pre.any() else np.nan
    out['max_dP_postpeak'] = float(np.abs(e[~pre]).max()) if (~pre).any() else np.nan
    out.update(dP_peak=(Pb - Pa) / Pa, ddelta_peak=db - da, increment_at_peak=max(inc_a, inc_b))
    sa, sb = slope(a['d'], a['P'], Pa), slope(b['d'], b['P'], Pa)
    out.update(slope_ref=sa, slope_new=sb, dslope=(sb - sa) / sa)
    # cohesive zone and crack tip
    ca, cb = check(rd_ref, job_ref), check(rd_new, job_new)
    for k in ('E_bal', 'E_bal_full', 'vd_rel', 'diss_rel', 'l_cz', 'l_cz_std', 'n_el_cz', 'n_steady',
              'pass_E4', 'pass_E5', 'pass_E6', 'completed'):
        out[k + '_ref'], out[k + '_new'] = ca.get(k), cb.get(k)
    out['dl_cz'] = float(cb.get('l_cz', np.nan) - ca.get('l_cz', np.nan))
    out['dl_cz_elements'] = out['dl_cz'] / h
    law = AlphaLaw(p['K0'], p['sigc'], p['Gc'], p['alpha'])
    fr = {}
    for nm, rd, job, x in (('ref', rd_ref, job_ref, a), ('new', rd_new, job_new, b)):
        xf, x0 = zone_fronts(rd, job, json.load(open(os.path.join(rd, job + '.json')))['params'], law)
        fr[nm] = (x['d'], xf, x0)
    for key, j in (('tip', 1), ('front', 2)):
        da_, xa = fr['ref'][0], fr['ref'][j]
        db_, xb = fr['new'][0], fr['new'][j]
        ok_a, ok_b = np.isfinite(xa), np.isfinite(xb)
        if ok_a.sum() < 2 or ok_b.sum() < 2:
            out['max_d' + key] = np.nan
            continue
        lo = max(da_[ok_a].min(), db_[ok_b].min())
        gg = du[(du >= lo) & (du <= d_win)]
        if len(gg) == 0:
            out['max_d' + key] = np.nan
            continue
        dx = np.interp(gg, db_[ok_b], xb[ok_b]) - np.interp(gg, da_[ok_a], xa[ok_a])
        k = int(np.argmax(np.abs(dx)))
        out.update(**{'max_d' + key: float(np.abs(dx[k])), 'max_d' + key + '_elements': float(np.abs(dx[k]) / h),
                      'max_d' + key + '_delta': float(gg[k]),
                      'rms_d' + key: float(np.sqrt(np.mean(dx ** 2)))})
    gates = dict(max_dP=out['max_dP'] <= GATES['max_dP'], dP_peak=abs(out['dP_peak']) <= GATES['dP_peak'],
                 ddelta_peak=abs(out['ddelta_peak']) <= out['increment_at_peak'] + 1e-12,
                 l_cz=abs(out['dl_cz_elements']) <= GATES['l_cz_elements'],
                 E4_E6=bool(ca.get('pass_E4') and ca.get('pass_E5') and ca.get('pass_E6') and
                            cb.get('pass_E4') and cb.get('pass_E5') and cb.get('pass_E6')))
    out['gates'] = gates
    out['all_gates_pass'] = bool(all(gates.values()))
    out['target_max_dP'] = out['max_dP'] <= TARGET['max_dP']
    return out


def fmt(o):
    L = ['%s (new) vs %s (ref); window 0-%.4f mm' % (o['job_new'], o['job_ref'], o['d_window']),
         '  whole curve: max |dP|/P_peak %.2e at delta %.4f (pre-peak %.2e, post-peak %.2e), rms %.2e   [gate 5e-3, target 1e-3]'
         % (o['max_dP'], o['max_dP_delta'], o['max_dP_prepeak'], o['max_dP_postpeak'], o['rms_dP']),
         '  peak: dP_peak %+.2e [gate 1e-3]; ddelta_peak %+.2e mm (increment %.2e)' % (o['dP_peak'], o['ddelta_peak'],
                                                                                    o['increment_at_peak']),
         '  elastic slope: %.6g vs %.6g N/mm (%+.2e)' % (o['slope_new'], o['slope_ref'], o['dslope']),
         '  l_cz: %.4f vs %.4f mm (%+.2f elements) [gate 2]' % (o['l_cz_new'], o['l_cz_ref'], o['dl_cz_elements']),
         '  crack tip: max |dx| %.4f mm (%.2f el.) at delta %.4f; sigma_c front max |dx| %.4f mm (%.2f el.)'
         % (o.get('max_dtip', np.nan), o.get('max_dtip_elements', np.nan), o.get('max_dtip_delta', np.nan),
            o.get('max_dfront', np.nan), o.get('max_dfront_elements', np.nan)),
         '  E4 %.1e / %.1e, E5 %.1e / %.1e, E6 %+.1e / %+.1e (new / ref)'
         % (o['E_bal_new'], o['E_bal_ref'], o['vd_rel_new'], o['vd_rel_ref'], o['diss_rel_new'], o['diss_rel_ref']),
         '  gates: %s -> %s; target 0.1 %%: %s' % (o['gates'], 'PASS' if o['all_gates_pass'] else 'FAIL',
                                                  o['target_max_dP'])]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rundir_ref')
    ap.add_argument('job_ref')
    ap.add_argument('rundir_new')
    ap.add_argument('job_new')
    ap.add_argument('--out', default=None)
    o = ap.parse_args()
    r = compare(o.rundir_ref, o.job_ref, o.rundir_new, o.job_new)
    print(fmt(r))
    if o.out:
        with open(o.out, 'w') as f:
            json.dump(r, f, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v))


if __name__ == '__main__':
    main()
