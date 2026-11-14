"""
extract_tpeel_odb.py -- per-frame response of a build_tpeel_inp.py run.

    abaqus python extract_tpeel_odb.py <job>.odb [--thresholds 0.005,0.01,0.02,0.05]

Writes <job>_response.csv with one row per field frame:
    time, delta (U2 top RP - U2 bottom RP), P (RF2 top RP, N),
    ell_sdeg1   end of the fully damaged zone (SDEG >= 0.99, tip layer)
    ell_sdeg0   leading edge of the process zone (SDEG > 0, tip layer)
    ell_br1     end of the fully damaged bridging layer (bridging law only)
    ell_hump1   end of the fully damaged hump layer (three-layer law only)
    ell_smax    position of the peak normal traction in the tip layer (the sigma_c front)
    delta_mouth opening of the groove faces at the crack mouth, x = 0
    ell_open_<d> crack length from the bond-face opening, first x past the notch
                 where opening < d (mm), linear interpolation -- the DIC-style measure
    W_ext       external work, integral of P d(delta)
    ALLSE, ALLPD, ALLDMD, ALLVD, ALLDMD_TIP, ALLDMD_BR (from history, interpolated)
and <job>_profiles.npz with, for every frame, the opening along the bond faces and,
for each cohesive layer present (tip, br, hump), the normal traction and damage per
element (traction_<layer>, sdeg_<layer>, x_<layer>), for bridging-zone analysis.
"""
import json
import os
import sys

import numpy as np
from odbAccess import openOdb


def find_set(odb, name, kind):
    asm = odb.rootAssembly
    pools = [getattr(asm, kind)] + [getattr(i, kind) for i in asm.instances.values()]
    for pool in pools:
        for k in pool.keys():
            if k.upper() == name.upper():
                return pool[k]
    return None


def members(seq):
    """Set members: assembly-level sets nest them per instance, instance-level sets do not."""
    first = seq[0]
    return first if hasattr(first, '__len__') and not hasattr(first, 'label') else seq


def history(step, key_part, var):
    for rname, reg in step.historyRegions.items():
        if key_part in rname and var in reg.historyOutputs.keys():
            d = np.array(reg.historyOutputs[var].data)
            return d[:, 0], d[:, 1]
    return None, None


def node_values(frame, var, nset, comp):
    sub = frame.fieldOutputs[var].getSubset(region=nset)
    return {v.nodeLabel: v.data[comp] for v in sub.values}


def main():
    args = sys.argv[1:]
    odb_path = args[0]
    th = [0.0143, 0.02, 0.04, 0.0838, 0.15, 0.25, 0.005, 0.01, 0.05]
    if '--thresholds' in args:
        th = [float(v) for v in args[args.index('--thresholds') + 1].split(',')]
    job = os.path.splitext(odb_path)[0]
    meta = json.load(open(job + '.json')) if os.path.exists(job + '.json') else {}
    t_bond = meta.get('params', {}).get('t', 0.4)
    x_notch = meta.get('params', {}).get('x_notch', 17.0)

    odb = openOdb(odb_path, readOnly=True)
    step = odb.steps.values()[-1]
    inst = odb.rootAssembly.instances.values()[0]
    coords = {n.label: n.coordinates for n in inst.nodes}

    top, bot = find_set(odb, 'IFACE_U', 'nodeSets'), find_set(odb, 'IFACE_L', 'nodeSets')
    rp1, rp2 = find_set(odb, 'RP1', 'nodeSets'), find_set(odb, 'RP2', 'nodeSets')
    tip = find_set(odb, 'COH_TIP', 'elementSets')
    br = find_set(odb, 'COH_BR', 'elementSets')
    hump = find_set(odb, 'COH_HUMP', 'elementSets')

    # bond-face node pairs ordered by x
    top_ids = sorted((coords[n.label][0], n.label) for n in members(top.nodes))
    bot_ids = sorted((coords[n.label][0], n.label) for n in members(bot.nodes))
    xs = np.array([x for x, _ in top_ids])

    def elem_x(eset):
        out = {}
        for e in members(eset.elements):
            out[e.label] = np.mean([coords[n][0] for n in e.connectivity])
        return out

    tip_x = elem_x(tip)
    br_x = elem_x(br) if br is not None else {}
    layers = [(k, es, elem_x(es)) for k, es in (('tip', tip), ('br', br), ('hump', hump)) if es is not None]
    x0 = min(tip_x.values()) - 0.5 * (sorted(tip_x.values())[1] - sorted(tip_x.values())[0])

    def per_element(frame, var, eset, ex, comp, agg):
        """Field reduced to one value per element (agg over integration points), sorted by x."""
        acc = {}
        for v in frame.fieldOutputs[var].getSubset(region=eset).values:
            acc.setdefault(v.elementLabel, []).append(v.data if comp is None else v.data[comp])
        lab = sorted(acc, key=lambda k: ex[k])
        return np.array([ex[k] for k in lab]), np.array([agg(acc[k]) for k in lab])

    def front(frame, eset, ex, level, strict):
        xe, s = per_element(frame, 'SDEG', eset, ex, None, max)
        hit = s > level if strict else s >= level
        if not hit[0]:
            return x0
        k = np.argmin(hit) if not hit.all() else len(hit)    # first element not past level
        h = xe[1] - xe[0]
        return xe[k - 1] + 0.5 * h

    rows, prof_open, prof_trac, prof_trac_br = [], [], [], []
    prof = {k: {'s22': [], 'sdeg': []} for k, _, _ in layers}
    for fr in step.frames:
        u2_top = node_values(fr, 'U', top, 1)
        u2_bot = node_values(fr, 'U', bot, 1)
        opening = np.array([u2_top[a] - u2_bot[b] for (_, a), (_, b) in zip(top_ids, bot_ids)])
        r1u = node_values(fr, 'U', rp1, 1)
        r2u = node_values(fr, 'U', rp2, 1)
        r1f = node_values(fr, 'RF', rp1, 1)
        row = dict(time=fr.frameValue,
                   delta=list(r1u.values())[0] - list(r2u.values())[0],
                   P=list(r1f.values())[0])
        # opening at the crack mouth (x = 0, groove faces), the DIC CMOD candidate
        row['delta_mouth'] = opening[0] if xs[0] < 1e-6 else np.nan
        if 'SDEG' in fr.fieldOutputs.keys():
            row['ell_sdeg1'] = front(fr, tip, tip_x, 0.99, False)
            row['ell_sdeg0'] = front(fr, tip, tip_x, 0.0, True)
            if br is not None:
                row['ell_br1'] = front(fr, br, br_x, 0.99, False)
            if hump is not None:
                row['ell_hump1'] = front(fr, hump, layers[-1][2], 0.99, False)
        for d in th:
            m = (xs >= x_notch - 1e-9)
            xo, op = xs[m], opening[m]
            below = np.nonzero(op < d)[0]
            if len(below) == 0:
                ell = xo[-1]
            elif below[0] == 0:
                ell = xo[0]
            else:
                k = below[0]
                ell = xo[k - 1] + (op[k - 1] - d) / (op[k - 1] - op[k]) * (xo[k] - xo[k - 1])
            row['ell_open_%g' % d] = ell
        rows.append(row)
        prof_open.append(opening)
        if 'S' in fr.fieldOutputs.keys():
            # cohesive S carries the solid layout (S11, S22, S33, S12); normal traction is S22
            xe, s22 = per_element(fr, 'S', tip, tip_x, 1, np.mean)
            prof_trac.append(s22)
            row['ell_smax'] = xe[np.argmax(s22)]           # peak-traction front of the tip law
            row['smax_tip'] = s22.max()
            if br is not None:
                xb, s22b = per_element(fr, 'S', br, br_x, 1, np.mean)
                prof_trac_br.append(s22b)
            for k, es, ex in layers:
                prof[k]['s22'].append(per_element(fr, 'S', es, ex, 1, np.mean)[1])
                if 'SDEG' in fr.fieldOutputs.keys():
                    prof[k]['sdeg'].append(per_element(fr, 'SDEG', es, ex, None, max)[1])

    t = np.array([r['time'] for r in rows])
    P = np.array([r['P'] for r in rows])
    dl = np.array([r['delta'] for r in rows])
    W_ext = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(dl))])
    for r, w in zip(rows, W_ext):
        r['W_ext'] = w
    for var, part, col in (('ALLSE', 'Assembly', 'ALLSE'), ('ALLPD', 'Assembly', 'ALLPD'),
                           ('ALLDMD', 'Assembly', 'ALLDMD'), ('ALLVD', 'Assembly', 'ALLVD'),
                           ('ALLDMD', 'COH_TIP', 'ALLDMD_TIP'), ('ALLDMD', 'COH_BR', 'ALLDMD_BR'),
                           ('ALLDMD', 'COH_HUMP', 'ALLDMD_HUMP')):
        ht, hv = history(step, part, var)
        if ht is not None:
            iv = np.interp(t, ht, hv)
            for r, v in zip(rows, iv):
                r[col] = v

    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = job + '_response.csv'
    with open(out, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join('%.8g' % r.get(k, np.nan) for k in keys) + '\n')
    np.savez_compressed(job + '_profiles.npz', x_face=xs, opening=np.array(prof_open),
                        time=t, t_bond=t_bond,
                        **{n: v for k, _, ex in layers for n, v in
                           (('x_' + k, np.array(sorted(ex.values()))),
                            ('traction_' + k, np.array(prof[k]['s22'])),
                            ('sdeg_' + k, np.array(prof[k]['sdeg'])))})
    print('wrote %s (%d frames) and %s_profiles.npz' % (out, len(rows), job))
    odb.close()


if __name__ == '__main__':
    main()
