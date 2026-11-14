#!/usr/bin/env python3
# Imported from basu56/TPeel_cohesive_zone@742e1ae (scripts/build_tpeel_inp.py); changes for this
# project: --law alpha (src/tslid/laws.py, energy checks E1-E3), D11 defaults H=6.324,
# a0=18, elastic adherends (--plastic to re-enable).
"""
build_tpeel_inp.py -- parametric 2D plane-strain T-peel deck for ABAQUS/Standard.

Writes a flat (no *Part) input file: two L-shaped Al 6063-T6 adherends joined by
one row of COH2D4 cohesive elements spanning the bond line, loaded through
reference points coupled to the leg end faces.

Geometry (mm), symmetric about y = 0:
    upper bond face y = +t/2, lower bond face y = -t/2
    arm      x in [0, L],  thickness H
    leg      x in [0, H],  rising leg_h beyond the arm
    groove   depth g on each bond face for x < x_n - taper, tapering to the
             bond face at the notch tip x_n
    crack    free faces for x < a0 (notch + precrack), cohesive for x in [a0, L]

Cohesive laws (--law):
    elastic     linear traction-separation only (compliance runs)
    alpha       Phase 1 shape family of src/tslid/laws.py: K0, --sigc, --Gc, --alpha,
                --softening linear|exponential|tabular (auto: linear if alpha = 0)
    bilinear    one layer: K0, sig_tip, G_tip
    bridging    two coincident layers sharing nodes (Davila, Rose & Camanho 2009
                superposition): tip layer (sig_tip, G_tip) + bridging layer
                (sig_br, G_br). K0 is split in proportion to strength so both
                layers initiate at the same separation d0 = (sig_tip+sig_br)/K0,
                unless --K-br is given (bridging hump peaking at sig_br/K_br).
                --sig-hump/--G-hump/--K-hump add a third layer with its own stiffness.

Constitutive thickness is 1.0, so nominal strain = separation and K0 is in MPa/mm
(K0 = E_a / t is the physical choice). Section thicknesses are W, so reaction
forces are total forces in N and energies are total energies in N*mm.

Usage:
    python3 build_tpeel_inp.py --job tp_null --law bilinear --G-tip 0.40 --sig-tip 60
A JSON file with every parameter is written next to the .inp.
"""
import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tslid.laws import AlphaLaw  # noqa: E402

# Al 6063-T6, Ludwik A=245, B=402, n=0.628 (docs/campaign01/02_material.md)
AL6063_PLASTIC = [
    (245.0000, 0.00000), (248.3977, 0.00050), (250.2508, 0.00100), (253.1147, 0.00200),
    (259.4271, 0.00500), (267.2960, 0.01000), (273.7615, 0.01500), (279.4566, 0.02000),
    (289.4486, 0.03000), (298.2500, 0.04000), (306.2603, 0.05000), (313.6918, 0.06000),
    (327.2937, 0.08000), (339.6730, 0.10000), (351.1578, 0.12000), (361.9484, 0.14000),
    (372.1784, 0.16000), (381.9422, 0.18000), (391.3096, 0.20000), (400.3343, 0.22000),
    (409.0585, 0.24000), (417.5160, 0.26000), (425.7346, 0.28000), (433.7375, 0.30000),
    (441.5442, 0.32000), (452.9222, 0.35000),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a('--job', required=True, help='job name; writes <outdir>/<job>.inp')
    a('--outdir', default='.')
    # geometry
    a('--H', type=float, default=6.324, help='adherend thickness = leg width (D11)')
    a('--t', type=float, default=0.4, help='bond-line thickness')
    a('--L', type=float, default=76.2, help='specimen length')
    a('--W', type=float, default=15.0, help='specimen width (out of plane)')
    a('--leg-h', type=float, default=15.4, help='leg length beyond the arm')
    a('--groove', type=float, default=1.5, help='notch groove depth on each face')
    a('--taper', type=float, default=2.3, help='x-length of groove taper')
    a('--x-notch', type=float, default=17.0, help='notch tip, measured from x = 0')
    a('--a0', type=float, default=18.0, help='initial crack tip (notch + precrack) (D11)')
    # mesh
    a('--h-coh', type=float, default=0.1, help='element size along the bond for x >= x_notch')
    a('--h-far', type=float, default=0.5, help='element size along x for x < x_notch - taper')
    a('--ny', type=int, default=20, help='elements through each arm thickness')
    a('--h-y0', type=float, default=0.1, help='first element height at the bond face')
    a('--h-leg', type=float, default=0.5, help='element size along the leg')
    a('--eltype', default='CPE4I')
    # materials
    a('--E-al', type=float, default=70000.0)
    a('--nu-al', type=float, default=0.33)
    a('--no-plastic', dest='no_plastic', action='store_true', default=True,
      help='elastic adherends (default, D11)')
    a('--plastic', dest='no_plastic', action='store_false', help='Al 6063-T6 plasticity')
    a('--law', choices=['elastic', 'bilinear', 'bridging', 'alpha'], default='alpha')
    a('--sigc', type=float, default=20.0, help='alpha law: cohesive strength, MPa')
    a('--Gc', type=float, default=0.40, help='alpha law: fracture energy, N/mm')
    a('--alpha', type=float, default=0.0, help='alpha law: softening shape parameter')
    a('--softening', choices=['auto', 'linear', 'exponential', 'tabular'], default='auto')
    a('--n-tab', type=int, default=1000, help='alpha law: rows of the TABULAR damage table')
    a('--K0', type=float, default=8500.0, help='total normal stiffness, MPa/mm (E_a/t)')
    a('--sig-tip', type=float, default=60.0)
    a('--G-tip', type=float, default=0.40)
    a('--sig-br', type=float, default=10.0)
    a('--G-br', type=float, default=0.80)
    a('--K-br', type=float, default=None,
      help='bridging-layer stiffness, MPa/mm. Default: Davila split of K0 (both layers peak at the '
           'same d0). Given explicitly, the tip layer keeps K0 and the bridging layer peaks at '
           'd0_br = sig_br/K_br -- a bridging hump engaging at large opening')
    a('--sig-hump', type=float, default=None, help='optional third layer (bridging law only): peak traction')
    a('--G-hump', type=float, default=None)
    a('--K-hump', type=float, default=None, help='third-layer stiffness; peaks at sig_hump/K_hump')
    a('--viscosity', type=float, default=1e-3, help='cohesive viscous regularization, s')
    # loading
    a('--bc-bottom', choices=['pin', 'clamp'], default='clamp',
      help='pin: RP U1=U2=0; clamp: RP encastre (same as pinning every node of the '
           'lower leg end face, since the coupling keeps that face rigid)')
    a('--bc-top', choices=['load', 'pin', 'clamp'], default='load',
      help='load: RP U2 prescribed only; pin: also U1=0; clamp: also U1=UR3=0')
    a('--delta', type=float, default=5.0, help='prescribed U2 of the top RP, mm')
    a('--rate', type=float, default=1.0, help='crosshead rate, mm/min (sets step time)')
    a('--nlgeom', choices=['YES', 'NO'], default='YES')
    a('--n-inc', type=int, default=400, help='target increments (sets max increment)')
    a('--field-every', type=int, default=1, help='field output frequency for the bond, increments')
    a('--field-adh-every', type=int, default=10, help='field output frequency for adherend S, PEEQ')
    return p.parse_args(argv)


def check_law(o):
    """Bilinear admissibility: onset separation must sit below failure separation."""
    if o.law == 'elastic':
        return {}
    if o.law == 'alpha':
        try:
            lw = AlphaLaw(o.K0, o.sigc, o.Gc, o.alpha)
            lw.check_energy_constraints()                       # E2, E3
            cards, rep = lw.abaqus_cards(o.softening, o.n_tab)  # E1
        except ValueError as e:
            sys.exit('%s: law rejected -- %s' % (o.job, e))
        d = dict(sig=o.sigc, G=o.Gc, K=o.K0, d0=lw.d0, df=lw.df, alpha=o.alpha,
                 phi_e=lw.phi_e, representation=rep)
        d['cards'] = cards
        return {'TIP': d}
    if o.law == 'bilinear':
        layers = {'TIP': (o.sig_tip, o.G_tip, o.K0)}
    else:
        if o.K_br is None:
            s = o.sig_tip + o.sig_br
            layers = {'TIP': (o.sig_tip, o.G_tip, o.K0 * o.sig_tip / s),
                      'BR': (o.sig_br, o.G_br, o.K0 * o.sig_br / s)}
        else:
            layers = {'TIP': (o.sig_tip, o.G_tip, o.K0), 'BR': (o.sig_br, o.G_br, o.K_br)}
        if o.sig_hump is not None:
            layers['HUMP'] = (o.sig_hump, o.G_hump, o.K_hump)
    out = {}
    for k, (sig, G, K) in layers.items():
        d0, df = sig / K, 2.0 * G / sig
        if d0 >= df:
            sys.exit('inadmissible %s law: d0=%.4g >= df=%.4g (needs sig < sqrt(2 G K))' % (k, d0, df))
        out[k] = dict(sig=sig, G=G, K=K, d0=d0, df=df)
    return out


def graded(length, n, h0):
    """n intervals over [0, length], geometric growth from first size h0."""
    if abs(h0 * n - length) < 1e-12:
        return [length * i / n for i in range(n + 1)]
    lo, hi = 1e-6, 10.0
    f = lambda r: h0 * (r ** n - 1.0) / (r - 1.0) - length if abs(r - 1) > 1e-12 else h0 * n - length
    if (h0 * n - length) < 0:
        lo = 1.0 + 1e-12
    else:
        hi = 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    pts, s, h = [0.0], 0.0, h0
    for _ in range(n):
        s += h
        pts.append(s)
        h *= r
    return [length * v / pts[-1] for v in pts]


def uniform(x0, x1, h):
    n = max(1, int(round((x1 - x0) / h)))
    return [x0 + (x1 - x0) * i / n for i in range(n + 1)]


def build(o):
    if not (o.H < o.x_notch - o.taper and o.x_notch <= o.a0 < o.L):
        sys.exit('need H < x_notch - taper and x_notch <= a0 < L')
    law = check_law(o)
    xg = o.x_notch - o.taper
    # x stations shared by both arms; a0 is always a node
    xs = uniform(0.0, o.H, o.h_far)
    for seg in (uniform(o.H, xg, o.h_far), uniform(xg, o.x_notch, o.h_coh * 2.5),
                uniform(o.x_notch, o.a0, o.h_coh), uniform(o.a0, o.L, o.h_coh)):
        if seg[-1] - seg[0] > 1e-9:          # a0 == x_notch leaves an empty precrack segment
            xs += seg[1:]
    ileg = [i for i, x in enumerate(xs) if x <= o.H + 1e-9]
    eta = [v / o.H for v in graded(o.H, o.ny, o.h_y0)]
    y_face, y_top = 0.5 * o.t, 0.5 * o.t + o.H

    def y_bot(x):
        if x <= xg:
            return y_face + o.groove
        if x >= o.x_notch:
            return y_face
        return y_face + o.groove * (o.x_notch - x) / o.taper

    nodes, els = {}, []          # els: (id, type, conn, elset)
    nx = len(xs)
    nleg = max(1, int(round(o.leg_h / o.h_leg)))
    sets = {}

    def add_arm(sign, base):
        """Upper arm sign=+1, lower sign=-1. Returns node-id grid g[i][j], j=0 at bond face."""
        g = [[0] * (o.ny + 1) for _ in range(nx)]
        for i, x in enumerate(xs):
            yb = y_bot(x)
            for j, e in enumerate(eta):
                nid = base + i * (o.ny + 1) + j + 1
                nodes[nid] = (x, sign * (yb + e * (y_top - yb)))
                g[i][j] = nid
        # leg above the arm over the leg columns
        lg = [[0] * (nleg + 1) for _ in ileg]
        lbase = base + nx * (o.ny + 1)
        for k, i in enumerate(ileg):
            lg[k][0] = g[i][o.ny]
            for m in range(1, nleg + 1):
                nid = lbase + k * nleg + m
                nodes[nid] = (xs[i], sign * (y_top + o.leg_h * m / nleg))
                lg[k][m] = nid
        return g, lg

    up, uleg = add_arm(+1, 0)
    lo, lleg = add_arm(-1, 1000000)

    eid = 0
    for sign, g, lg, tag in ((+1, up, uleg, 'U'), (-1, lo, lleg, 'L')):
        for i in range(nx - 1):
            for j in range(o.ny):
                q = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]]
                eid += 1
                els.append((eid, o.eltype, q if sign > 0 else q[::-1], 'ARM_' + tag))
        for k in range(len(ileg) - 1):
            for m in range(nleg):
                q = [lg[k][m], lg[k + 1][m], lg[k + 1][m + 1], lg[k][m + 1]]
                eid += 1
                els.append((eid, o.eltype, q if sign > 0 else q[::-1], 'LEG_' + tag))
    # cohesive row(s): face 1 on the lower arm, face 2 on the upper arm
    ibond = [i for i, x in enumerate(xs) if x >= o.a0 - 1e-9]
    layers = ['TIP'] if o.law == 'elastic' else list(law)
    for lay in layers:
        eid = {'TIP': 1, 'BR': 2, 'HUMP': 3}[lay] * 500000
        for i in ibond[:-1]:
            eid += 1
            els.append((eid, 'COH2D4', [lo[i][0], lo[i + 1][0], up[i + 1][0], up[i][0]], 'COH_' + lay))

    # bond and groove faces from the crack mouth (x = 0) to the end; mouth corner nodes
    sets['IFACE_U'] = [up[i][0] for i in range(nx)]
    sets['IFACE_L'] = [lo[i][0] for i in range(nx)]
    sets['MOUTH_U'] = [up[0][0]]
    sets['MOUTH_L'] = [lo[0][0]]
    sets['LEGTOP_U'] = [r[nleg] for r in uleg]
    sets['LEGTOP_L'] = [r[nleg] for r in lleg]
    rp1, rp2 = 2000001, 2000002
    nodes[rp1] = (0.5 * o.H, y_top + o.leg_h)
    nodes[rp2] = (0.5 * o.H, -(y_top + o.leg_h))
    geo = dict(x_groove_end=xg, n_x=nx, n_leg=nleg, n_bond=len(ibond) - 1,
               bond_ligament=o.L - o.a0, y_face=y_face, y_top=y_top,
               load_line_x=0.5 * o.H, n_nodes=len(nodes), n_elems=len(els))
    return nodes, els, sets, (rp1, rp2), law, geo


def fmt_ids(ids, per=16):
    return '\n'.join(', '.join(str(v) for v in ids[k:k + per]) for k in range(0, len(ids), per))


def git_rev():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.check_output(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        d = subprocess.check_output(['git', '-C', here, 'status', '--porcelain', '--', '.'], stderr=subprocess.DEVNULL)
        return r.decode().strip() + ('-dirty' if d.strip() else '')
    except Exception:
        return 'unknown'


def write_inp(o, nodes, els, sets, rps, law, geo):
    rp1, rp2 = rps
    T = o.delta / (o.rate / 60.0)                         # step time, s
    L = []
    w = L.append
    w('*Heading')
    w('T-peel %s | law=%s | H=%g t=%g a0=%g L=%g W=%g | bc %s/%s' %
      (o.job, o.law, o.H, o.t, o.a0, o.L, o.W, o.bc_bottom, o.bc_top))
    w('** generated by src/tslid/build_tpeel_inp.py @ %s' % git_rev())
    for k, v in sorted(vars(o).items()):
        w('** param %s = %s' % (k, v))
    w('*Preprint, echo=NO, model=NO, history=NO, contact=NO')
    w('*Node')
    for nid in sorted(nodes):
        w('%d, %.6f, %.6f' % ((nid,) + nodes[nid]))
    by = {}
    for e in els:
        by.setdefault((e[1], e[3]), []).append(e)
    for (et, es), group in sorted(by.items()):
        w('*Element, type=%s, elset=%s' % (et, es))
        for e in group:
            w('%d, %s' % (e[0], ', '.join(str(n) for n in e[2])))
    for k, ids in sets.items():
        w('*Nset, nset=%s' % k)
        w(fmt_ids(ids))
    w('*Nset, nset=RP1\n%d,' % rp1)
    w('*Nset, nset=RP2\n%d,' % rp2)
    w('*Elset, elset=ADHERENDS\nARM_U, ARM_L, LEG_U, LEG_L')
    coh = ['COH_TIP'] if o.law == 'elastic' else ['COH_' + k for k in law]
    w('*Elset, elset=COH_ALL\n' + ', '.join(coh))
    w('*Surface, type=NODE, name=S_LEGTOP_U\nLEGTOP_U, 1.')
    w('*Surface, type=NODE, name=S_LEGTOP_L\nLEGTOP_L, 1.')
    w('*Coupling, constraint name=C_TOP, ref node=RP1, surface=S_LEGTOP_U\n*Kinematic')
    w('*Coupling, constraint name=C_BOT, ref node=RP2, surface=S_LEGTOP_L\n*Kinematic')
    # sections
    w('*Solid Section, elset=ADHERENDS, material=AL6063T6\n%g,' % o.W)
    w('*Section Controls, name=EC_COH, viscosity=%g' % o.viscosity)
    for c in coh:
        w('*Cohesive Section, elset=%s, material=MAT_%s, response=TRACTION SEPARATION, '
          'thickness=SPECIFIED, controls=EC_COH' % (c, c[4:]))
        w('1.0, %g' % o.W)
    # materials
    w('*Material, name=AL6063T6\n*Elastic\n%g, %g' % (o.E_al, o.nu_al))
    if not o.no_plastic:
        w('*Plastic')
        for s, e in AL6063_PLASTIC:
            w('%.4f, %.5f' % (s, e))
    if o.law == 'elastic':
        w('*Material, name=MAT_TIP\n*Elastic, type=TRACTION\n%g, %g, %g' % (o.K0, o.K0, o.K0))
    for k, d in law.items():
        w('*Material, name=MAT_%s' % k)
        w('*Elastic, type=TRACTION\n%.6g, %.6g, %.6g' % (d['K'], d['K'], d['K']))
        if 'cards' in d:
            L.extend(d['cards'])
            continue
        w('*Damage Initiation, criterion=QUADS\n%.6g, %.6g, %.6g' % (d['sig'], d['sig'], d['sig']))
        w('*Damage Evolution, type=ENERGY, softening=LINEAR\n%.6g,' % d['G'])
    # boundary conditions: fixed part outside the step
    w('*Boundary')
    if o.bc_bottom == 'pin':
        w('RP2, 1, 2')
    else:
        w('RP2, ENCASTRE')
    if o.bc_top in ('pin', 'clamp'):
        w('RP1, 1, 1')
    if o.bc_top == 'clamp':
        w('RP1, 6, 6')
    # step
    dtmax = T / o.n_inc
    w('*Step, name=PEEL, nlgeom=%s, inc=1000000' % o.nlgeom)
    w('*Static')
    dt0 = T if o.law == 'elastic' else min(dtmax, T / 10.0) * 0.1
    w('%g, %g, %g, %g' % (dt0, T, T * 1e-9, dtmax))
    w('*Boundary\nRP1, 2, 2, %g' % o.delta)
    w('*Output, field, frequency=%d' % o.field_every)
    w('*Node Output\nU, RF')
    w('*Element Output, elset=COH_ALL\nS, E' + ('' if o.law == 'elastic' else ', SDEG, STATUS, QUADSCRT'))
    w('*Output, field, frequency=%d' % o.field_adh_every)
    w('*Element Output, elset=ADHERENDS\nS, PEEQ')
    w('*Output, history, frequency=1')
    w('*Node Output, nset=RP1\nU1, U2, UR3, RF1, RF2, RM3')
    w('*Node Output, nset=RP2\nU1, U2, UR3, RF1, RF2, RM3')
    w('*Energy Output\nALLSE, ALLPD, ALLDMD, ALLVD, ALLWK, ALLIE, ALLKE, ALLSD, ETOTAL')
    for c in coh:
        w('*Energy Output, elset=%s\nALLSE, ALLDMD, ALLVD' % c)
    w('*Node Print, nset=RP1, frequency=1\nU2, RF2')
    w('*Node Print, nset=MOUTH_U, frequency=1\nU2')
    w('*Node Print, nset=MOUTH_L, frequency=1\nU2')
    w('*End Step')
    os.makedirs(o.outdir, exist_ok=True)
    path = os.path.join(o.outdir, o.job + '.inp')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    law_meta = dict((k, dict((kk, vv) for kk, vv in d.items() if kk != 'cards')) for k, d in law.items())
    meta = dict(params=vars(o), law=law_meta, geometry=geo, step_time_s=T, git=git_rev())
    with open(os.path.join(o.outdir, o.job + '.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return path, meta


def main(argv=None):
    o = parse_args(argv)
    nodes, els, sets, rps, law, geo = build(o)
    path, meta = write_inp(o, nodes, els, sets, rps, law, geo)
    print('wrote %s  (%d nodes, %d elements, %d bond elements, ligament %.2f mm)' %
          (path, geo['n_nodes'], geo['n_elems'], geo['n_bond'], geo['bond_ligament']))
    for k, d in law.items():
        print('  %-3s sig=%-6g G=%-6g K=%-8.4g d0=%.4g df=%.4g' % (k, d['sig'], d['G'], d['K'], d['d0'], d['df']))


if __name__ == '__main__':
    main()
