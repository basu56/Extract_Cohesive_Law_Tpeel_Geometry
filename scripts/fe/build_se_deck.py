#!/usr/bin/env python3
"""build_se_deck.py -- single COH2D4 decks for S2 tests 3-4 and the quadrature probe.

Units: mm, N, MPa, s.  Python 3.6+, stdlib only.  New deck generator (the Phase 0 script
src/tslid/single_element.py is not modified); same element, section and law cards.

One COH2D4, 1 mm long, out-of-plane thickness 1, constitutive thickness 1 (separation = nominal
strain).  Bottom nodes 1, 2 fixed; top nodes 3, 4 prescribed:
    --mode n   : U2 = 1.2 df            (mode I)
    --mode s   : U1 = 1.2 df, U2 = 0    (pure shear)
    --mode ns  : U1 = U2 = 1.2 df / sqrt(2)  (proportional, delta_s = delta_n; delta_m = 1.2 df)
    --probe    : elastic probe of the integration rule: node 3 U2 = d_probe, node 4 U2 = 0
Step time --T (s), or --rate (mm/min of the largest prescribed displacement) as in the T-peel
builder (step time = displacement / (rate / 60)).  --viscosity sets *Section Controls viscosity.

    python3 scripts/fe/build_se_deck.py --job se_ns_exp_a3 --sigc 20 --Gc 0.4 --alpha 3 --mode ns
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from tslid.laws import AlphaLaw  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a('--job', required=True)
    a('--outdir', default='.')
    a('--K', type=float, default=8500.0)
    a('--sigc', type=float, default=20.0)
    a('--Gc', type=float, default=0.4)
    a('--alpha', type=float, default=3.0)
    a('--softening', choices=['auto', 'linear', 'exponential', 'tabular'], default='auto')
    a('--n-tab', type=int, default=1000)
    a('--mode', choices=['n', 's', 'ns'], default='n')
    a('--probe', type=float, default=None, help='elastic quadrature probe: node-3 opening, mm')
    a('--viscosity', type=float, default=0.0)
    a('--T', type=float, default=1.0, help='step time, s')
    a('--rate', type=float, default=None, help='mm/min; overrides --T')
    a('--n-inc', type=int, default=2000)
    o = p.parse_args(argv)

    law = AlphaLaw(o.K, o.sigc, o.Gc, o.alpha)
    dmg, rep = law.abaqus_cards(o.softening, o.n_tab)
    dm = 1.2 * law.df
    if o.probe is not None:
        U = {3: (0.0, o.probe), 4: (0.0, 0.0)}
    else:
        un, us = {'n': (dm, 0.0), 's': (0.0, dm), 'ns': (dm / math.sqrt(2.0), dm / math.sqrt(2.0))}[o.mode]
        U = {3: (us, un), 4: (us, un)}
    umax = max(max(abs(v) for v in uv) for uv in U.values())
    T = umax / (o.rate / 60.0) if o.rate else o.T
    n_inc = 1 if o.probe is not None else o.n_inc
    L = []
    w = L.append
    w('*Heading\nS2 single COH2D4 %s | %s' % (o.job, json.dumps(law.summary())))
    w('*Node\n1, 0., 0.\n2, 1., 0.\n3, 1., 0.\n4, 0., 0.')
    w('*Element, type=COH2D4, elset=COH\n1, 1, 2, 3, 4')
    w('*Section Controls, name=EC, viscosity=%.10g, element deletion=NO' % o.viscosity)
    w('*Cohesive Section, elset=COH, material=MAT, response=TRACTION SEPARATION, '
      'thickness=SPECIFIED, controls=EC\n1.0, 1.0')
    w('*Material, name=MAT\n*Elastic, type=TRACTION\n%.8g, %.8g, %.8g' % (o.K, o.K, o.K))
    if o.probe is None:
        L.extend(dmg)
    w('*Boundary\n1, 1, 2\n2, 1, 2')
    w('*Step, name=OPEN, nlgeom=NO, inc=1000000\n*Static, direct\n%.10g, %.10g' % (T / n_inc, T))
    w('*Boundary')
    for n in (3, 4):
        w('%d, 1, 1, %.10g' % (n, U[n][0]))
        w('%d, 2, 2, %.10g' % (n, U[n][1]))
    w('*Output, field, frequency=1\n*Element Output, elset=COH\nS, E, SDEG\n*Node Output\nU, RF')
    w('*Output, history, frequency=1\n*Energy Output\nALLSE, ALLDMD, ALLVD, ALLWK, ALLIE, ETOTAL')
    w('*End Step')
    os.makedirs(o.outdir, exist_ok=True)
    with open(os.path.join(o.outdir, o.job + '.inp'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    meta = dict(params=vars(o), law=law.summary(), representation=rep, dm_max=dm, U_top=U,
                step_time=T, n_inc=n_inc)
    with open(os.path.join(o.outdir, o.job + '.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print('%s: mode=%s T=%.6g s visc=%g rep=%s' % (o.job, o.mode, T, o.viscosity, rep['softening']))


if __name__ == '__main__':
    main()
