#!/usr/bin/env python3
"""
single_element.py -- one COH2D4 element in pure mode-I opening, for verifying a law
representation against the analytic sigma(d) (CLAUDE.md: tests before sweeps).

Element: 1 mm long, out-of-plane thickness 1 mm, constitutive thickness 1, so nominal
strain E22 = separation (mm), S22 = traction (MPa), and energies are per mm^2:
the dissipated energy at complete failure must equal Gc (N/mm).

Loading (displacement control of the two top nodes):
    monotonic   0 -> 1.2 df
    cyclic      0 -> d_un -> 0 -> 1.2 df   (secant unloading check; d_un mid-softening)

Usage:
    python3 single_element.py --job se_exp_a3 --sigc 20 --Gc 0.4 --alpha 3 --softening exponential
Writes <job>.inp and <job>.json (law summary, analytic energy, representation energy).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tslid.laws import AlphaLaw  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a('--job', required=True)
    a('--outdir', default='.')
    a('--K', type=float, default=8500.0)
    a('--sigc', type=float, required=True)
    a('--Gc', type=float, required=True)
    a('--alpha', type=float, required=True)
    a('--softening', choices=['auto', 'linear', 'exponential', 'tabular'], default='auto')
    a('--n-tab', type=int, default=1000)
    a('--viscosity', type=float, default=0.0)
    a('--cyclic', type=float, default=None, help='unload at d = d0 + this fraction of (df - d0)')
    a('--n-inc', type=int, default=2000)
    o = p.parse_args(argv)

    law = AlphaLaw(o.K, o.sigc, o.Gc, o.alpha)
    dmg, rep = law.abaqus_cards(o.softening, o.n_tab)
    dmax = 1.2 * law.df
    L = []
    w = L.append
    w('*Heading\nsingle COH2D4 | %s' % json.dumps(law.summary()))
    w('*Node\n1, 0., 0.\n2, 1., 0.\n3, 1., 0.\n4, 0., 0.')
    w('*Element, type=COH2D4, elset=COH\n1, 1, 2, 3, 4')
    w('*Nset, nset=TOP\n3, 4')
    w('*Section Controls, name=EC, viscosity=%g, element deletion=NO' % o.viscosity)
    w('*Cohesive Section, elset=COH, material=MAT, response=TRACTION SEPARATION, '
      'thickness=SPECIFIED, controls=EC\n1.0, 1.0')
    w('*Material, name=MAT\n*Elastic, type=TRACTION\n%.8g, %.8g, %.8g' % (o.K, o.K, o.K))
    L.extend(dmg)
    w('*Boundary\n1, 1, 2\n2, 1, 2\n3, 1, 1\n4, 1, 1')
    if o.cyclic is not None:
        d_un = law.d0 + o.cyclic * (law.df - law.d0)
        t1, t2 = d_un / (2 * d_un + dmax), 2 * d_un / (2 * d_un + dmax)
        w('*Amplitude, name=CYC, definition=TABULAR\n0., 0., %.8g, %.8g, %.8g, 0., 1., 1.'
          % (t1, d_un / dmax, t2))
        amp = ', amplitude=CYC'
    else:
        amp = ''
    w('*Step, name=OPEN, nlgeom=NO, inc=1000000\n*Static, direct\n%g, 1.' % (1.0 / o.n_inc))
    w('*Boundary%s\nTOP, 2, 2, %.8g' % (amp, dmax))
    w('*Output, field, frequency=1\n*Element Output, elset=COH\nS, E, SDEG\n*Node Output\nU, RF')
    w('*Output, history, frequency=1\n*Energy Output\nALLSE, ALLDMD, ALLVD, ALLWK, ALLIE, ETOTAL')
    w('*End Step')
    os.makedirs(o.outdir, exist_ok=True)
    with open(os.path.join(o.outdir, o.job + '.inp'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    meta = dict(params=vars(o), law=law.summary(), representation=rep, dmax=dmax,
                d_unload=(law.d0 + o.cyclic * (law.df - law.d0)) if o.cyclic is not None else None)
    with open(os.path.join(o.outdir, o.job + '.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print('%s: d0=%.5g df=%.5g phi_e=%.3f rep=%s E1 err=%.1e'
          % (o.job, law.d0, law.df, law.phi_e, rep['softening'], rep['energy_rel_err']))


if __name__ == '__main__':
    main()
