"""build_coh_probe.py -- single elastic COH2D4 probes for the S3a interface-kinematics question.

Units: mm, MPa.  Light.  python scripts/fe/build_coh_probe.py <outdir>

One COH2D4 (length 1 mm, out-of-plane thickness 1, constitutive thickness 1, K = 8500 MPa/mm in
both directions, no damage), bottom face y = 0, top face y = g (geometric gap), every node's
displacement prescribed (so the element's reported S and E are a pure function of the motion):

  pg0_shear      g = 0,   NLGEOM NO,  top nodes u = (1e-3, 0)
  pg4_shear      g = 0.4, NLGEOM NO,  top nodes u = (1e-3, 0)
  pg4_shear_nl   g = 0.4, NLGEOM YES, top nodes u = (1e-3, 0)
  pg0_rot_nl     g = 0,   NLGEOM YES, all nodes rigid rotation 0.1 rad about (0.5, g/2)
  pg4_rot_nl     g = 0.4, NLGEOM YES, all nodes rigid rotation 0.1 rad about (0.5, g/2)
  pg4_rot        g = 0.4, NLGEOM NO,  same rigid rotation (linearised: only displacements given)
  pg4_open_rot_nl g = 0.4, NLGEOM YES, top face opened by 1e-3 along the rotated normal, then
                 everything rotated 0.1 rad (separation should be pure normal 1e-3 if the
                 element is objective)
Rotation applied linearly in 10 increments (displacement BCs interpolate linearly, so the
intermediate frames are not rigid; only the last frame is used).
"""
import math
import os
import sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
K = 8500.0


def deck(job, g, nlgeom, disp):
    X = [(0.0, 0.0), (1.0, 0.0), (1.0, g), (0.0, g)]
    L = ['*Heading\nCOH2D4 kinematics probe %s' % job, '*Node']
    L += ['%d, %.10g, %.10g' % (k + 1, x, y) for k, (x, y) in enumerate(X)]
    L += ['*Element, type=COH2D4, elset=COH\n1, 1, 2, 3, 4',
          '*Cohesive Section, elset=COH, material=MAT, response=TRACTION SEPARATION, thickness=SPECIFIED\n1.0, 1.0',
          '*Material, name=MAT\n*Elastic, type=TRACTION\n%g, %g, %g' % (K, K, K),
          '*Step, name=P, nlgeom=%s, inc=1000\n*Static, direct\n0.1, 1.0' % nlgeom, '*Boundary']
    for k, (x, y) in enumerate(X):
        u = disp(k, x, y)
        L += ['%d, 1, 1, %.12g' % (k + 1, u[0]), '%d, 2, 2, %.12g' % (k + 1, u[1])]
    L += ['*Output, field, frequency=1\n*Element Output, elset=COH\nS, E\n*Node Output\nU, RF',
          '*Output, history, frequency=1\n*Energy Output\nALLSE, ALLWK', '*End Step']
    with open(os.path.join(OUT, job + '.inp'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def rigid(th, g, extra=None):
    c, s = math.cos(th), math.sin(th)
    cx, cy = 0.5, g / 2

    def f(k, x, y):
        dx, dy = x - cx, y - cy
        if extra is not None and k in (2, 3):
            dy += extra                                  # open along the (unrotated) normal first
        return (cx + c * dx - s * dy - x, cy + s * dx + c * dy - y)
    return f


def shear(k, x, y):
    return (1e-3, 0.0) if k in (2, 3) else (0.0, 0.0)


deck('pg0_shear', 0.0, 'NO', shear)
deck('pg4_shear', 0.4, 'NO', shear)
deck('pg4_shear_nl', 0.4, 'YES', shear)
deck('pg0_rot_nl', 0.0, 'YES', rigid(0.1, 0.0))
deck('pg4_rot_nl', 0.4, 'YES', rigid(0.1, 0.4))
deck('pg4_rot', 0.4, 'NO', rigid(0.1, 0.4))
deck('pg4_open_rot_nl', 0.4, 'YES', rigid(0.1, 0.4, extra=1e-3))
print('wrote 7 probe decks to', OUT)
