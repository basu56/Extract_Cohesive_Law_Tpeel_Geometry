"""
laws.py -- Phase 1 cohesive traction-separation law family (RESEARCH_PLAN.md section 4).

Pure standard library (runs under system python 3.6, abaqus python and the analysis env).

Units: mm, N, MPa.  K [MPa/mm], sigc [MPa], Gc [N/mm = kJ/m^2], separations [mm].

Law: linear loading to sigc at d0 = sigc/K, then
    sigma(d) = sigc * s(x; alpha),  x = (d - d0)/(df - d0) in [0, 1]
    s(x; alpha) = (exp(-alpha x) - exp(-alpha)) / (1 - exp(-alpha)),  s(x; 0) = 1 - x
    Gc = sigc d0 / 2 + sigc (df - d0) g(alpha),   g(alpha) = 1/alpha - 1/(exp(alpha) - 1)

ABAQUS representations (DECISIONS.md D8):
    'linear'       *Damage Evolution, type=DISPLACEMENT, softening=LINEAR       (alpha = 0 only)
    'exponential'  *Damage Evolution, type=DISPLACEMENT, softening=EXPONENTIAL  (df - d0, alpha)
    'tabular'      *Damage Evolution, type=DISPLACEMENT, softening=TABULAR      (D, d - d0) table
For a mode-I opening, ABAQUS's damage variable D gives sigma = (1 - D) K d, so
D(d) = 1 - sigma(d) / (K d).  The ABAQUS exponential law is
    D = 1 - (d0/d) {1 - [1 - exp(-alpha x)] / [1 - exp(-alpha)]},
which gives exactly sigc s(x; alpha) -- the hypothesis tested by the single-element check.

Energy constraints (RESEARCH_PLAN.md section 4b): E1 exact Gc, E2 elastic fraction
phi_e = sigc d0 / (2 Gc) <= PHI_E_MAX, E3 admissibility.
"""
import math

PHI_E_MAX = 0.5          # E2, agreed 2026-11-14
E1_RTOL = 1e-4           # E1 relative tolerance on the energy of the ABAQUS representation


def g_alpha(alpha):
    """Normalized softening energy: integral of s(x; alpha) over x in [0, 1]. g(0) = 1/2."""
    if abs(alpha) < 1e-6:
        return 0.5 - alpha / 12.0
    return 1.0 / alpha - 1.0 / math.expm1(alpha)


def s_shape(x, alpha):
    """Normalized softening shape, s(0) = 1, s(1) = 0."""
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    if abs(alpha) < 1e-9:
        return 1.0 - x
    # (e^{-a x} - e^{-a}) / (1 - e^{-a}) = expm1(a (1 - x)) / expm1(a); stable for either sign
    return math.expm1(alpha * (1.0 - x)) / math.expm1(alpha)


class AlphaLaw(object):
    """Cohesive law of the one-parameter shape family, defined by (K, sigc, Gc, alpha)."""

    def __init__(self, K, sigc, Gc, alpha):
        self.K, self.sigc, self.Gc, self.alpha = float(K), float(sigc), float(Gc), float(alpha)
        self.d0 = self.sigc / self.K
        self.g = g_alpha(self.alpha)
        soft = self.Gc - 0.5 * self.sigc * self.d0
        if soft <= 0.0:
            raise ValueError('inadmissible: Gc=%g <= sigc^2/(2K)=%g (needs sigc < sqrt(2 Gc K))'
                             % (self.Gc, 0.5 * self.sigc * self.d0))
        self.df = self.d0 + soft / (self.sigc * self.g)

    # --- descriptors -------------------------------------------------------------
    @property
    def phi_e(self):
        """E2: fraction of Gc stored elastically at damage onset, sigc^2 / (2 K Gc)."""
        return 0.5 * self.sigc * self.d0 / self.Gc

    def l_ch(self, Eprime):
        """Characteristic (Hillerborg/Irwin) length E' Gc / sigc^2 [mm]."""
        return Eprime * self.Gc / self.sigc ** 2

    def summary(self):
        return dict(K=self.K, sigc=self.sigc, Gc=self.Gc, alpha=self.alpha, d0=self.d0,
                    df=self.df, g=self.g, phi_e=self.phi_e)

    def check_energy_constraints(self):
        """E2 and E3; raises ValueError on violation. (E1 is checked on representations.)"""
        if self.phi_e > PHI_E_MAX + 1e-12:
            raise ValueError('E2 violated: phi_e=%.3f > %.2f' % (self.phi_e, PHI_E_MAX))
        if not (self.df > self.d0 > 0.0):
            raise ValueError('E3 violated: need df > d0 > 0')

    # --- the law -----------------------------------------------------------------
    def traction(self, d):
        """Monotonic-loading traction sigma(d) [MPa]."""
        if d <= 0.0:
            return 0.0
        if d <= self.d0:
            return self.K * d
        return self.sigc * s_shape((d - self.d0) / (self.df - self.d0), self.alpha)

    def damage(self, d):
        """ABAQUS damage variable D(d) = 1 - sigma / (K d) for d > d0."""
        if d <= self.d0:
            return 0.0
        if d >= self.df:
            return 1.0
        return 1.0 - self.traction(d) / (self.K * d)

    def energy_closed(self):
        return 0.5 * self.sigc * self.d0 + self.sigc * (self.df - self.d0) * self.g

    # --- ABAQUS representations -------------------------------------------------------
    def tabular(self, n=1000, p=2.0):
        """(D, u = d - d0) table, n+1 points clustered toward u = 0 with exponent p."""
        du = self.df - self.d0
        tab = []
        for i in range(n + 1):
            u = du * (float(i) / n) ** p
            tab.append((0.0 if i == 0 else (1.0 if i == n else self.damage(self.d0 + u)), u))
        return tab

    def energy_of_tabular(self, tab, m=50):
        """Work of separation of the law ABAQUS uses for a (D, u) table: D interpolated
        linearly in u, sigma = (1 - D) K d.  Composite Simpson on each segment."""
        W = 0.5 * self.sigc * self.d0
        for (D1, u1), (D2, u2) in zip(tab[:-1], tab[1:]):
            h = (u2 - u1) / (2 * m)
            acc = 0.0
            for k in range(2 * m + 1):
                u = u1 + k * h
                D = D1 + (D2 - D1) * (u - u1) / (u2 - u1)
                f = (1.0 - D) * self.K * (self.d0 + u)
                acc += f * (1 if k in (0, 2 * m) else (4 if k % 2 else 2))
            W += acc * h / 3.0
        return W

    def energy_quadrature(self, n=20000):
        """Composite Simpson of the analytic traction over [0, df] (independent of g)."""
        W = 0.5 * self.sigc * self.d0
        a, b = self.d0, self.df
        h = (b - a) / n
        acc = self.traction(a) + self.traction(b)
        for k in range(1, n):
            acc += self.traction(a + k * h) * (4 if k % 2 else 2)
        return W + acc * h / 3.0

    def abaqus_cards(self, softening='auto', n_tab=1000):
        """Material cards for the damage part of a cohesive material (list of lines).
        Returns (lines, rep) where rep describes the representation used and its energy."""
        if softening == 'auto':
            softening = 'linear' if abs(self.alpha) < 1e-12 else 'exponential'
        lines = ['*Damage Initiation, criterion=QUADS',
                 '%.8g, %.8g, %.8g' % (self.sigc, self.sigc, self.sigc)]
        du = self.df - self.d0
        if softening == 'linear':
            if abs(self.alpha) > 1e-12:
                raise ValueError('linear softening requires alpha = 0')
            lines += ['*Damage Evolution, type=DISPLACEMENT, softening=LINEAR', '%.8g,' % du]
            W = self.energy_closed()
        elif softening == 'exponential':
            lines += ['*Damage Evolution, type=DISPLACEMENT, softening=EXPONENTIAL',
                      '%.8g, %.8g' % (du, self.alpha)]
            W = self.energy_closed()
        elif softening == 'tabular':
            tab = self.tabular(n_tab)
            lines += ['*Damage Evolution, type=DISPLACEMENT, softening=TABULAR']
            lines += ['%.10g, %.10g' % (D, u) for D, u in tab]
            W = self.energy_of_tabular(tab)
        else:
            raise ValueError('unknown softening %r' % softening)
        rel = (W - self.Gc) / self.Gc
        if abs(rel) > E1_RTOL:
            raise ValueError('E1 violated: %s representation dissipates %.6g vs Gc %.6g (%.2e)'
                             % (softening, W, self.Gc, rel))
        return lines, dict(softening=softening, energy=W, energy_rel_err=rel)
