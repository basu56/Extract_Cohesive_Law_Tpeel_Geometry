"""Unit tests for src/tslid/laws.py.  Run:  python3 -m unittest discover -s tests -v"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tslid.laws import AlphaLaw, g_alpha, s_shape, PHI_E_MAX  # noqa: E402

ALPHAS = [-5.0, -2.0, -1e-8, 0.0, 1e-8, 1.0, 3.0, 7.0]
K = 8500.0


class TestShape(unittest.TestCase):
    def test_endpoints(self):
        for a in ALPHAS:
            self.assertAlmostEqual(s_shape(0.0, a), 1.0, places=12)
            self.assertAlmostEqual(s_shape(1.0, a), 0.0, places=12)
            self.assertAlmostEqual(s_shape(1.0 - 1e-12, a), 0.0, places=6)

    def test_monotone_decreasing(self):
        for a in ALPHAS:
            v = [s_shape(i / 200.0, a) for i in range(201)]
            self.assertTrue(all(x >= y for x, y in zip(v[:-1], v[1:])), a)

    def test_g_matches_quadrature(self):
        n = 20000
        for a in ALPHAS:
            h = 1.0 / n
            q = (s_shape(0, a) + s_shape(1, a) + sum(s_shape(k * h, a) * (4 if k % 2 else 2)
                                                     for k in range(1, n))) * h / 3
            self.assertAlmostEqual(g_alpha(a), q, places=9, msg=str(a))

    def test_g_limit_and_range(self):
        self.assertAlmostEqual(g_alpha(0.0), 0.5, places=12)
        for a in ALPHAS:
            self.assertTrue(0.0 < g_alpha(a) < 1.0)

    def test_abaqus_exponential_formula_equals_family(self):
        """ABAQUS exponential softening D, converted to traction, equals sigc*s(x;alpha)."""
        law = AlphaLaw(K, 30.0, 0.4, 3.0)
        for i in range(1, 50):
            x = i / 50.0
            d = law.d0 + x * (law.df - law.d0)
            a = law.alpha
            D = 1 - (law.d0 / d) * (1 - (1 - math.exp(-a * x)) / (1 - math.exp(-a)))
            self.assertAlmostEqual((1 - D) * K * d, law.traction(d), places=9)


class TestEnergy(unittest.TestCase):
    def cases(self):
        for Gc in (0.1, 0.25, 0.4):
            for sigc in (5.0, 15.0, 25.0):
                for a in (-2.0, 0.0, 1.0, 3.0, 7.0):
                    yield AlphaLaw(K, sigc, Gc, a)

    def test_E1_closed_form(self):
        for law in self.cases():
            self.assertAlmostEqual(law.energy_closed() / law.Gc, 1.0, places=12)

    def test_E1_quadrature(self):
        for law in self.cases():
            self.assertAlmostEqual(law.energy_quadrature() / law.Gc, 1.0, places=7)

    def test_E1_abaqus_representations(self):
        for law in self.cases():
            reps = ['tabular'] + (['linear'] if law.alpha == 0 else ['exponential'])
            for rep in reps:
                _, info = law.abaqus_cards(rep)
                self.assertLess(abs(info['energy_rel_err']), 1e-4, (rep, law.summary()))

    def test_tabular_is_valid_abaqus_table(self):
        tab = AlphaLaw(K, 20.0, 0.4, 7.0).tabular()
        self.assertEqual(tab[0], (0.0, 0.0))
        self.assertEqual(tab[-1][0], 1.0)
        self.assertTrue(all(u2 > u1 and D2 >= D1 for (D1, u1), (D2, u2) in zip(tab[:-1], tab[1:])))

    def test_E2_phi_e(self):
        law = AlphaLaw(K, 40.0, 0.1, 0.0)
        self.assertAlmostEqual(law.phi_e, 40.0 ** 2 / (2 * K * 0.1), places=12)
        with self.assertRaises(ValueError):
            law.check_energy_constraints()          # 0.94 > PHI_E_MAX
        AlphaLaw(K, math.sqrt(2 * PHI_E_MAX * K * 0.1) * 0.999, 0.1, 0.0).check_energy_constraints()

    def test_E3_inadmissible(self):
        with self.assertRaises(ValueError):
            AlphaLaw(K, math.sqrt(2 * 0.1 * K) * 1.001, 0.1, 0.0)


if __name__ == '__main__':
    unittest.main()
