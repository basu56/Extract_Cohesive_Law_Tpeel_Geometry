"""S1 test 1: patch test on a distorted 5-element patch, plus frame invariance of SVK.

Units: mm, N, MPa.  Threshold (S0_S2_INSTRUCTIONS.md): constant-strain energy exact to 1e-12 relative.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import tslid.fe  # noqa: F401
from tslid.fe.elements import BulkModel
from tslid.fe.kinematics import ENERGIES, lame
from tslid.fe.meshing import patch5
from tslid.fe.solve import SparseNewton

E, NU = 70000.0, 0.33
A = np.array([[1.0e-3, 2.0e-3], [-0.5e-3, 3.0e-3]])   # linear field u = A X + b
B = np.array([1.0e-3, -2.0e-3])


@pytest.mark.parametrize("kind", ["Q4", "Q4IM"])
@pytest.mark.parametrize("kin", ["linear", "svk"])
def test_patch(kind, kin):
    c, e, bnd = patch5()
    m = BulkModel(c, e, E, NU, 1.0, kind, kin)
    ue = c @ A.T + B
    fixed = np.r_[2 * bnd, 2 * bnd + 1]
    U0 = np.zeros(m.n_dof)
    U0[fixed] = np.r_[ue[bnd, 0], ue[bnd, 1]]
    r = SparseNewton(m.energy, m.n_dof, m.elem_dofs, fixed).solve(U0, rtol=1e-14, atol=1e-16)
    assert r.converged
    u, a = m.split(jnp.asarray(r.U))
    lam, mu = lame(E, NU)
    W_exact = float(ENERGIES[kin](jnp.asarray(A), lam, mu)) * 1.0      # unit-square area
    assert abs(float(m.energy(jnp.asarray(r.U))) - W_exact) <= 1e-12 * W_exact
    assert np.abs(np.asarray(u) - ue).max() <= 1e-12 * np.abs(ue).max()
    if a is not None:
        assert np.abs(np.asarray(a)).max() <= 1e-12 * np.abs(ue).max()


def test_svk_rigid_rotation_has_zero_energy():
    c, e, _ = patch5()
    m = BulkModel(c, e, E, NU, 1.0, "Q4IM", "svk")
    t = 0.3
    # rigid rotation by 0.3 rad: u = (R - I) X
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    U = np.r_[(c @ (R - np.eye(2)).T).ravel(), np.zeros(4 * len(e))]
    assert abs(float(m.energy(jnp.asarray(U)))) < 1e-20 * E


def test_svk_reduces_to_linear_for_small_strain():
    lam, mu = lame(E, NU)
    H = jnp.asarray(A) * 1e-4
    lin, svk = float(ENERGIES["linear"](H, lam, mu)), float(ENERGIES["svk"](H, lam, mu))
    assert abs(svk - lin) / lin < 1e-5
