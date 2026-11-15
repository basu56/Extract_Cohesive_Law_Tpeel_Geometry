"""S0 smoke tests for the tatva-based solver environment (docs/solver/S0_S2_INSTRUCTIONS.md).

Run:  uv run pytest tests/fe -m "not slow"
"""
import importlib.metadata
import subprocess
import sys

import pytest

import tslid.fe  # noqa: F401  (turns on x64)


def test_tatva_imports_and_version_is_pinned():
    import tatva  # noqa: F401
    assert importlib.metadata.version("tatva") == "0.11.6" == tslid.fe.TATVA_VERSION


def test_x64_is_active():
    import jax
    import jax.numpy as jnp
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64


@pytest.mark.parametrize("n", [2, 5])
def test_quad4_integrates_one_over_unit_square(n):
    from tatva import Mesh, Operator, element
    mesh = Mesh.unit_square(n, n, type="quad")
    op = Operator(mesh, element.Quad4())
    assert abs(float(op.integrate(1.0)) - 1.0) <= 1e-14


def test_tslid_laws_does_not_import_jax():
    """`import tslid.laws` must stay JAX-free for the ABAQUS-side interpreters."""
    code = "import sys, tslid, tslid.laws; assert 'jax' not in sys.modules, 'jax leaked'"
    subprocess.run([sys.executable, "-c", code], check=True)
