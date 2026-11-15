"""Differentiable T-peel forward solver on tatva (JAX).  Decision D19.

Units throughout ``tslid.fe``: mm, N, MPa, N/mm (same as the ABAQUS decks).

Importing this package turns on float64 in JAX; all verification is done in float64.
It does not import tatva or the submodules, so ``import tslid.fe`` stays light.
``tslid/__init__.py`` must never import this package: ``import tslid.laws`` has to
work in the ABAQUS-side interpreters (Python 3.6 / ``abaqus python``) without JAX.

Plan and verification ladder: ``docs/solver/S0_S2_INSTRUCTIONS.md``;
results per step: ``docs/SOLVER_LOG.md``.
"""
import jax

jax.config.update("jax_enable_x64", True)

#: tatva version this layer is written and verified against (pinned in pyproject.toml).
TATVA_VERSION = "0.11.6"

__all__ = ["TATVA_VERSION"]
