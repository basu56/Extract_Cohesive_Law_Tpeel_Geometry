"""Sparse linear solvers for the Newton step.  Step S3c (benchmark: docs/SOLVER_LOG.md, S3c).

Units: none (A in N/mm, b in N, x in mm).

Each entry of ``SOLVERS`` is a factory returning a callable ``solve(A, b) -> x`` (A: SciPy CSR or
CSC, square).  A factory gets a fresh object per run, so solvers that keep state (the symbolic
factorization of a fixed pattern) reuse it across Newton iterations and increments.
"""
import numpy as np
import scipy.sparse.linalg as spla


class SuperLU:
    """SciPy SuperLU (spsolve, COLAMD ordering): no reuse; the S1-S3b default."""
    name = "superlu"

    def __call__(self, A, b):
        return spla.spsolve(A.tocsc(), b)


SOLVERS = {"superlu": SuperLU}
