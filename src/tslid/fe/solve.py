"""Newton solver with a sparse AD tangent.  Step S1 (static solves); load stepping grows in S3.

Units: mm, N, MPa, N/mm.

The problem is: minimise  Pi(U) = energy(U, *params) - f_ext . U  over the free dofs, with
prescribed values on the fixed dofs.  The residual is jax.grad(energy) - f_ext; the tangent is
assembled by ``tatva.sparse.jacfwd`` (forward-mode AD with distance-2 graph colouring of a
sparsity pattern built from the element dof map) and factorised with SciPy (SuperLU).

The Newton loop is a plain Python loop, deliberately not ``lax.while_loop``: gradients through
the solve will come from implicit differentiation (S5), not from unrolling the iterations.
"""
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from tatva.sparse import ColoredMatrix
from tatva.sparse import jacfwd as sparse_jacfwd


def pattern_from_elem_dofs(elem_dofs, n_dof):
    """CSR sparsity pattern (symmetric, sorted, with diagonal) from an (n_el, n_e) dof map."""
    d = np.asarray(elem_dofs, dtype=np.int64)
    n_e = d.shape[1]
    rows = np.repeat(d, n_e, axis=1).ravel()
    cols = np.tile(d, (1, n_e)).ravel()
    A = sps.coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    A.data[:] = 1
    return A


@dataclass
class NewtonResult:
    U: np.ndarray
    iterations: int
    residual_history: list
    converged: bool
    wall_time: float


class SparseNewton:
    """Static Newton solver for  grad energy(U, *params) = f_ext  on the free dofs.

    Args:
        energy: scalar function energy(U, *params) (N*mm).
        n_dof: length of U.
        elem_dofs: (n_el, n_e) global dof indices per element (sparsity pattern).
        fixed: indices of prescribed dofs.
    """

    def __init__(self, energy, n_dof, elem_dofs, fixed):
        self.n_dof = n_dof
        self.fixed = np.unique(np.asarray(fixed, dtype=np.int64))
        self.free = np.setdiff1d(np.arange(n_dof), self.fixed)
        t0 = time.perf_counter()
        pattern = pattern_from_elem_dofs(elem_dofs, n_dof)
        self.colored = ColoredMatrix.from_csr(pattern)
        self.n_colors = int(np.asarray(self.colored.colors).max()) + 1
        self.setup_time = time.perf_counter() - t0
        grad = jax.grad(energy)
        self._grad = jax.jit(grad)
        self._jac = jax.jit(sparse_jacfwd(grad, self.colored))

    def residual(self, U, f_ext, *params):
        return np.asarray(self._grad(jnp.asarray(U), *params)) - f_ext

    def tangent(self, U, *params):
        """Full sparse tangent (SciPy CSR) at U."""
        return self._jac(jnp.asarray(U), *params).to_csr()

    def solve(self, U0, f_ext=None, *params, rtol=1e-10, atol=1e-12, max_iter=25, verbose=False):
        """Newton iterations from U0 (fixed dofs of U0 hold the prescribed values).

        Converged when ||r_free||_inf <= max(atol, rtol * f_ref), with f_ref the largest of
        ||f_ext||_inf and the reaction ||r_fixed||_inf (so displacement control also works).
        """
        t0 = time.perf_counter()
        U = np.array(U0, dtype=float)
        f_ext = np.zeros(self.n_dof) if f_ext is None else np.asarray(f_ext, dtype=float)
        hist, ok = [], False
        for it in range(max_iter + 1):
            r = self.residual(U, f_ext, *params)
            rf = r[self.free]
            f_ref = max(np.abs(f_ext).max(), np.abs(r[self.fixed]).max() if self.fixed.size else 0.0)
            hist.append(float(np.abs(rf).max()))
            if verbose:
                print(f"  newton {it}: |r|_inf = {hist[-1]:.3e} (f_ref {f_ref:.3e})")
            if hist[-1] <= max(atol, rtol * f_ref):
                ok = True
                break
            if it == max_iter:
                break
            K = self.tangent(U, *params)
            Kff = K[self.free][:, self.free].tocsc()
            U[self.free] -= spla.spsolve(Kff, rf)
        return NewtonResult(U, it, hist, ok, time.perf_counter() - t0)

    def reactions(self, U, f_ext=None, *params):
        """Residual at the fixed dofs = reaction forces (N) that hold U."""
        f_ext = np.zeros(self.n_dof) if f_ext is None else f_ext
        return self.residual(U, f_ext, *params)[self.fixed], self.fixed
