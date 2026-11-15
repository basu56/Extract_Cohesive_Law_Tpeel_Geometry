"""Sparse linear solvers for the Newton step.  Step S3c (benchmark: docs/SOLVER_LOG.md, S3c).

Units: none (A in N/mm, b in N, x in mm).

Each entry of ``SOLVERS`` is a factory returning a callable ``solve(A, b) -> x`` (A: SciPy CSR or
CSC, square).  A factory gets a fresh object per run, so solvers that keep state (the symbolic
factorization of a fixed pattern) reuse it across Newton iterations and increments.

- superlu:      SciPy spsolve (SuperLU, COLAMD column ordering).  No reuse.  S1-S3b default.
- superlu_mmd:  SciPy splu with MMD on A^T + A (symmetric-pattern ordering).  No reuse.
- pardiso:      Intel MKL PARDISO via pypardiso (optional dependency), real nonsymmetric (mtype 11).
                Symbolic analysis (phase 11) once per sparsity pattern, then numerical factorization +
                solve (phase 23) per call.
- pardiso_sym:  same, real symmetric indefinite (mtype -2) on the upper triangle; valid because the
                Newton matrix is the Hessian of the incremental potential (symmetric to roundoff).
All accept b of shape (n,) or (n, p) (p right-hand sides on one factorization; S5a sensitivities).
Determinism (S5, D27): PARDISO with one thread and MKL_CBWR=COMPATIBLE, or the SuperLU solvers, give
bit-identical repeated runs (shown in the S5a log); multithreaded PARDISO need not.
UMFPACK and CHOLMOD need SuiteSparse, which Negishi does not provide (no module, no conda): not benchmarked.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


class SuperLU:
    name = "superlu"

    def __call__(self, A, b):
        return spla.spsolve(A.tocsc(), b)


class SuperLU_MMD:
    name = "superlu_mmd"

    def __call__(self, A, b):
        return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(b)


def _find_mkl_rt():
    """Point pypardiso at libmkl_rt from the mkl wheel when it is not on the loader path (uv overlays
    install it under <prefix>/lib of a layer other than sys.prefix)."""
    import glob
    import os
    import sys
    if os.environ.get("PYPARDISO_MKL_RT"):
        return
    for p in sys.path:
        if p.endswith("site-packages"):
            hits = sorted(glob.glob(os.path.join(p, "..", "..", "..", "lib", "libmkl_rt.so*")), key=len)
            if hits:
                os.environ["PYPARDISO_MKL_RT"] = os.path.realpath(hits[0])
                return


class Pardiso:
    """MKL PARDISO with reuse of the symbolic factorization for a fixed sparsity pattern."""
    name = "pardiso"
    mtype = 11

    def __init__(self):
        _find_mkl_rt()
        import pypardiso                                    # fe extra (x86_64)
        self.ps = pypardiso.PyPardisoSolver(mtype=self.mtype)
        # iparm(1) = 0: MKL defaults for this mtype (mtype 11: METIS ordering, scaling + weighted
        # matching, pivot perturbation 1e-13; mtype -2: Bunch-Kaufman, perturbation 1e-8)
        self.key = None
        self.n_analysis = 0
        self.n_perturbed = 0

    def _prep(self, A):
        A = A.tocsr()
        if not A.has_sorted_indices:
            A.sort_indices()
        return A

    def __call__(self, A, b):
        A = self._prep(A)
        key = (A.shape[0], A.nnz, hash(A.indptr.tobytes()), hash(A.indices.tobytes()))
        if key != self.key:
            if self.key is not None:
                self.ps.free_memory(everything=True)
            self.ps.set_phase(11)
            self.ps._check_A(A)
            self.ps._call_pardiso(A, np.zeros((A.shape[0], 1)))
            self.key = key
            self.n_analysis += 1
        self.ps.set_phase(23)
        self.ps._check_A(A)
        b = np.asarray(b, dtype=float)
        x = self.ps._call_pardiso(A, np.asfortranarray(b))
        self.n_perturbed += int(self.ps.get_iparm(14) > 0)      # perturbed pivots in this factorization
        return np.asarray(x).ravel() if b.ndim == 1 else np.asarray(x).reshape(b.shape)


class PardisoSym(Pardiso):
    """Symmetric indefinite PARDISO (Bunch-Kaufman pivoting) on the upper triangle."""
    name = "pardiso_sym"
    mtype = -2

    def __init__(self):
        super().__init__()
        self._pat = None

    def _prep(self, A):
        A = A.tocsr()
        if not A.has_sorted_indices:
            A.sort_indices()
        pat = (A.shape[0], A.nnz, hash(A.indptr.tobytes()), hash(A.indices.tobytes()))
        if pat != self._pat:                                    # upper-triangle extraction map, once
            r = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
            self._keep = np.where(A.indices >= r)[0]
            U = sps.csr_matrix((np.ones(len(self._keep)), A.indices[self._keep],
                                np.r_[0, np.cumsum(np.bincount(r[self._keep], minlength=A.shape[0]))]),
                               shape=A.shape)
            self._ind, self._ptr = U.indices, U.indptr
            self._pat = pat
        return sps.csr_matrix((A.data[self._keep], self._ind, self._ptr), shape=A.shape)


SOLVERS = {"superlu": SuperLU, "superlu_mmd": SuperLU_MMD, "pardiso": Pardiso, "pardiso_sym": PardisoSym}

DEFAULT = "pardiso_sym"          # D24
FALLBACK = "superlu_mmd"         # fastest SuperLU variant in the S3c benchmark


def make_solver(name=DEFAULT, fallback=FALLBACK):
    """Solver object for ``name``.  If it cannot be created because pypardiso / MKL is unavailable
    (no x86_64 wheel, e.g. Apple Silicon, or no libmkl_rt), warn and return ``fallback`` instead
    (D24).  The returned object has ``.name`` (the solver actually used) and ``.requested``."""
    import warnings
    try:
        s = SOLVERS[name]()
    except (ImportError, OSError) as e:
        if fallback is None:
            raise
        warnings.warn("linear solver %r unavailable (%s); falling back to %r" % (name, e, fallback),
                      RuntimeWarning, stacklevel=2)
        s = SOLVERS[fallback]()
    s.requested = name
    return s
