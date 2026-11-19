"""p-multigrid preconditioned CG for the quadratic tetrahedron (TET10).

A TET10 mesh contains a linear (TET4) mesh on its corner nodes. The preconditioner
smooths on the full quadratic system (one Gauss-Seidel sweep before and after) and
corrects the smooth part of the error on that linear mesh, which has about an eighth of
the unknowns and is cheap to factorize exactly. The iteration count does not grow with
the mesh (measured 22-23 from 85k to 245k DOF), so the solve time grows linearly where
a sparse factorization grows faster than quadratically:

    DOF      direct (SuperLU)   AMG-CG   this
    85k      15 s               7-11 s   2.6 s
    245k     (not attempted)    48 s     8 s

It needs a matrix without penalty equations: a stiff tie between two parts cannot be
followed by the linear coarse functions and the iteration stalls. Constraints are
therefore eliminated first (Constraints.elimination) and this solver sees K̂ = Tᵀ K T.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kstruct.fem import tet10
from kstruct.fem.mesh import FEMesh

log = logging.getLogger(__name__)

TOLERANCE = 1e-10  # relative residual
MAX_ITERATIONS = 200  # a healthy system converges in 20-40; more means it is not suitable


class NotSuitable(Exception):
    """The iteration cannot solve this system (singular, or it does not converge); the
    caller falls back to the direct / AMG path, which also diagnoses an unheld part."""


def node_prolongation(mesh: FEMesh) -> sp.csr_matrix:
    """(N, Nc) linear-to-quadratic interpolation: a corner node copies its coarse value, a
    midside node takes the mean of the two ends of its edge. Nodes of no element get an
    empty row."""
    tets = mesh.tets
    corners = np.unique(tets[:, :4])
    coarse = np.full(mesh.n_nodes, -1)
    coarse[corners] = np.arange(len(corners))
    mid = tets[:, 4:].ravel()
    ends = tets[:, np.array(tet10.EDGES)].reshape(-1, 2)
    mid, first = np.unique(mid, return_index=True)
    ends = ends[first]
    rows = np.concatenate([corners, mid, mid])
    cols = np.concatenate([coarse[corners], coarse[ends[:, 0]], coarse[ends[:, 1]]])
    vals = np.concatenate([np.ones(len(corners)), np.full(2 * len(mid), 0.5)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(mesh.n_nodes, len(corners)))


def dof_prolongation(mesh: FEMesh, components: int = 3) -> sp.csr_matrix:
    """node_prolongation for `components` unknowns per node (3: displacements)."""
    P = node_prolongation(mesh)
    return sp.kron(P, sp.identity(components), format="csr") if components > 1 else P


class PMGSolver:
    """Solve A x = b for a symmetric positive definite A assembled on TET10 nodes.

    P: (n, nc) prolongation restricted to the rows of A (columns for fixed coarse
    unknowns removed by the caller; empty columns are dropped here).
    """

    def __init__(self, A: sp.spmatrix, P: sp.spmatrix) -> None:
        from pyamg.relaxation.relaxation import gauss_seidel

        self._gauss_seidel = gauss_seidel
        self.A = A.tocsr()
        self.A.sort_indices()
        P = P.tocsc()
        P = P[:, np.flatnonzero(np.diff(P.indptr) > 0)].tocsr()
        self.P, self.PT = P, P.T.tocsr()
        coarse = (self.PT @ self.A @ P).tocsc()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                self._coarse = spla.splu(
                    coarse,
                    permc_spec="MMD_AT_PLUS_A",
                    options={"SymmetricMode": True, "DiagPivotThresh": 0.0},
                )
            except (RuntimeError, spla.MatrixRankWarning) as e:
                raise NotSuitable("singular coarse matrix") from e
        self.iterations = 0  # of the last solve
        self._M = spla.LinearOperator(self.A.shape, matvec=self._precondition, dtype=float)

    def _precondition(self, r: np.ndarray) -> np.ndarray:
        r = np.ascontiguousarray(r, dtype=float)
        x = np.zeros_like(r)
        self._gauss_seidel(self.A, x, r, iterations=1, sweep="forward")
        x += self.P @ self._coarse.solve(self.PT @ (r - self.A @ x))
        self._gauss_seidel(self.A, x, r, iterations=1, sweep="backward")
        return x

    def solve(self, b: np.ndarray) -> np.ndarray:
        if not np.any(b):
            return np.zeros_like(b, dtype=float)
        count = [0]

        def step(_):
            count[0] += 1

        x, info = spla.cg(
            self.A, b, M=self._M, rtol=TOLERANCE, atol=0.0, maxiter=MAX_ITERATIONS, callback=step
        )
        self.iterations = count[0]
        if info != 0 or not np.all(np.isfinite(x)):
            raise NotSuitable(f"no convergence in {count[0]} iterations")
        log.debug("p-multigrid CG: %d iterations", count[0])
        return x
