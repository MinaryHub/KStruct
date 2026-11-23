"""Global matrix assembly, boundary conditions and linear solve."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kstruct.fem import pardiso, tet10
from kstruct.fem.mesh import FEMesh
from kstruct.model import Material

CHUNK = 4000  # elements per assembly batch (bounds peak memory)
# measured: at 90k DOF the direct solve (21 s) still beats AMG (25 s); contact and bolt
# preload loops factorize several times, so the direct path is kept up to this size
DIRECT_SOLVER_MAX_DOF = 120_000
# A refactorization costs about this many back-substitutions per nonzero-per-column of
# the factor (measured 0.17-0.30 from 17k to 85k DOF); the contact loop uses it to choose
# between a low-rank update and a fresh factorization without timing anything
REFACTOR_COLUMNS_PER_FILL = 0.25
AMG_BREAKEVEN = 2.0  # AMG setup is worth about this many preconditioned CG solves

# PARDISO (MKL, Windows/Linux x86-64): measured 18-55x faster factorization than SuperLU
# (85k-DOF bracket 16.2 s -> 0.9 s, 73k-DOF bolted joint 52 s -> 1.2 s, same answers),
# so with it the direct path is kept for much larger models (73k DOF peaked at 320 MB;
# fill grows about as n^(4/3), so 300k DOF stays near 2.3 GB)
PARDISO_MIN_DOF = 2_000  # below this SuperLU is as fast and starts no threads
PARDISO_MAX_DOF = 300_000
# a factor pivot below this fraction of its diagonal entry means K is singular (a part
# not held): an exact singularity came out at 1.7e-14, a 1e-9 stabilising spring at 5e-10
PARDISO_PIVOT_TOL = 1e-12
# refactorization vs low-rank update break-even for PARDISO: a factorization costs about
# this many (batched) back-substitutions per nonzero-per-column of the factor (73k DOF:
# 1.2 s factorization, 0.02 s per batched column, 460 nonzeros per column)
PARDISO_REFACTOR_COLUMNS_PER_FILL = 0.12

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


class SolverError(Exception):
    pass


def element_dofs(tets: np.ndarray) -> np.ndarray:
    """(E, 30) global DOF numbers in [u0x u0y u0z u1x ...] order."""
    return (tets[:, :, None] * 3 + np.arange(3)).reshape(len(tets), 30)


def element_materials(mesh: FEMesh, materials: dict[int, Material]) -> list[Material]:
    return [materials[int(pid)] for pid in mesh.tet_part]


def _assemble(mesh: FEMesh, element_matrices: Callable[[slice], np.ndarray]) -> sp.csr_matrix:
    n = mesh.n_nodes * 3
    total = sp.csr_matrix((n, n))
    for start in range(0, len(mesh.tets), CHUNK):
        sl = slice(start, start + CHUNK)
        Ke = element_matrices(sl)
        dofs = element_dofs(mesh.tets[sl])
        rows = np.repeat(dofs, 30, axis=1).ravel()
        cols = np.tile(dofs, (1, 30)).ravel()
        total = total + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    return total


def part_property(mesh: FEMesh, materials: dict[int, Material], attr: str) -> np.ndarray:
    lookup = {pid: getattr(m, attr) for pid, m in materials.items()}
    return np.array([lookup[int(p)] for p in mesh.tet_part], dtype=float)


def d_matrices(mesh: FEMesh, materials: dict[int, Material], sl: slice) -> np.ndarray:
    parts = mesh.tet_part[sl]
    D = np.empty((len(parts), 6, 6))
    for pid in np.unique(parts):
        m = materials[int(pid)]
        D[parts == pid] = tet10.elasticity_matrix(m.youngs_modulus, m.poisson_ratio)
    return D


def stiffness_matrix(mesh: FEMesh, materials: dict[int, Material]) -> sp.csr_matrix:
    return _assemble(
        mesh,
        lambda sl: tet10.stiffness(mesh.nodes[mesh.tets[sl]], d_matrices(mesh, materials, sl)),
    )


def mass_matrix(mesh: FEMesh, materials: dict[int, Material]) -> sp.csr_matrix:
    rho = part_property(mesh, materials, "density")
    return _assemble(mesh, lambda sl: tet10.consistent_mass(mesh.nodes[mesh.tets[sl]], rho[sl]))


def rigid_body_modes(nodes: np.ndarray) -> np.ndarray:
    """(3N, 6) translations + rotations, used as near-nullspace for AMG."""
    n = len(nodes)
    c = nodes - nodes.mean(axis=0)
    B = np.zeros((n, 3, 6))
    for i in range(3):
        B[:, i, i] = 1
    x, y, z = c.T
    B[:, 0, 3], B[:, 1, 3] = -y, x
    B[:, 1, 4], B[:, 2, 4] = -z, y
    B[:, 0, 5], B[:, 2, 5] = z, -x
    return B.reshape(3 * n, 6)


def make_solver(K: sp.csr_matrix, near_null: np.ndarray):
    """Factorize once, solve many right-hand sides: returns `solve(f) -> x`.

    near_null: rigid body modes restricted to the same DOFs (for AMG).
    A singular K (part not held in place) raises SolverError("unstable").
    """
    import warnings

    n = K.shape[0]
    if PARDISO_MIN_DOF <= n <= PARDISO_MAX_DOF and pardiso.available():
        solve = _pardiso_solver(K)
        if solve is not None:
            return solve
    if n <= DIRECT_SOLVER_MAX_DOF:
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                # K is symmetric positive definite: symmetric minimum-degree ordering with
                # plain diagonal pivots. Partial pivoting (SuperLU's default) wrecks the
                # ordering once stiff penalty ties are present (30x more fill, 30x slower).
                lu = spla.splu(
                    K.tocsc(),
                    permc_spec="MMD_AT_PLUS_A",
                    options={"SymmetricMode": True, "DiagPivotThresh": 0.0},
                )
            except (RuntimeError, spla.MatrixRankWarning) as e:
                raise SolverError("unstable") from e

        def solve(f: np.ndarray) -> np.ndarray:
            x = lu.solve(f)
            if not np.all(np.isfinite(x)):
                raise SolverError("unstable")
            return x

        solve.breakeven = REFACTOR_COLUMNS_PER_FILL * lu.nnz / max(n, 1)
        return solve

    import pyamg

    ml = pyamg.smoothed_aggregation_solver(
        K, B=near_null, symmetry="symmetric", strength="symmetric"
    )

    def solve(f: np.ndarray) -> np.ndarray:
        residuals: list[float] = []
        x = ml.solve(f, tol=1e-9, accel="cg", maxiter=800, residuals=residuals)
        if residuals and residuals[-1] > 1e-6 * residuals[0]:
            raise SolverError("not converged")
        if not np.all(np.isfinite(x)):
            raise SolverError("unstable")
        return x

    solve.breakeven = AMG_BREAKEVEN
    return solve


def _pardiso_solver(K: sp.csr_matrix):
    """PARDISO factorization of the SPD K, or None to fall back to SuperLU / AMG: a K
    that is not positive definite (singular, or an indefinite buckling shift) is left
    to them so that they diagnose it exactly as before."""
    try:
        factor = pardiso.factorize(K)
        df, da = factor.pivots()
    except pardiso.PardisoError as e:
        log.info("PARDISO declined (%s): using SuperLU", e)
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        smallest = float(np.min(df / np.abs(da))) if len(df) else 1.0
    if not smallest > PARDISO_PIVOT_TOL:
        factor.free()
        log.info("PARDISO: relative pivot %.1e, treating K as singular", smallest)
        return None

    def solve(f: np.ndarray) -> np.ndarray:
        x = factor.solve(f)
        if not np.all(np.isfinite(x)):
            raise SolverError("unstable")
        return x

    solve.breakeven = PARDISO_REFACTOR_COLUMNS_PER_FILL * factor.nnz_factor / K.shape[0]
    solve.factor = factor  # keeps the factorization alive as long as the solver
    return solve


def inverse_operator(A: sp.spmatrix, near_null: np.ndarray) -> spla.LinearOperator:
    """A⁻¹ as a LinearOperator for eigsh's OPinv (shift-invert).

    Without it eigsh factorizes with scipy's default splu (COLAMD + partial pivoting):
    measured 2x slower on a plain 94k-DOF bracket, far worse with penalty ties, and with
    no iterative fallback for large models. A must be the shifted matrix K - σM.
    """
    solve = make_solver(A.tocsr(), near_null)
    return spla.LinearOperator(A.shape, matvec=lambda x: solve(np.ravel(x)), dtype=float)


def solve_spd(K: sp.csr_matrix, f: np.ndarray, near_null: np.ndarray) -> np.ndarray:
    """Solve K x = f once for a symmetric positive definite K (free DOFs only)."""
    return make_solver(K, near_null)(f)
