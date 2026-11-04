"""Linear (eigenvalue) buckling: (K + λ K_G(σ)) φ = 0.

σ is the stress of a linear static solve under the project's loads; λ is the load factor
at which the structure buckles (critical load = λ × applied load).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kstruct.fem import tet10
from kstruct.fem.assembly import (
    CHUNK,
    SolverError,
    d_matrices,
    element_dofs,
    inverse_operator,
    part_property,
    rigid_body_modes,
    stiffness_matrix,
)
from kstruct.fem.bolts import bolt_stiffness_and_preload, build_bolts
from kstruct.fem.constraints import build_constraints
from kstruct.fem.mesh import FEMesh
from kstruct.fem.static import THERMAL_STRAIN, StaticResult, solve_static, temperature_change
from kstruct.model import Material, Project

Progress = Callable[[float, str], None]
NO_COMPRESSION = (
    "이 하중으로는 좌굴이 생기지 않습니다(누르는 힘이 없음). "
    "좌굴은 압축을 받는 가늘거나 얇은 부품에서 확인하세요."
)


@dataclass
class BucklingResult:
    mesh: FEMesh
    factors: np.ndarray  # (m,) positive load factors, ascending
    shapes: np.ndarray  # (m, N, 3) normalised mode shapes (max displacement 1)
    static: StaticResult  # the pre-buckling solution


def _geometric_stiffness(mesh: FEMesh, materials: dict[int, Material], u, dT) -> sp.csr_matrix:
    n = mesh.n_nodes * 3
    alpha = part_property(mesh, materials, "thermal_expansion")
    total = sp.csr_matrix((n, n))
    for start in range(0, len(mesh.tets), CHUNK):
        sl = slice(start, start + CHUNK)
        tets = mesh.tets[sl]
        X = mesh.nodes[tets]
        ue = u[element_dofs(tets)]
        D = d_matrices(mesh, materials, sl)
        G = np.zeros((len(tets), 10, 10))
        for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
            dNdx, det = tet10.gradients(X, p)
            eps = np.einsum("eij,ej->ei", tet10.strain_matrix(dNdx), ue)
            if dT is not None:
                dT_gp = dT[tets] @ tet10.shape_functions(p)
                eps = eps - np.outer(alpha[sl] * dT_gp, THERMAL_STRAIN)
            s = np.einsum("eij,ej->ei", D, eps)
            S = np.empty((len(tets), 3, 3))
            S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
            S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
            S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
            S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
            G += np.einsum("eai,eij,ebj,e->eab", dNdx, S, dNdx, det * w)
        Kg = np.zeros((len(tets), 30, 30))
        for k in range(3):
            Kg[:, k::3, k::3] = G
        dofs = element_dofs(tets)
        rows = np.repeat(dofs, 30, axis=1).ravel()
        cols = np.tile(dofs, (1, 30)).ravel()
        total = total + sp.csr_matrix((Kg.ravel(), (rows, cols)), shape=(n, n))
    return total


def solve_buckling(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    n_modes: int = 4,
    progress: Progress | None = None,
) -> BucklingResult:
    report = progress or (lambda frac, msg: None)
    static = solve_static(
        project, mesh, materials, lambda f, m: report(0.4 * f, f"하중 상태 계산: {m}")
    )
    # no compression anywhere -> nothing can buckle (and ARPACK would search in vain)
    if static.principal[:, 2].min() > -1e-6 * max(static.von_mises.max(), 1e-30):
        raise SolverError(NO_COMPRESSION)
    report(0.45, "기하 강성 행렬 조립 중")
    n_dof = mesh.n_nodes * 3
    u = static.displacement.ravel()
    thermal = static.thermal
    dT = temperature_change(project, mesh, thermal)
    Kg = _geometric_stiffness(mesh, materials, u, dT)
    c = build_constraints(project, mesh)
    K = stiffness_matrix(mesh, materials)
    K = K + c.penalty_matrix(n_dof, K.diagonal().max())
    K = K + bolt_stiffness_and_preload(build_bolts(project, mesh), n_dof)[0]
    free = np.flatnonzero(~c.fixed)
    Kf, Kgf = K[free][:, free].tocsc(), Kg[free][:, free].tocsc()

    report(0.6, "좌굴 하중 계산 중")
    # K φ = λ (-Kg) φ with K positive definite and Kg indefinite: ARPACK "buckling" mode.
    # Eigenvalues closest to the shift come first; negative ones (load reversal) are dropped,
    # so ask for extra.
    k = min(2 * n_modes + 4, len(free) - 2)
    sigma = 1e-3
    try:
        # buckling mode inverts K - σ(-Kg); with σ this small it stays positive definite
        OPinv = inverse_operator(Kf + sigma * Kgf, rigid_body_modes(mesh.nodes)[free])
        lam, phi = spla.eigsh(
            Kf, k=k, M=-Kgf, sigma=sigma, mode="buckling", which="LM", maxiter=300, OPinv=OPinv
        )
    except spla.ArpackNoConvergence as e:
        # no buckling mode anywhere near practical load levels (only tiny local compression)
        lam, phi = e.eigenvalues, e.eigenvectors
        if len(lam) == 0:
            raise SolverError(NO_COMPRESSION) from e
    except RuntimeError as e:
        raise SolverError("not converged") from e
    positive = lam > 1e-9
    if not positive.any():
        raise SolverError(NO_COMPRESSION)
    lam, phi = lam[positive], phi[:, positive]
    order = np.argsort(lam)[:n_modes]
    factors = lam[order]
    shapes = np.zeros((len(factors), n_dof))
    shapes[:, free] = phi[:, order].T
    shapes = shapes.reshape(len(factors), -1, 3)
    peak = np.linalg.norm(shapes, axis=2).max(axis=1)
    shapes /= np.where(peak > 0, peak, 1.0)[:, None, None]
    report(1.0, "완료")
    return BucklingResult(mesh, factors, shapes, static)
