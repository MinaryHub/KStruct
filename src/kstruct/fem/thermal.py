"""Steady-state heat conduction on the TET10 mesh.

Internal units (mm, N, s): power in N·mm/s (1 W = 1000), conductivity k in W/(m·K) is
numerically N/(s·K), film coefficient h in W/(m²·K) is h·1e-3 N/(mm·s·K).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from kstruct.fem import tet10
from kstruct.fem.assembly import SolverError, part_property, solve_spd
from kstruct.fem.constraints import Constraints, _tri6_shape, add_bonded
from kstruct.fem.mesh import FEMesh
from kstruct.model import THERMAL_LOADS, LoadKind, Material, Project

Progress = Callable[[float, str], None]
WATT = 1000.0  # N·mm/s
H_SCALE = 1e-3  # W/(m²K) -> N/(mm·s·K)

# 6-point degree-4 rule on the unit triangle (barycentric), weights sum to 1
_A, _WA = 0.445948490915965, 0.223381589678011
_B, _WB = 0.091576213509771, 0.109951743655322
TRI_POINTS = np.array(
    [
        [_A, _A, 1 - 2 * _A],
        [_A, 1 - 2 * _A, _A],
        [1 - 2 * _A, _A, _A],
        [_B, _B, 1 - 2 * _B],
        [_B, 1 - 2 * _B, _B],
        [1 - 2 * _B, _B, _B],
    ]
)
TRI_WEIGHTS = np.array([_WA] * 3 + [_WB] * 3)


def has_thermal_loads(project: Project) -> bool:
    return any(ld.kind in THERMAL_LOADS for ld in project.loads)


@dataclass
class ThermalResult:
    mesh: FEMesh
    temperature: np.ndarray  # (N,) °C
    heat_flux: np.ndarray  # (N, 3) W/m² (nodal average)
    heat_in: float  # W through heat-flow faces
    heat_out: float  # W leaving through convection and fixed-temperature faces

    @property
    def heat_flux_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.heat_flux, axis=1)


def _conductivity_matrix(mesh: FEMesh, materials: dict[int, Material]) -> sp.csr_matrix:
    k = part_property(mesh, materials, "thermal_conductivity")
    X = mesh.nodes[mesh.tets]
    Ke = np.zeros((len(mesh.tets), 10, 10))
    for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
        dNdx, det = tet10.gradients(X, p)
        Ke += np.einsum("eni,emi,e->enm", dNdx, dNdx, k * det * w)
    rows = np.repeat(mesh.tets, 10, axis=1).ravel()
    cols = np.tile(mesh.tets, (1, 10)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))


def _surface_matrices(mesh: FEMesh, tris: np.ndarray, h: float):
    """Convection: H = ∫ h N Nᵀ dA (S,6,6) and ∫ N dA (S,6) on flat TRI6 faces."""
    _, _, area = tet10.tri6_load_weights(mesh.nodes[tris])
    H = np.zeros((len(tris), 6, 6))
    integral_n = np.zeros((len(tris), 6))
    for (l0, l1, l2), w in zip(TRI_POINTS, TRI_WEIGHTS, strict=True):
        N = _tri6_shape(np.float64(l0), np.float64(l1), np.float64(l2))
        H += np.einsum("s,ij->sij", h * area * w, np.outer(N, N))
        integral_n += np.outer(area * w, N)
    return H, integral_n


def solve_thermal(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    progress: Progress | None = None,
) -> ThermalResult:
    report = progress or (lambda frac, msg: None)
    n = mesh.n_nodes
    report(0.1, "열전도 행렬 조립 중")
    K = _conductivity_matrix(mesh, materials).tolil()
    f = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    T = np.zeros(n)
    convection: list[tuple[np.ndarray, float, float, np.ndarray]] = []

    for ld in project.loads:
        if ld.kind not in THERMAL_LOADS:
            continue
        tris = mesh.tris[mesh.face_tris(ld.faces)]
        if len(tris) == 0:
            raise SolverError("열 조건을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
        if ld.kind == LoadKind.FACE_TEMPERATURE:
            nodes = np.unique(tris)
            fixed[nodes] = True
            T[nodes] = ld.magnitude
        elif ld.kind == LoadKind.HEAT_FLOW:
            weights, _, area = tet10.tri6_load_weights(mesh.nodes[tris])
            q = ld.magnitude * WATT / area.sum()
            np.add.at(f, tris.ravel(), (weights * q).ravel())
        else:
            h = ld.magnitude * H_SCALE
            H, integral_n = _surface_matrices(mesh, tris, h)
            rows = np.repeat(tris, 6, axis=1).ravel()
            cols = np.tile(tris, (1, 6)).ravel()
            K = K + sp.csr_matrix((H.ravel(), (rows, cols)), shape=(n, n)).tolil()
            np.add.at(f, tris.ravel(), (integral_n * h * ld.ambient).ravel())
            convection.append((tris, h, ld.ambient, integral_n))

    if not fixed.any() and not any(ld.kind == LoadKind.CONVECTION for ld in project.loads):
        raise SolverError(
            "온도의 기준이 없습니다. '고정 온도'나 '대류 냉각'을 하나 이상 지정하세요."
        )

    K = K.tocsr()
    # bonded parts conduct heat across the joint: reuse the tie equations (x component)
    ties = Constraints(fixed=np.zeros(3 * n, dtype=bool))
    for c in project.connections:
        add_bonded(ties, mesh, c)
    scalar = Constraints(fixed=np.zeros(n, dtype=bool))
    for dofs, coefs in zip(ties.penalty_rows, ties.penalty_coefs, strict=True):
        if np.all(dofs % 3 == 0):
            scalar.add_equation(dofs // 3, coefs)
    Kp = K + scalar.penalty_matrix(n, K.diagonal().max())

    report(0.5, "온도 계산 중")
    free = np.flatnonzero(~fixed)
    rhs = f - Kp @ T
    T[free] = solve_spd(Kp[free][:, free].tocsr(), rhs[free], np.ones((len(free), 1)))

    report(0.85, "열 흐름 계산 중")
    flux = _nodal_flux(mesh, materials, T)
    heat_in = sum(ld.magnitude for ld in project.loads if ld.kind == LoadKind.HEAT_FLOW)
    # heat leaving through fixed-temperature faces: minus the conduction "reaction" there
    heat_out = -float((_conductivity_matrix(mesh, materials) @ T)[fixed].sum())
    for tris, h, ambient, integral_n in convection:
        heat_out += float(h * ((integral_n * T[tris]).sum() - ambient * integral_n.sum()))
    heat_out /= WATT
    report(1.0, "완료")
    return ThermalResult(mesh, T, flux, heat_in, heat_out)


def _nodal_flux(mesh: FEMesh, materials: dict[int, Material], T: np.ndarray) -> np.ndarray:
    k = part_property(mesh, materials, "thermal_conductivity")
    X = mesh.nodes[mesh.tets]
    Te = T[mesh.tets]
    acc = np.zeros((mesh.n_nodes, 3))
    count = np.zeros(mesh.n_nodes)
    for j, p in enumerate(tet10.NODE_POINTS):
        dNdx, _ = tet10.gradients(X, p)
        grad = np.einsum("eni,en->ei", dNdx, Te)  # K/mm
        np.add.at(acc, mesh.tets[:, j], -k[:, None] * grad * 1e3)  # W/m²
        np.add.at(count, mesh.tets[:, j], 1)
    return acc / np.maximum(count, 1)[:, None]
