"""Steady-state heat conduction on the TET10 mesh.

Internal units (mm, N, s): power in N·mm/s (1 W = 1000), conductivity k in W/(m·K) is
numerically N/(s·K), film coefficient h in W/(m²·K) is h·1e-3 N/(mm·s·K).

Radiation to the surroundings (q = εσ(T⁴ − T∞⁴)) and temperature-dependent conductivity
make the problem nonlinear; it is solved by fixed-point iteration: radiation is written as
an equivalent film coefficient h_r = εσ(T² + T∞²)(T + T∞) evaluated at the last
temperatures, conductivity is taken at the last element temperatures, and the linear
problem is solved again until the temperatures settle.
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
from kstruct.model import TEMPERATURE_REFERENCES, THERMAL_LOADS, LoadKind, Material, Project

Progress = Callable[[float, str], None]
WATT = 1000.0  # N·mm/s
H_SCALE = 1e-3  # W/(m²K) -> N/(mm·s·K)
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
KELVIN = 273.15
MAX_ITERATIONS = 60
TOLERANCE = 1e-4  # of the temperature range, between successive iterations

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
    heat_out: float  # W leaving through convection, radiation and fixed-temperature faces
    iterations: int = 1  # fixed-point iterations (1: linear problem)

    @property
    def heat_flux_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.heat_flux, axis=1)


def element_conductivity(
    mesh: FEMesh, materials: dict[int, Material], T: np.ndarray | None = None
) -> np.ndarray:
    """k per element (W/(m·K)); materials with a table are evaluated at the element's mean
    temperature when T is given, otherwise at their nominal value."""
    k = part_property(mesh, materials, "thermal_conductivity")
    if T is None:
        return k
    for pid, m in materials.items():
        if not m.conductivity_table:
            continue
        points = np.array(sorted(m.conductivity_table))
        mask = mesh.tet_part == pid
        mean = T[mesh.tets[mask]].mean(axis=1)
        k[mask] = np.interp(mean, points[:, 0], points[:, 1])
    return k


def radiation_coefficient(T: np.ndarray, ambient: float, emissivity: float) -> np.ndarray:
    """Equivalent film coefficient h_r (W/(m²·K)) with q = h_r (T − T∞) = εσ(T⁴ − T∞⁴)."""
    Ts, Ta = T + KELVIN, ambient + KELVIN
    return emissivity * STEFAN_BOLTZMANN * (Ts**2 + Ta**2) * (Ts + Ta)


def _conductivity_matrix(mesh: FEMesh, k: np.ndarray) -> sp.csr_matrix:
    X = mesh.nodes[mesh.tets]
    Ke = np.zeros((len(mesh.tets), 10, 10))
    for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
        dNdx, det = tet10.gradients(X, p)
        Ke += np.einsum("eni,emi,e->enm", dNdx, dNdx, k * det * w)
    rows = np.repeat(mesh.tets, 10, axis=1).ravel()
    cols = np.tile(mesh.tets, (1, 10)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))


def _surface_matrices(mesh: FEMesh, tris: np.ndarray, h):
    """Convection: H = ∫ h N Nᵀ dA (S,6,6) and ∫ N dA (S,6) on flat TRI6 faces; h may be
    a scalar or one value per triangle."""
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
    loads = [ld for ld in project.loads if ld.kind in THERMAL_LOADS]
    if not any(ld.kind in TEMPERATURE_REFERENCES for ld in loads):
        raise SolverError(
            "온도의 기준이 없습니다. '고정 온도', '대류 냉각', '복사 냉각' 중 하나 이상을 "
            "지정하세요."
        )
    faces = {}
    for i, ld in enumerate(loads):
        tris = mesh.tris[mesh.face_tris(ld.faces)]
        if len(tris) == 0:
            raise SolverError("열 조건을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
        faces[i] = tris
    nonlinear = any(ld.kind == LoadKind.RADIATION for ld in loads) or any(
        m.conductivity_table for m in materials.values()
    )

    # bonded parts conduct heat across the joint: reuse the tie equations (x component)
    report(0.1, "열전도 행렬 조립 중")
    ties = Constraints(fixed=np.zeros(3 * n, dtype=bool))
    for c in project.connections:
        add_bonded(ties, mesh, c)
    scalar = Constraints(fixed=np.zeros(n, dtype=bool))
    for dofs, coefs in zip(ties.penalty_rows, ties.penalty_coefs, strict=True):
        if np.all(dofs % 3 == 0):
            scalar.add_equation(dofs // 3, coefs)
    K0 = _conductivity_matrix(mesh, element_conductivity(mesh, materials))
    tie_matrix = scalar.penalty_matrix(n, K0.diagonal().max())

    def assemble(T_prev: np.ndarray | None):
        """Linear problem at the previous temperatures: (K, f, fixed mask, prescribed T,
        film terms (tris, h, ambient, ∫N dA) for the heat balance)."""
        k = element_conductivity(mesh, materials, T_prev)
        K = (K0 if T_prev is None else _conductivity_matrix(mesh, k)).tolil()
        f = np.zeros(n)
        fixed = np.zeros(n, dtype=bool)
        T = np.zeros(n)
        films: list[tuple[np.ndarray, np.ndarray | float, float, np.ndarray]] = []
        for i, ld in enumerate(loads):
            tris = faces[i]
            if ld.kind == LoadKind.FACE_TEMPERATURE:
                nodes = np.unique(tris)
                fixed[nodes] = True
                T[nodes] = ld.magnitude
            elif ld.kind == LoadKind.HEAT_FLOW:
                weights, _, area = tet10.tri6_load_weights(mesh.nodes[tris])
                q = ld.magnitude * WATT / area.sum()
                np.add.at(f, tris.ravel(), (weights * q).ravel())
            else:
                if ld.kind == LoadKind.CONVECTION:
                    h = ld.magnitude * H_SCALE
                else:  # radiation linearised at the face temperatures of the last pass
                    surface = (
                        np.full(len(tris), ld.ambient)
                        if T_prev is None
                        else T_prev[tris].mean(axis=1)
                    )
                    h = radiation_coefficient(surface, ld.ambient, ld.magnitude) * H_SCALE
                H, integral_n = _surface_matrices(mesh, tris, h)
                rows = np.repeat(tris, 6, axis=1).ravel()
                cols = np.tile(tris, (1, 6)).ravel()
                K = K + sp.csr_matrix((H.ravel(), (rows, cols)), shape=(n, n)).tolil()
                np.add.at(
                    f, tris.ravel(), (integral_n * (np.asarray(h)[..., None] * ld.ambient)).ravel()
                )
                films.append((tris, h, ld.ambient, integral_n))
        return K.tocsr() + tie_matrix, f, fixed, T, films

    def solve_once(T_prev):
        Kp, f, fixed, T, films = assemble(T_prev)
        free = np.flatnonzero(~fixed)
        rhs = f - Kp @ T
        T[free] = solve_spd(Kp[free][:, free].tocsr(), rhs[free], np.ones((len(free), 1)))
        return T, fixed, films

    report(0.5, "온도 계산 중")
    T, fixed, films = solve_once(None)
    iterations = 1
    if nonlinear:
        for iterations in range(2, MAX_ITERATIONS + 1):
            report(0.5 + 0.3 * iterations / MAX_ITERATIONS, f"온도 계산 중 (반복 {iterations})")
            T_new, fixed, films = solve_once(T)
            change = np.abs(T_new - T).max()
            T = T_new
            if change <= TOLERANCE * max(np.ptp(T), 1.0):
                break
        else:
            raise SolverError("not converged")

    report(0.85, "열 흐름 계산 중")
    k = element_conductivity(mesh, materials, T if nonlinear else None)
    flux = _nodal_flux(mesh, k, T)
    heat_in = sum(ld.magnitude for ld in loads if ld.kind == LoadKind.HEAT_FLOW)
    # heat leaving through fixed-temperature faces: minus the conduction "reaction" there
    heat_out = -float((_conductivity_matrix(mesh, k) @ T)[fixed].sum())
    for tris, h, ambient, integral_n in films:
        per_face = (integral_n * T[tris]).sum(axis=1) - ambient * integral_n.sum(axis=1)
        heat_out += float((h * per_face).sum())
    heat_out /= WATT
    report(1.0, "완료")
    return ThermalResult(mesh, T, flux, heat_in, heat_out, iterations)


def _nodal_flux(mesh: FEMesh, k: np.ndarray, T: np.ndarray) -> np.ndarray:
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
