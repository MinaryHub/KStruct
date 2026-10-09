"""Linear static analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kstruct.fem import tet10
from kstruct.fem.assembly import (
    SolverError,
    d_matrices,
    element_dofs,
    part_property,
    rigid_body_modes,
    solve_spd,
    stiffness_matrix,
)
from kstruct.fem.constraints import Constraints, build_constraints
from kstruct.fem.mesh import FEMesh
from kstruct.fem.thermal import ThermalResult, has_thermal_loads, solve_thermal
from kstruct.model import LoadKind, Material, Project

Progress = Callable[[float, str], None]

THERMAL_STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
FOS_CAP = 100.0  # safety factors above this are shown as "100+"


@dataclass
class StaticResult:
    mesh: FEMesh
    materials: dict[int, Material]
    displacement: np.ndarray  # (N, 3) mm
    stress: np.ndarray  # (N, 6) MPa, nodal average within each part
    reactions: list[np.ndarray]  # per fixture, (3,) N
    applied_load: np.ndarray  # (3,) N, sum of external forces
    thermal: ThermalResult | None = None  # temperature field used for thermal stress

    @property
    def von_mises(self) -> np.ndarray:
        s = self.stress
        return np.sqrt(
            0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
            + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
        )

    @property
    def principal(self) -> np.ndarray:
        """(N, 3) principal stresses, descending."""
        s = self.stress
        T = np.empty((len(s), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
        T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
        T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
        return np.linalg.eigvalsh(T)[:, ::-1]

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.displacement, axis=1)

    def node_material(self, attr: str) -> np.ndarray:
        """Per-node material property (nodes belong to exactly one part)."""
        out = np.zeros(self.mesh.n_nodes)
        for pid, m in self.materials.items():
            out[self.mesh.part_nodes(pid)] = getattr(m, attr)
        return out

    @property
    def safety_factor(self) -> np.ndarray:
        """Per node. Ductile: yield / von Mises. Brittle: tensile strength / max principal."""
        fos = np.full(self.mesh.n_nodes, FOS_CAP)
        for pid, m in self.materials.items():
            nodes = self.mesh.part_nodes(pid)
            if m.ductile:
                demand, capacity = self.von_mises[nodes], m.yield_strength
            else:
                demand, capacity = np.maximum(self.principal[nodes, 0], 0), m.tensile_strength
            with np.errstate(divide="ignore"):
                fos[nodes] = np.minimum(capacity / demand, FOS_CAP)
        return fos


def _load_vector(project: Project, mesh: FEMesh, materials: dict[int, Material]) -> np.ndarray:
    f = np.zeros((mesh.n_nodes, 3))
    for load in project.loads:
        if load.kind in (LoadKind.FORCE, LoadKind.PRESSURE):
            tris = mesh.tris[mesh.face_tris(load.faces)]
            if len(tris) == 0:
                raise SolverError("하중을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
            weights, normal, area = tet10.tri6_load_weights(mesh.nodes[tris])
            if load.kind == LoadKind.FORCE:
                traction = np.broadcast_to(np.asarray(load.vector) / area.sum(), normal.shape)
            else:
                traction = -load.magnitude * normal
            np.add.at(f, tris.ravel(), (weights[:, :, None] * traction[:, None, :]).reshape(-1, 3))
        elif load.kind == LoadKind.GRAVITY:
            X = mesh.nodes[mesh.tets]
            rho = part_property(mesh, materials, "density")
            w = tet10.mass_lumped_volume(X) * rho[:, None]
            np.add.at(f, mesh.tets.ravel(), np.outer(w.ravel(), load.vector))
    return f.ravel()


def temperature_change(
    project: Project, mesh: FEMesh, thermal: ThermalResult | None
) -> np.ndarray | None:
    """Nodal temperature change from the stress-free state, or None without thermal loads."""
    uniform = sum(ld.magnitude for ld in project.loads if ld.kind == LoadKind.TEMPERATURE)
    if thermal is None and uniform == 0:
        return None
    dT = np.full(mesh.n_nodes, float(uniform))
    if thermal is not None:
        dT += thermal.temperature - project.reference_temperature
    return dT


def _thermal_load(mesh: FEMesh, materials: dict[int, Material], dT: np.ndarray) -> np.ndarray:
    """Equivalent nodal forces of free thermal expansion alpha * dT."""
    alpha = part_property(mesh, materials, "thermal_expansion")
    X = mesh.nodes[mesh.tets]
    D = d_matrices(mesh, materials, slice(None))
    dTe = dT[mesh.tets]
    fe = np.zeros((len(mesh.tets), 30))
    for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
        dNdx, det = tet10.gradients(X, p)
        B = tet10.strain_matrix(dNdx)
        eps_th = np.outer(alpha * (dTe @ tet10.shape_functions(p)), THERMAL_STRAIN)
        fe += np.einsum("eji,ejk,ek,e->ei", B, D, eps_th, det * w)
    f = np.zeros(mesh.n_nodes * 3)
    np.add.at(f, element_dofs(mesh.tets).ravel(), fe.ravel())
    return f


def recover_stress(
    mesh: FEMesh, materials: dict[int, Material], u: np.ndarray, dT: np.ndarray | None = None
) -> np.ndarray:
    """Nodal stresses: evaluated at each element's nodes, averaged over elements."""
    X = mesh.nodes[mesh.tets]
    ue = u[element_dofs(mesh.tets)]
    D = d_matrices(mesh, materials, slice(None))
    alpha = part_property(mesh, materials, "thermal_expansion")
    acc = np.zeros((mesh.n_nodes, 6))
    count = np.zeros(mesh.n_nodes)
    for k, p in enumerate(tet10.NODE_POINTS):
        dNdx, _ = tet10.gradients(X, p)
        eps = np.einsum("eij,ej->ei", tet10.strain_matrix(dNdx), ue)
        if dT is not None:
            eps = eps - np.outer(alpha * dT[mesh.tets[:, k]], THERMAL_STRAIN)
        sig = np.einsum("eij,ej->ei", D, eps)
        np.add.at(acc, mesh.tets[:, k], sig)
        np.add.at(count, mesh.tets[:, k], 1)
    return acc / np.maximum(count, 1)[:, None]


def solve_static(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    progress: Progress | None = None,
    constraints: Constraints | None = None,
) -> StaticResult:
    report = progress or (lambda frac, msg: None)
    n_dof = mesh.n_nodes * 3

    report(0.05, "경계조건 준비 중")
    c = constraints or build_constraints(project, mesh)

    report(0.1, "강성행렬 조립 중")
    K = stiffness_matrix(mesh, materials)
    f_mech = _load_vector(project, mesh, materials)
    thermal = None
    if has_thermal_loads(project):
        thermal = solve_thermal(
            project, mesh, materials, lambda frac, msg: report(0.1 + 0.2 * frac, f"열전달: {msg}")
        )
    dT = temperature_change(project, mesh, thermal)
    f = f_mech if dT is None else f_mech + _thermal_load(mesh, materials, dT)
    if not np.any(f):
        raise SolverError("하중이 없습니다. 힘, 압력, 중력 또는 온도 하중을 추가하세요.")

    report(0.4, "연립방정식 푸는 중")
    scale = K.diagonal().max()
    Kp = K + c.penalty_matrix(n_dof, scale)
    free = np.flatnonzero(~c.fixed)
    near_null = rigid_body_modes(mesh.nodes)[free]
    u = np.zeros(n_dof)
    u[free] = solve_spd(Kp[free][:, free].tocsr(), f[free], near_null)

    size = np.ptp(mesh.nodes, axis=0).max()
    if np.abs(u).max() > 1e3 * size:
        raise SolverError("unstable")

    report(0.85, "응력 계산 중")
    stress = recover_stress(mesh, materials, u, dT)

    residual = K @ u - f
    reactions = []
    for dofs, equations in zip(c.fixture_dofs, c.fixture_equations, strict=True):
        r = np.zeros(3)
        np.add.at(r, dofs % 3, residual[dofs])
        reactions.append(r + c.penalty_forces(u, equations, scale))
    applied = f_mech.reshape(-1, 3).sum(axis=0)

    report(1.0, "완료")
    return StaticResult(
        mesh=mesh,
        materials=materials,
        displacement=u.reshape(-1, 3),
        stress=stress,
        reactions=reactions,
        applied_load=applied,
        thermal=thermal,
    )
