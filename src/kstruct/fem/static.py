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
    make_solver,
    part_property,
    rigid_body_modes,
    stiffness_matrix,
)
from kstruct.fem.bolts import (
    axial_vectors,
    bolt_stiffness_and_preload,
    build_bolts,
    evaluate,
    locked_pulls,
)
from kstruct.fem.constraints import Constraints, build_constraints
from kstruct.fem.contact import ContactSolver, build_contact
from kstruct.fem.loads import bearing_forces, centrifugal_forces, torque_forces
from kstruct.fem.mesh import FEMesh
from kstruct.fem.thermal import ThermalResult, has_thermal_loads, solve_thermal
from kstruct.model import ConnectionKind, LoadKind, Material, Project, with_case

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
    contact: list[tuple[int, int, int]] | None = None  # per contact: (closed, total, slipping)
    bolts: list | None = None  # fem.bolts.BoltResult per bolt
    bolt_nodes: np.ndarray | None = None  # nodes under bolt heads / nuts

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
        elif load.kind == LoadKind.TORQUE:
            f += torque_forces(project, mesh, load)
        elif load.kind == LoadKind.BEARING:
            f += bearing_forces(project, mesh, load)
        elif load.kind == LoadKind.CENTRIFUGAL:
            f += centrifugal_forces(project, mesh, materials, load)
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


@dataclass
class _Stiffness:
    """Everything of a static solve that does not depend on the loads, so the load cases
    of one project share a single assembly and (without contact) a single factorization."""

    constraints: Constraints
    connectors: list
    K: object  # stiffness incl. bolt springs, without penalty rows
    Kp: object  # K + penalty equations
    free: np.ndarray
    near_null: np.ndarray
    scale: float
    solver: Callable | None = None  # factorization of Kp[free][:, free], made on first use


def _stiffness(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    constraints: Constraints | None = None,
) -> _Stiffness:
    n_dof = mesh.n_nodes * 3
    c = constraints or build_constraints(project, mesh, contacts_as_bonded=False)
    connectors = build_bolts(project, mesh)
    K = stiffness_matrix(mesh, materials)
    if connectors:
        K = K + bolt_stiffness_and_preload(connectors, n_dof)[0]
    scale = K.diagonal().max()
    free = np.flatnonzero(~c.fixed)
    return _Stiffness(
        constraints=c,
        connectors=connectors,
        K=K,
        Kp=K + c.penalty_matrix(n_dof, scale),
        free=free,
        near_null=rigid_body_modes(mesh.nodes)[free],
        scale=scale,
    )


def solve_static(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    progress: Progress | None = None,
    constraints: Constraints | None = None,
    shared: _Stiffness | None = None,
) -> StaticResult:
    """`shared` (from _stiffness, same mesh, materials, fixtures, connections and bolts)
    skips the assembly and reuses its factorization; only the loads may differ."""
    report = progress or (lambda frac, msg: None)
    n_dof = mesh.n_nodes * 3

    report(0.05, "경계조건 준비 중")
    contact_connections = [x for x in project.connections if x.kind == ConnectionKind.CONTACT]
    contacts = [build_contact(mesh, x) for x in contact_connections]

    report(0.1, "강성행렬 조립 중")
    st = shared or _stiffness(project, mesh, materials, constraints)
    c, connectors, K, Kp = st.constraints, st.connectors, st.K, st.Kp
    free, near_null, scale = st.free, st.near_null, st.scale
    f_mech = _load_vector(project, mesh, materials)
    thermal = None
    if has_thermal_loads(project):
        thermal = solve_thermal(
            project, mesh, materials, lambda frac, msg: report(0.1 + 0.2 * frac, f"열전달: {msg}")
        )
    dT = temperature_change(project, mesh, thermal)
    f = f_mech if dT is None else f_mech + _thermal_load(mesh, materials, dT)
    pulls = np.zeros(len(connectors))
    unit_solver = None
    if not np.any(f) and not any(c.bolt.preload for c in connectors):
        raise SolverError("하중이 없습니다. 힘, 압력, 중력 또는 온도 하중을 추가하세요.")

    report(0.4, "연립방정식 푸는 중")
    size = np.ptp(mesh.nodes, axis=0).max()

    def linear_solver():
        """Kp on the free DOFs, factorized once per _Stiffness; the returned function
        solves full-size right-hand sides."""
        if st.solver is None:
            st.solver = make_solver(Kp[free][:, free].tocsr(), near_null)
        solve = st.solver

        def run(rhs):
            out = np.zeros(n_dof)
            out[free] = solve(rhs[free])
            return out

        return run

    contact_solver = (
        ContactSolver(Kp, free, contacts, lambda A: make_solver(A.tocsr(), near_null), scale)
        if contacts
        else None
    )

    if any(c.bolt.preload for c in connectors):
        joined = {
            frozenset((x.faces_a[0].part_id, x.faces_b[0].part_id)) for x in project.connections
        }
        for bc in connectors:
            pair = frozenset((bc.bolt.hole_a.part_id, bc.bolt.hole_b.part_id))
            if bc.bolt.preload and pair not in joined:
                raise SolverError("bolt without contact")
        # lock the preload: settle contact under the nominal pulls, then (with that contact
        # state frozen) solve for the pulls that give exactly P in each bolt
        report(0.35, "볼트 체결 상태 계산 중")
        _, f_nominal = bolt_stiffness_and_preload(connectors, n_dof)
        if contact_solver is not None:
            contact_solver.solve(f_nominal)
            unit_solver = contact_solver.current_solver()
        else:
            unit_solver = linear_solver()
        axial = axial_vectors(connectors, n_dof)
        pulls = locked_pulls(connectors, axial, unit_solver)
        f = f + bolt_stiffness_and_preload(connectors, n_dof, pulls)[1]
    if contact_solver is not None:
        report(0.5, "접촉 상태 계산 중 (반복)")
        u = contact_solver.solve(f)[0]
        # only the stabilising springs hold a part that slid or lifted off
        if np.abs(u).max() > 0.2 * size:
            raise SolverError("contact unstable")
    elif unit_solver is not None:
        u = unit_solver(f)  # the preload stage already factorized this matrix
    else:
        u = linear_solver()(f)
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
        contact=[(int(cs.active.sum()), len(cs.active), int(cs.slipping.sum())) for cs in contacts],
        bolts=evaluate(connectors, u, pulls),
        bolt_nodes=np.unique(
            np.concatenate([np.r_[b.weights_a[0], b.weights_b[0]] for b in connectors])
        )
        if connectors
        else np.zeros(0, dtype=int),
    )


def solve_cases(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    progress: Progress | None = None,
) -> list[StaticResult]:
    """One StaticResult per load case, in project.cases order."""
    report = progress or (lambda frac, msg: None)
    n = len(project.cases)
    results = []
    shared = _stiffness(project, mesh, materials) if project.cases else None
    for i, case in enumerate(project.cases):
        prefix = f"[{case.name}] " if n > 1 else ""

        def sub(frac, msg, i=i, prefix=prefix):
            report((i + frac) / n, prefix + msg)

        results.append(solve_static(with_case(project, i), mesh, materials, sub, shared=shared))
    return results
