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
from kstruct.i18n import tr, tr_data
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
    # transient runs: output times (s) and the temperature field at each, (frames, N);
    # `temperature` is then the field at the end
    times: np.ndarray | None = None
    history: np.ndarray | None = None

    @property
    def transient(self) -> bool:
        return self.times is not None

    def peak(self) -> tuple[float, float]:
        """(highest temperature reached anywhere, the time it was reached; 0 for steady)."""
        if not self.transient:
            return float(self.temperature.max()), 0.0
        per_frame = self.history.max(axis=1)
        i = int(np.argmax(per_frame))
        return float(per_frame[i]), float(self.times[i])

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


def radiation_tangent(
    T: np.ndarray, ambient: float, emissivity: float
) -> tuple[np.ndarray, np.ndarray]:
    """Newton linearisation of q = εσ(T⁴ − T∞⁴) about T: (h, T_ref) with
    q ≈ h (T_new − T_ref), h = 4εσT³ in W/(m²·K). The secant coefficient
    radiation_coefficient() makes the fixed-point iteration oscillate and diverge as soon
    as radiation carries a few watts; the tangent converges quadratically."""
    Ts, Ta = np.maximum(T + KELVIN, 1.0), ambient + KELVIN
    h = 4.0 * emissivity * STEFAN_BOLTZMANN * Ts**3
    q = emissivity * STEFAN_BOLTZMANN * (Ts**4 - Ta**4)
    return h, T - q / h


def _thermal_groups(project: Project, part_ids: list[int]) -> list[set[int]]:
    """Sets of parts that conduct heat to each other through their connections."""
    group = {pid: {pid} for pid in part_ids}
    for c in project.connections:
        a, b = c.faces_a[0].part_id, c.faces_b[0].part_id
        if a in group and b in group and group[a] is not group[b]:
            merged = group[a] | group[b]
            for pid in merged:
                group[pid] = merged
    seen, out = set(), []
    for g in group.values():
        if id(g) not in seen:
            seen.add(id(g))
            out.append(g)
    return out


TRANSIENT_STEPS = 60  # backward-Euler steps over the requested duration
TRANSIENT_FRAMES = 20  # stored output times (plus the start)


def capacity_matrix(mesh: FEMesh, materials: dict[int, Material]) -> sp.csr_matrix:
    """C = ∫ ρ c N Nᵀ dV (consistent, Keast rule; N·mm/K in the internal units: ρ in
    t/mm³ times c in mm²/(s²·K))."""
    from kstruct.fem.assembly import part_property

    rho_c = part_property(mesh, materials, "density") * part_property(
        mesh, materials, "specific_heat"
    )
    X = mesh.nodes[mesh.tets]
    Ce = np.zeros((len(mesh.tets), 10, 10))
    for p, w in zip(tet10._KEAST_POINTS, tet10._KEAST_WEIGHTS, strict=True):
        _, det = tet10.gradients(X, p)
        N = tet10.shape_functions(p)
        Ce += np.einsum("e,i,j->eij", rho_c * det * w, N, N)
    rows = np.repeat(mesh.tets, 10, axis=1).ravel()
    cols = np.tile(mesh.tets, (1, 10)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((Ce.ravel(), (rows, cols)), shape=(n, n))


class _Conduction:
    """Element conduction matrices for k = 1, computed once per mesh. k is constant over
    an element (the element mean temperature picks it), so K(k) only rescales them:
    rebuilding the gradients every transient step cost 19 of 168 s on a 94k-DOF part."""

    def __init__(self, mesh: FEMesh):
        X = mesh.nodes[mesh.tets]
        Ke = np.zeros((len(mesh.tets), 10, 10))
        for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
            dNdx, det = tet10.gradients(X, p)
            Ke += np.einsum("eni,emi,e->enm", dNdx, dNdx, det * w)
        self.unit = Ke
        self.rows = np.repeat(mesh.tets, 10, axis=1).ravel()
        self.cols = np.tile(mesh.tets, (1, 10)).ravel()
        self.n = mesh.n_nodes

    def matrix(self, k: np.ndarray) -> sp.csr_matrix:
        data = (self.unit * np.asarray(k, dtype=float)[:, None, None]).ravel()
        return sp.csr_matrix((data, (self.rows, self.cols)), shape=(self.n, self.n))


def _conductivity_matrix(mesh: FEMesh, k: np.ndarray) -> sp.csr_matrix:
    return _Conduction(mesh).matrix(k)


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
    duration = float(getattr(project, "thermal_duration", 0.0) or 0.0)
    transient = duration > 0
    if transient:
        missing = [m.name for m in materials.values() if m.specific_heat <= 0 or m.density <= 0]
        if missing:
            raise SolverError(
                tr(
                    "시간에 따른 온도 변화를 계산하려면 비열과 밀도가 필요합니다: {names}",
                    names=", ".join(tr_data(m) for m in missing),
                )
            )
    if not transient and not any(ld.kind in TEMPERATURE_REFERENCES for ld in loads):
        raise SolverError(
            tr(
                "온도의 기준이 없습니다. '고정 온도', '대류 냉각', '복사 냉각' 중 하나 이상을 "
                "지정하세요."
            )
        )
    # every group of connected parts needs its own temperature reference: without one its
    # conduction matrix is singular and the solve used to return 0 °C for it, silently
    referenced = {r.part_id for ld in loads if ld.kind in TEMPERATURE_REFERENCES for r in ld.faces}
    # (a transient run starts from a known temperature and needs no reference)
    for group in _thermal_groups(project, [int(p) for p in np.unique(mesh.tet_part)]):
        if not transient and not group & referenced:
            names = ", ".join(f"'{project.part(pid).name}'" for pid in sorted(group))
            raise SolverError(
                tr(
                    "{names} 부품에 온도의 기준이 없습니다. 이 부품에 '고정 온도'·'대류 냉각'·"
                    "'복사 냉각' 중 하나를 지정하거나, 기준이 있는 부품과 접합(맞닿은 면 연결)하세요.",
                    names=names,
                )
            )
    faces = {}
    for i, ld in enumerate(loads):
        tris = mesh.tris[mesh.face_tris(ld.faces)]
        if len(tris) == 0:
            raise SolverError(tr("열 조건을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요."))
        faces[i] = tris
    nonlinear = any(ld.kind == LoadKind.RADIATION for ld in loads) or any(
        m.conductivity_table for m in materials.values()
    )

    # bonded parts conduct heat across the joint: reuse the tie equations (x component)
    report(0.1, tr("열전도 행렬 조립 중"))
    ties = Constraints(fixed=np.zeros(3 * n, dtype=bool))
    for c in project.connections:
        add_bonded(ties, mesh, c)
    scalar = Constraints(fixed=np.zeros(n, dtype=bool))
    for dofs, coefs in zip(ties.penalty_rows, ties.penalty_coefs, strict=True):
        if np.all(dofs % 3 == 0):
            scalar.add_equation(dofs // 3, coefs)
    conduction = _Conduction(mesh)
    K0 = conduction.matrix(element_conductivity(mesh, materials))
    k_depends_on_t = any(m.conductivity_table for m in materials.values())
    tie_matrix = scalar.penalty_matrix(n, K0.diagonal().max())

    def assemble(T_prev: np.ndarray | None):
        """Linear problem at the previous temperatures: (K, f, fixed mask, prescribed T,
        film terms (tris, h, ambient, ∫N dA) for the heat balance)."""
        if T_prev is None or not k_depends_on_t:
            K = K0
        else:
            K = conduction.matrix(element_conductivity(mesh, materials, T_prev))
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
                ambient: np.ndarray | float = ld.ambient
                if ld.kind == LoadKind.CONVECTION:
                    h = ld.magnitude * H_SCALE
                else:  # radiation: tangent at the face temperatures of the last pass
                    surface = (
                        np.full(len(tris), ld.ambient)
                        if T_prev is None
                        else T_prev[tris].mean(axis=1)
                    )
                    h, ambient = radiation_tangent(surface, ld.ambient, ld.magnitude)
                    h = h * H_SCALE
                H, integral_n = _surface_matrices(mesh, tris, h)
                rows = np.repeat(tris, 6, axis=1).ravel()
                cols = np.tile(tris, (1, 6)).ravel()
                K = K + sp.csr_matrix((H.ravel(), (rows, cols)), shape=(n, n))
                source = np.broadcast_to(np.asarray(h * ambient, dtype=float), (len(tris),))
                np.add.at(f, tris.ravel(), (integral_n * source[:, None]).ravel())
                films.append((tris, h, ambient, integral_n))
        return K + tie_matrix, f, fixed, T, films

    def solve_once(T_prev):
        Kp, f, fixed, T, films = assemble(T_prev)
        free = np.flatnonzero(~fixed)
        rhs = f - Kp @ T
        T[free] = solve_spd(Kp[free][:, free].tocsr(), rhs[free], np.ones((len(free), 1)))
        return T, fixed, films

    if transient:
        T, fixed, films, times, history = _march(
            assemble, capacity_matrix(mesh, materials), project, duration, nonlinear, report
        )
        iterations = TRANSIENT_STEPS
        nonlinear = False  # already handled step by step
    else:
        report(0.5, tr("온도 계산 중"))
        T, fixed, films = solve_once(None)
        times = history = None
        iterations = 1
    if nonlinear:
        for iterations in range(2, MAX_ITERATIONS + 1):
            report(
                0.5 + 0.3 * iterations / MAX_ITERATIONS, tr("온도 계산 중 (반복 {n})", n=iterations)
            )
            T_new, fixed, films = solve_once(T)
            change = np.abs(T_new - T).max()
            T = T_new
            if change <= TOLERANCE * max(np.ptp(T), 1.0):
                break
        else:
            raise SolverError("not converged")

    report(0.85, tr("열 흐름 계산 중"))
    k = element_conductivity(mesh, materials, T if nonlinear else None)
    flux = _nodal_flux(mesh, k, T)
    # Heat entering and leaving are summed separately: netting them made a bar between a
    # hot and a cold face report "0 W in, 0 W out" while 4 W flowed through it.
    flows = [ld.magnitude * WATT for ld in loads if ld.kind == LoadKind.HEAT_FLOW]
    # fixed-temperature faces: the conduction "reaction" is the heat each one supplies
    supplied = _conductivity_matrix(mesh, k) @ T
    for ld in loads:
        if ld.kind == LoadKind.FACE_TEMPERATURE:
            flows.append(float(supplied[mesh.face_nodes(ld.faces)].sum()))
    for tris, h, ambient, integral_n in films:
        per_face = (integral_n * T[tris]).sum(axis=1) - ambient * integral_n.sum(axis=1)
        flows.append(-float((h * per_face).sum()))
    heat_in = sum(q for q in flows if q > 0) / WATT
    heat_out = -sum(q for q in flows if q < 0) / WATT
    report(1.0, tr("완료"))
    return ThermalResult(mesh, T, flux, heat_in, heat_out, iterations, times, history)


def _march(assemble, C, project: Project, duration: float, nonlinear: bool, report):
    """Backward Euler from the initial temperature: (K + C/Δt) Tₙ₊₁ = f + C/Δt Tₙ with the
    fixed-temperature faces held from the first step on. Linear problems factorize once;
    radiation and temperature-dependent conductivity are linearised at the previous
    step's temperatures (one solve per step)."""
    from kstruct.fem.assembly import make_solver

    n = C.shape[0]
    T = np.full(n, float(getattr(project, "initial_temperature", 20.0)))
    dt = duration / TRANSIENT_STEPS
    every = max(TRANSIENT_STEPS // TRANSIENT_FRAMES, 1)
    times, history = [0.0], [T.copy()]
    solver = None
    fixed, films = np.zeros(n, dtype=bool), []
    for step in range(1, TRANSIENT_STEPS + 1):
        report(
            0.5 + 0.35 * step / TRANSIENT_STEPS,
            tr("시간에 따른 온도 계산 중 ({step}/{total})", step=step, total=TRANSIENT_STEPS),
        )
        if solver is None or nonlinear:
            Kp, f, fixed, T_fix, films = assemble(T if nonlinear else None)
            A = (Kp + C / dt).tocsr()
            free = np.flatnonzero(~fixed)
            solver = make_solver(A[free][:, free].tocsr(), np.ones((len(free), 1)))
        T_next = T_fix.copy()
        rhs = f + C @ T / dt - A @ T_fix
        T_next[free] = solver(rhs[free])
        T = T_next
        if step % every == 0 or step == TRANSIENT_STEPS:
            times.append(step * dt)
            history.append(T.copy())
    return T, fixed, films, np.array(times), np.array(history)


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
