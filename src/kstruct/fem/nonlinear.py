"""Nonlinear static analysis: yielding (elastic–plastic material) and large deformation.

* Material: J2 plasticity with linear isotropic hardening (fem/plasticity.py).
* Geometry: total Lagrangian — Green–Lagrange strain E = ½(FᵀF − I), second
  Piola–Kirchhoff stress S, nonlinear strain-displacement matrix B(F) and the initial
  stress (geometric) stiffness. Plasticity works additively on E, which is exact for
  large rotations and adequate for the moderate strains of metal parts.
* Solution: the load factor λ goes from 0 to 1 in increments, each solved by Newton's
  method with the consistent tangent; an increment that does not converge is halved.
  Loads are dead loads (a pressure keeps its original direction). Imposed displacements
  grow with λ.
* After the full load, a part that yielded is unloaded in two steps: what remains is the
  permanent deformation.

Where the load cannot be carried (Newton fails even with tiny steps, the tangent turns
singular, or the plastic strain runs away) the analysis stops and reports the load
factor it reached: the part collapses at about that share of the load.

Constraints are eliminated exactly where possible (u = T û, as for the multigrid path of
the linear solver), else enforced by penalty springs. Contact, bolts and the per-case
reuse of factorizations of the linear solver are not supported here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from kstruct.fem import multigrid, tet10
from kstruct.fem.assembly import (
    CHUNK,
    SolverError,
    element_dofs,
    element_materials,
    make_solver,
    part_property,
    rigid_body_modes,
)
from kstruct.fem.constraints import Constraints, build_constraints
from kstruct.fem.mesh import FEMesh
from kstruct.fem.plasticity import point_material, return_map
from kstruct.i18n import tr
from kstruct.model import ConnectionKind, Material, Project

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

FIRST_STEP = 0.1  # load factor of the first increment
MIN_STEP = 1e-3  # smallest increment before the load is declared not carried
MAX_NEWTON = 12  # iterations per increment
TOLERANCE = 1e-5  # residual norm relative to the load (or to the first residual)
RUNAWAY_PLASTIC_STRAIN = 0.5  # beyond this the part has collapsed for any practical use
ITERATIVE_MIN_DOF = 30_000  # p-multigrid CG for the Newton systems from this size
_THERMAL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_SHEAR_PAIRS = ((0, 1), (1, 2), (2, 0))  # Voigt 3, 4, 5


class NotCarried(Exception):
    """The current increment cannot be completed (no convergence / singular tangent)."""


@dataclass
class NonlinearInfo:
    plasticity: bool
    large_deformation: bool
    load_factor: float  # share of the full load that was carried (1.0 = all)
    plastic_strain: np.ndarray  # (N,) equivalent plastic strain, nodal average
    permanent: np.ndarray | None  # (N, 3) displacement left after unloading (mm)
    increments: int
    iterations: int
    limit_reason: str = ""  # why the load was not carried in full

    @property
    def completed(self) -> bool:
        return self.load_factor >= 1.0 - 1e-9

    @property
    def yielded(self) -> bool:
        return bool(self.plastic_strain.max() > 0)


# --- element routine ------------------------------------------------------------------


class _Elements:
    """Gradients at the four Gauss points of every element (computed once) and the
    material constants per point."""

    def __init__(self, mesh: FEMesh, materials: dict[int, Material], dT_nodes):
        X = mesh.nodes[mesh.tets]
        self.dN: list[np.ndarray] = []
        self.dv: list[np.ndarray] = []
        for p, w in zip(tet10.GAUSS_POINTS, tet10.GAUSS_WEIGHTS, strict=True):
            dNdx, det = tet10.gradients(X, p)
            self.dN.append(dNdx)
            self.dv.append(det * w)
        self.mat = point_material(element_materials(mesh, materials))
        alpha = part_property(mesh, materials, "thermal_expansion")
        self.thermal: list[np.ndarray | None] = []
        for p in tet10.GAUSS_POINTS:
            if dT_nodes is None:
                self.thermal.append(None)
            else:
                self.thermal.append(alpha * (dT_nodes[mesh.tets] @ tet10.shape_functions(p)))
        self.n = len(mesh.tets)


def _deformation_gradient(ue: np.ndarray, dN: np.ndarray) -> np.ndarray:
    """F = I + Σ_a u_a ⊗ ∇N_a, (E, 3, 3)."""
    return np.eye(3) + np.einsum("eai,eaj->eij", ue, dN)


def _strain_and_B(ue, dN, large: bool):
    """Voigt strain (E, 6) and its derivative B (E, 6, 30) with respect to the element
    displacements."""
    if not large:
        B = tet10.strain_matrix(dN)
        return (B @ ue.reshape(len(ue), 30, 1))[:, :, 0], B, None
    F = _deformation_gradient(ue, dN)
    C = np.einsum("eki,ekj->eij", F, F)
    strain = np.empty((len(ue), 6))
    strain[:, 0] = 0.5 * (C[:, 0, 0] - 1)
    strain[:, 1] = 0.5 * (C[:, 1, 1] - 1)
    strain[:, 2] = 0.5 * (C[:, 2, 2] - 1)
    for row, (k, m) in zip((3, 4, 5), _SHEAR_PAIRS, strict=True):
        strain[:, row] = C[:, k, m]  # engineering shear 2 E_km
    B = np.zeros((len(ue), 6, 10, 3))
    for k in range(3):  # δE_kk = Σ F_ik ∂N_a/∂X_k δu_ai
        B[:, k] = dN[:, :, k, None] * F[:, None, :, k]
    for row, (k, m) in zip((3, 4, 5), _SHEAR_PAIRS, strict=True):
        B[:, row] = dN[:, :, m, None] * F[:, None, :, k] + dN[:, :, k, None] * F[:, None, :, m]
    return strain, B.reshape(len(ue), 6, 30), F


def _evaluate(
    el: _Elements,
    mesh: FEMesh,
    u: np.ndarray,
    state: tuple[np.ndarray, np.ndarray],
    large: bool,
    want_matrix: bool = True,
):
    """Internal forces, tangent stiffness (or None) and the trial plastic state for the
    displacement u. state = (plastic strain (E, 4, 6), alpha (E, 4)), committed."""
    n_dof = mesh.n_nodes * 3
    eps_p, alpha = state
    new_eps_p, new_alpha = eps_p.copy(), alpha.copy()
    f_int = np.zeros(n_dof)
    K = sp.csr_matrix((n_dof, n_dof)) if want_matrix else None
    S_points = np.zeros((el.n, 4, 6))
    for start in range(0, el.n, CHUNK):
        sl = slice(start, min(start + CHUNK, el.n))
        tets = mesh.tets[sl]
        dofs = element_dofs(tets)
        ue = u[dofs].reshape(len(tets), 10, 3)
        fe = np.zeros((len(tets), 30))
        Ke = np.zeros((len(tets), 30, 30)) if want_matrix else None
        mat = el.mat.subset(sl)
        for g in range(4):
            dN, dv = el.dN[g][sl], el.dv[g][sl]
            strain, B, _ = _strain_and_B(ue, dN, large)
            if el.thermal[g] is not None:
                strain = strain - np.outer(el.thermal[g][sl], _THERMAL)
            S, D, ep, a = return_map(strain, eps_p[sl, g], alpha[sl, g], mat, want_matrix)
            new_eps_p[sl, g], new_alpha[sl, g] = ep, a
            S_points[sl, g] = S
            fe += np.einsum("eji,ej->ei", B, S) * dv[:, None]
            if want_matrix:
                Ke += (B.transpose(0, 2, 1) @ (D @ B)) * dv[:, None, None]
                if large:  # initial stress stiffness: ∇N_a · S · ∇N_b on each axis
                    St = _tensor(S)
                    Gab = np.einsum("eai,eij,ebj->eab", dN, St, dN) * dv[:, None, None]
                    for k in range(3):
                        Ke[:, k::3, k::3] += Gab
        np.add.at(f_int, dofs.ravel(), fe.ravel())
        if want_matrix:
            rows = np.repeat(dofs, 30, axis=1).ravel()
            cols = np.tile(dofs, (1, 30)).ravel()
            K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
    return f_int, K, (new_eps_p, new_alpha), S_points


def _tensor(s: np.ndarray) -> np.ndarray:
    T = np.empty((len(s), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
    T[:, 2, 0] = T[:, 0, 2] = s[:, 5]
    return T


# --- constraint handling --------------------------------------------------------------


class _Space:
    """The unknowns of the Newton systems: û with u = T û + λ u_set when the constraints
    can be eliminated exactly, else the free DOFs with penalty springs P."""

    def __init__(self, c: Constraints, mesh: FEMesh, scale: float):
        self.c, self.mesh, self.scale = c, mesh, scale
        self.n_dof = mesh.n_nodes * 3
        self.u_set = c.prescribed()
        eliminated = c.elimination()
        if eliminated is not None:
            self.T, self.unknowns = eliminated
            self.TT = self.T.T.tocsr()
            self.P = None
        else:
            free = np.flatnonzero(~c.fixed)
            self.T = sp.csr_matrix(
                (np.ones(len(free)), (free, np.arange(len(free)))), shape=(self.n_dof, len(free))
            )
            self.TT = self.T.T.tocsr()
            self.unknowns = free
            self.P = c.penalty_matrix(self.n_dof, scale)
        self.springs = c.spring_matrix() if c.springs else None
        self.near_null = rigid_body_modes(mesh.nodes)[self.unknowns]

    @property
    def exact(self) -> bool:
        return self.P is None

    def internal(self, f_int: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the linear parts (supports, penalty springs) to the internal forces."""
        if self.springs is not None:
            f_int = f_int + self.springs @ u
        if self.P is not None:
            f_int = f_int + self.P @ u
        return f_int

    def tangent(self, K: sp.csr_matrix) -> sp.csr_matrix:
        if self.springs is not None:
            K = K + self.springs
        if self.P is not None:
            K = K + self.P
        return (self.TT @ K @ self.T).tocsr()

    def solver(self, A: sp.csr_matrix):
        """Solve A x = b for the reduced tangent: p-multigrid CG for large exact systems,
        a factorization otherwise. A singular / indefinite tangent means the structure
        can no longer carry an increase of the load."""
        if self.exact and A.shape[0] >= ITERATIVE_MIN_DOF:
            from kstruct.fem.static import _coarse_dofs

            coarse_free = np.flatnonzero(~self.c.fixed[_coarse_dofs(self.mesh)])
            P = multigrid.dof_prolongation(self.mesh)[self.unknowns][:, coarse_free]
            try:
                pmg = multigrid.PMGSolver(A, P)

                def run(b):
                    try:
                        return pmg.solve(b)
                    except multigrid.NotSuitable:
                        return make_solver(A, self.near_null)(b)

                return run
            except multigrid.NotSuitable:
                pass
        try:
            return make_solver(A, self.near_null)
        except SolverError as e:
            raise NotCarried("singular tangent") from e


# --- driver ---------------------------------------------------------------------------


def unsupported(project: Project) -> str | None:
    """Why this project cannot be solved nonlinearly (None: it can)."""
    if any(c.kind == ConnectionKind.CONTACT for c in project.connections):
        return tr("접촉 연결이 있는 모델은 비선형(항복·큰 변형) 해석을 아직 지원하지 않습니다.")
    if project.bolts:
        return tr("볼트가 있는 모델은 비선형(항복·큰 변형) 해석을 아직 지원하지 않습니다.")
    return None


def solve_nonlinear(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    progress: Progress | None = None,
):
    """Static analysis with yielding and/or large deformation (project.plasticity,
    project.large_deformation). Returns a StaticResult with `nonlinear` filled in."""
    from kstruct.fem.static import (
        StaticResult,
        _load_vector,
        temperature_change,
    )
    from kstruct.fem.thermal import has_thermal_loads, solve_thermal

    report = progress or (lambda frac, msg: None)
    reason = unsupported(project)
    if reason:
        raise SolverError(reason)
    plastic = bool(getattr(project, "plasticity", False))
    large = bool(getattr(project, "large_deformation", False))
    n_dof = mesh.n_nodes * 3

    report(0.02, tr("경계조건 준비 중"))
    c = build_constraints(project, mesh, contacts_as_bonded=False)
    f_mech = _load_vector(project, mesh, materials)
    thermal = None
    if has_thermal_loads(project):
        thermal = solve_thermal(
            project, mesh, materials, lambda f, m: report(0.02 + 0.08 * f, tr("열전달: {m}", m=m))
        )
    dT = temperature_change(project, mesh, thermal)
    if not plastic:  # large deformation only: nothing yields
        materials_used = {pid: _never_yields(m) for pid, m in materials.items()}
    else:
        materials_used = materials
    el = _Elements(mesh, materials_used, dT)
    zero_state = (np.zeros((el.n, 4, 6)), np.zeros((el.n, 4)))
    # scale of the penalty springs: like the linear solver, from the elastic stiffness
    from kstruct.fem.assembly import stiffness_matrix

    K0 = stiffness_matrix(mesh, materials)
    space = _Space(c, mesh, K0.diagonal().max())
    if not np.any(f_mech) and not np.any(space.u_set) and dT is None:
        raise SolverError(tr("하중이 없습니다. 힘, 압력, 중력 또는 온도 하중을 추가하세요."))
    size = float(np.ptp(mesh.nodes, axis=0).max())

    # thermal strain also ramps with λ: it is part of the load
    def thermal_scale(lam):
        for g in range(4):
            if el.thermal[g] is not None:
                el.thermal[g] = base_thermal[g] * lam

    base_thermal = [t.copy() if t is not None else None for t in el.thermal]

    state = zero_state
    u = np.zeros(n_dof)
    lam, step = 0.0, FIRST_STEP
    increments = iterations = 0
    limit_reason = ""
    stress_points = np.zeros((el.n, 4, 6))

    def increment(target: float, u_start: np.ndarray, committed, lam_start: float):
        """Newton iterations to equilibrium at load factor `target`, starting with a
        tangent predictor: the whole increment (load and imposed movement) is first
        applied through the last converged tangent. Moving only the imposed DOFs and
        iterating from there yields a thin layer of spurious plastic strain next to them
        that Newton's method then chases for many iterations."""
        nonlocal iterations
        moved = space.u_set != 0
        f_ext = target * f_mech
        thermal_scale(lam_start)
        f_int, K, _, _ = _evaluate(el, mesh, u_start, committed, large)
        du_set = np.zeros(n_dof)
        du_set[moved] = (target - lam_start) * space.u_set[moved]
        thermal_scale(target)
        A = space.tangent(K)
        # linearised residual at the end of the increment (thermal strain enters through
        # the internal forces of the next evaluation, not the predictor)
        Kfull = K if space.P is None else K + space.P
        if space.springs is not None:
            Kfull = Kfull + space.springs
        r_pred = space.TT @ (space.internal(f_int, u_start) + Kfull @ du_set - f_ext)
        u = u_start + du_set + space.T @ space.solver(A)(-r_pred)
        reference = max(np.linalg.norm(space.TT @ f_ext), np.linalg.norm(r_pred), 1e-30)
        previous = np.inf
        for _ in range(MAX_NEWTON):
            f_int, K, trial, S = _evaluate(el, mesh, u, committed, large)
            r = space.TT @ (space.internal(f_int, u) - f_ext)
            norm = np.linalg.norm(r)
            if norm <= TOLERANCE * reference:
                return u, trial, S
            if norm > 10 * previous:
                raise NotCarried("diverging")
            previous = norm
            du = space.solver(space.tangent(K))(-r)
            if not np.all(np.isfinite(du)):
                raise NotCarried("singular tangent")
            step_du = space.T @ du
            # backtracking: take a shorter step when the full one makes things worse
            for _ in range(4):
                trial_u = u + step_du
                f_try, _, _, _ = _evaluate(el, mesh, trial_u, committed, large, False)
                r_try = np.linalg.norm(space.TT @ (space.internal(f_try, trial_u) - f_ext))
                if r_try < norm or r_try <= TOLERANCE * reference:
                    break
                step_du = 0.5 * step_du
            u = trial_u
            iterations += 1
            if np.abs(u).max() > 0.5 * size:
                raise NotCarried("runaway displacement")
        raise NotCarried("no convergence")

    report(0.1, tr("비선형 해석 시작"))
    while lam < 1.0 - 1e-12:
        target = min(1.0, lam + step)
        try:
            u_new, trial, S = increment(target, u, state, lam)
            if trial[1].max() > RUNAWAY_PLASTIC_STRAIN:
                raise NotCarried("runaway plastic strain")
        except NotCarried as e:
            step /= 2
            if step < MIN_STEP:
                limit_reason = str(e)
                log.info("nonlinear: load not carried beyond %.3f (%s)", lam, e)
                break
            continue
        u, state, stress_points, lam = u_new, trial, S, target
        increments += 1
        report(
            0.1 + 0.75 * lam,
            tr("하중 {pct:.0f}% 계산 완료 (증분 {n})", pct=lam * 100, n=increments),
        )
        step = min(step * 1.5, 0.25)  # easy increments: larger steps
    if increments == 0:
        raise SolverError(
            tr("처음 하중 단계부터 풀리지 않습니다. 부품이 충분히 고정되었는지 확인하세요.")
            if limit_reason
            else "unstable"
        )

    report(0.88, tr("응력 계산 중"))
    loaded_u = u.copy()
    points = _cauchy(stress_points, el, mesh, u) if large else stress_points.copy()
    # in elements that yielded the stress is capped by the flow stress: a linear
    # extrapolation to the corners overshoots it (676 MPa shown for a 360 MPa material),
    # so those elements contribute their Gauss point average
    yielded = (state[1] > 0).any(axis=1)
    points[yielded] = points[yielded].mean(axis=1, keepdims=True)
    stress = _nodal(mesh, points)
    peeq = np.maximum(_nodal(mesh, state[1][:, :, None])[:, 0], 0.0)
    f_int, _, _, _ = _evaluate(el, mesh, u, state, large, False)
    # equilibrium residual without the penalty springs: at the fixed DOFs it is the support
    # force; on the free DOFs it is what the constraint equations carry
    residual = f_int - lam * f_mech
    if space.springs is not None:
        residual = residual + space.springs @ u
    reactions = _reactions(c, space, residual, u)

    permanent = None
    if plastic and state[1].max() > 0 and lam >= 1.0 - 1e-9:
        report(0.94, tr("하중을 뺀 뒤 남는 변형 계산 중"))
        try:
            for start, target in ((1.0, 0.5), (0.5, 0.0)):
                u, state, _ = increment(target, u, state, start)
            permanent = u.reshape(-1, 3)
        except NotCarried:
            log.info("nonlinear: unloading did not converge; no permanent deformation")
    thermal_scale(lam)  # leave the element data as loaded

    info = NonlinearInfo(
        plasticity=plastic,
        large_deformation=large,
        load_factor=lam,
        plastic_strain=peeq,
        permanent=permanent,
        increments=increments,
        iterations=iterations,
        limit_reason=limit_reason,
    )
    report(1.0, tr("완료"))
    return StaticResult(
        mesh=mesh,
        materials=materials,
        displacement=loaded_u.reshape(-1, 3),
        stress=stress,
        reactions=reactions,
        applied_load=lam * f_mech.reshape(-1, 3).sum(axis=0),
        thermal=thermal,
        contact=[],
        bolts=[],
        bolt_nodes=np.zeros(0, dtype=int),
        nonlinear=info,
    )


def _never_yields(m: Material) -> Material:
    from dataclasses import replace

    return replace(m, yield_strength=np.inf)


def _cauchy(S_points: np.ndarray, el: _Elements, mesh: FEMesh, u: np.ndarray) -> np.ndarray:
    """σ = F S Fᵀ / det F at each Gauss point (the stress a user expects to see)."""
    out = np.empty_like(S_points)
    ue = u[element_dofs(mesh.tets)].reshape(el.n, 10, 3)
    for g in range(4):
        F = _deformation_gradient(ue, el.dN[g])
        sigma = F @ _tensor(S_points[:, g]) @ F.transpose(0, 2, 1) / np.linalg.det(F)[:, None, None]
        out[:, g] = np.stack(
            [
                sigma[:, 0, 0],
                sigma[:, 1, 1],
                sigma[:, 2, 2],
                sigma[:, 0, 1],
                sigma[:, 1, 2],
                sigma[:, 2, 0],
            ],
            axis=1,
        )
    return out


# Gauss point values -> a linear field per element: values at the corners are A⁻¹ g with
# A[g, c] the barycentric coordinate of Gauss point g for corner c
_GP_BARY = np.column_stack(
    [1 - tet10.GAUSS_POINTS.sum(axis=1), tet10.GAUSS_POINTS[:, 0], tet10.GAUSS_POINTS[:, 1],
     tet10.GAUSS_POINTS[:, 2]]
)  # fmt: skip
_TO_CORNERS = np.linalg.inv(_GP_BARY)


def _nodal(mesh: FEMesh, values: np.ndarray) -> np.ndarray:
    """(E, 4, k) Gauss point values -> (N, k) nodal averages (linear extrapolation within
    each element, mid-side nodes halfway between their corners)."""
    corners = np.einsum("cg,egk->eck", _TO_CORNERS, values)
    nodal = np.concatenate(
        [
            corners,
            0.5
            * (corners[:, [i for i, _ in tet10.EDGES]] + corners[:, [j for _, j in tet10.EDGES]]),
        ],
        axis=1,
    )  # (E, 10, k)
    k = values.shape[2]
    n_entries = mesh.tets.size
    scatter = sp.csr_matrix(
        (np.ones(n_entries), (mesh.tets.ravel(), np.arange(n_entries))),
        shape=(mesh.n_nodes, n_entries),
    )
    count = np.asarray(scatter.sum(axis=1)).ravel()
    return (scatter @ nodal.reshape(-1, k)) / np.maximum(count, 1)[:, None]


def _reactions(c: Constraints, space: _Space, residual: np.ndarray, u: np.ndarray):
    """Per fixture (3,) support force from the equilibrium residual."""
    multipliers = c.multiplier_forces(residual) if space.exact else None
    out = []
    spring_force = {i: -(Ks @ u) for i, Ks in c.springs}
    for i, (dofs, equations) in enumerate(zip(c.fixture_dofs, c.fixture_equations, strict=True)):
        r = np.zeros(3)
        np.add.at(r, dofs % 3, residual[dofs])
        if i in spring_force:
            r += spring_force[i].reshape(-1, 3).sum(axis=0)
        elif len(equations):
            if multipliers is not None:
                r += multipliers[equations.start : equations.stop].sum(axis=0)
            else:
                r += c.penalty_forces(u, equations, space.scale)
        out.append(r)
    return out
