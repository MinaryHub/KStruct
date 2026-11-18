"""Natural frequencies and mode shapes: K φ = ω² M φ."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from kstruct.fem.assembly import (
    SolverError,
    inverse_operator,
    mass_matrix,
    rigid_body_modes,
    stiffness_matrix,
)
from kstruct.fem.bolts import bolt_stiffness_and_preload, build_bolts
from kstruct.fem.constraints import build_constraints
from kstruct.fem.mesh import FEMesh
from kstruct.model import Material, Project

Progress = Callable[[float, str], None]
RIGID_HZ = 1e-2  # modes below this are rigid-body motion of an unheld part


@dataclass
class ModalResult:
    mesh: FEMesh
    frequencies: np.ndarray  # (m,) Hz, ascending
    shapes: np.ndarray  # (m, N, 3) mass-normalised displacement shapes
    participation: np.ndarray  # (m, 3) effective mass fraction in X, Y, Z
    rigid_modes: int  # number of ~0 Hz modes dropped (free-floating parts)

    def near(self, frequency: float, band: float = 0.2) -> list[int]:
        """Modes within ±band of an excitation frequency (resonance risk)."""
        if frequency <= 0:
            return []
        return [i for i, f in enumerate(self.frequencies) if abs(f - frequency) <= band * frequency]


def solve_modal(
    project: Project,
    mesh: FEMesh,
    materials: dict[int, Material],
    n_modes: int = 6,
    progress: Progress | None = None,
) -> ModalResult:
    report = progress or (lambda frac, msg: None)
    n_dof = mesh.n_nodes * 3
    report(0.05, "경계조건 준비 중")
    if project.fixtures:
        c = build_constraints(project, mesh)
    else:  # free-free modal is legitimate: rigid modes come out at ~0 Hz and are skipped
        from kstruct.fem.constraints import Constraints, add_bonded

        c = Constraints(fixed=np.zeros(n_dof, dtype=bool))
        for conn in project.connections:
            add_bonded(c, mesh, conn)

    report(0.15, "강성·질량 행렬 조립 중")
    K = stiffness_matrix(mesh, materials)
    K = K + c.penalty_matrix(n_dof, K.diagonal().max()) + c.spring_matrix()
    K = K + bolt_stiffness_and_preload(build_bolts(project, mesh), n_dof)[0]
    M = mass_matrix(mesh, materials)
    free = np.flatnonzero(~c.fixed)
    Kf, Mf = K[free][:, free].tocsc(), M[free][:, free].tocsc()

    report(0.4, "고유진동수 계산 중")
    extra = 0 if project.fixtures else 6
    k = min(n_modes + extra, len(free) - 2)
    # shift slightly below zero so rigid-body (zero) modes don't make K - σM singular
    sigma = -1.0 * (2 * np.pi) ** 2
    try:
        # K - σM = K + |σ|M is positive definite even for a free-floating part
        OPinv = inverse_operator(Kf - sigma * Mf, rigid_body_modes(mesh.nodes)[free])
        omega2, phi = spla.eigsh(Kf, k=k, M=Mf, sigma=sigma, which="LM", OPinv=OPinv)
    except (RuntimeError, spla.ArpackNoConvergence) as e:
        raise SolverError("not converged") from e
    order = np.argsort(omega2)
    omega2, phi = omega2[order], phi[:, order]
    freq = np.sqrt(np.clip(omega2, 0, None)) / (2 * np.pi)
    keep = freq > RIGID_HZ
    rigid = int((~keep).sum())
    if project.fixtures and rigid:
        raise SolverError("unstable")
    freq, phi = freq[keep][:n_modes], phi[:, keep][:, :n_modes]

    shapes = np.zeros((len(freq), n_dof))
    shapes[:, free] = phi.T
    # effective modal mass fraction per direction: (φᵀ M r)² / (φᵀ M φ) / total mass
    participation = np.zeros((len(freq), 3))
    total = np.zeros(3)
    for d in range(3):
        r = np.zeros(n_dof)
        r[d::3] = 1.0
        Mr = M @ r
        total[d] = r @ Mr
        participation[:, d] = (shapes @ Mr) ** 2 / np.einsum("mi,mi->m", shapes @ M, shapes)
    participation /= total

    report(1.0, "완료")
    return ModalResult(mesh, freq, shapes.reshape(len(freq), -1, 3), participation, rigid)
