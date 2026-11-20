"""Bolted joints as spring connectors between two hole faces.

Each hole face is represented by the area-weighted average displacement of its nodes.
The bolt couples those two averages with an axial spring (E·As/L) along the hole axis and
a transverse spring (bending of a clamped-clamped shank, 12·E·I/L³) across it. Preload P
is applied as an equal-and-opposite pair pulling the holes together.

Preload is "locked" like a pretension element: the pull applied to each bolt is solved for
so that in the preload-only state every bolt carries exactly its target P; that pull is
then kept while the external loads act. The reported axial force is the force really
acting in the model, N = pull + k_a·elongation, so bolts, contact and fixtures are always
in equilibrium and the external load shares between bolt and clamped parts naturally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
import scipy.sparse as sp
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

from kstruct.fem import tet10
from kstruct.fem.constraints import ConstraintError
from kstruct.fem.mesh import FEMesh
from kstruct.geometry.tessellate import faces_of
from kstruct.i18n import tr
from kstruct.model import Bolt, FaceRef, Project


@dataclass(frozen=True)
class BoltSize:
    name: str
    d: float
    stress_area: float
    clearance: float


@dataclass(frozen=True)
class BoltGrade:
    name: str
    tensile: float  # Rm, MPa
    proof: float  # Rp, MPa
    E: float


@cache
def _data() -> dict:
    text = resources.files("kstruct.resources").joinpath("bolts.json").read_text("utf-8")
    return json.loads(text)


def sizes() -> dict[str, BoltSize]:
    return {
        s["name"]: BoltSize(s["name"], s["d"], s["stress_area"], s["clearance"])
        for s in _data()["sizes"]
    }


def grades() -> dict[str, BoltGrade]:
    return {g["name"]: BoltGrade(g["name"], g["Rm"], g["Rp"], g["E"]) for g in _data()["grades"]}


def suggest_size(hole_diameter: float) -> str:
    """Largest bolt that fits the hole (clearance hole or tapped hole)."""
    fitting = [s for s in sizes().values() if s.d <= hole_diameter + 1e-6]
    return (
        max(fitting, key=lambda s: s.d) if fitting else min(sizes().values(), key=lambda s: s.d)
    ).name


def recommended_preload(size: str, grade: str) -> float:
    """Typical assembly preload: a fraction of the proof load (N)."""
    return _data()["default_preload_ratio"] * grades()[grade].proof * sizes()[size].stress_area


def hole_axis(project: Project, ref: FaceRef) -> tuple[np.ndarray, np.ndarray, float]:
    surf = BRepAdaptor_Surface(faces_of(project.part(ref.part_id).shape)[ref.face_index])
    if surf.GetType() != GeomAbs_Cylinder:
        raise ConstraintError(tr("볼트는 구멍(원통면)끼리만 연결할 수 있습니다."))
    cyl = surf.Cylinder()
    ax = cyl.Axis()
    return np.array(ax.Location().Coord()), np.array(ax.Direction().Coord()), 2 * cyl.Radius()


@dataclass
class BoltConnector:
    bolt: Bolt
    weights_a: tuple[np.ndarray, np.ndarray]  # (nodes, area weights summing to 1)
    weights_b: tuple[np.ndarray, np.ndarray]
    axis: np.ndarray  # unit vector from hole a to hole b
    length: float
    k_axial: float
    k_shear: float

    def gradients(self, n_dof: int) -> list[sp.csr_matrix]:
        """Row vectors c (1 x n_dof): axial elongation and two transverse slips."""
        t1 = np.cross(self.axis, [1.0, 0, 0])
        if np.linalg.norm(t1) < 0.1:
            t1 = np.cross(self.axis, [0, 1.0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(self.axis, t1)
        rows = []
        for direction in (self.axis, t1, t2):
            cols, vals = [], []
            for (nodes, w), sign in ((self.weights_b, 1.0), (self.weights_a, -1.0)):
                for k in range(3):
                    cols.append(nodes * 3 + k)
                    vals.append(sign * w * direction[k])
            rows.append(
                sp.csr_matrix(
                    (
                        np.concatenate(vals),
                        (np.zeros(sum(len(c) for c in cols), int), np.concatenate(cols)),
                    ),
                    shape=(1, n_dof),
                )
            )
        return rows


def _face_weights(mesh: FEMesh, ref: FaceRef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tris = mesh.tris[mesh.face_tris([ref])]
    if len(tris) == 0:
        raise ConstraintError(tr("볼트 구멍 면이 메시에 없습니다. 메시를 다시 만들어 주세요."))
    weights, _, _ = tet10.tri6_load_weights(mesh.nodes[tris])
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, tris.ravel(), weights.ravel())
    nodes = np.flatnonzero(w)
    centre = (mesh.nodes[nodes] * w[nodes, None]).sum(0) / w[nodes].sum()
    return nodes, w[nodes] / w[nodes].sum(), centre


HEAD_RADIUS = 0.8  # bearing radius of head / nut / washer, in nominal diameters


def _bearing_weights(mesh, part_id, origin, axis, s_end, r_in, r_out):
    """Annulus under a bolt head or nut: boundary triangles of the part lying in the plane
    s = s_end (s measured along the bolt axis) between the hole and the head radius."""
    tris = mesh.tris[mesh.tri_part == part_id]
    corners = mesh.nodes[tris[:, :3]]
    rel = corners - origin
    s = rel @ axis
    radial = np.linalg.norm(rel - s[..., None] * axis, axis=2).mean(axis=1)
    tol = 1e-3 * max(r_out, 1.0)
    on_plane = np.all(np.abs(s - s_end) < tol, axis=1)
    in_ring = (radial > r_in * 0.95) & (radial < r_out)
    chosen = tris[on_plane & in_ring]
    if len(chosen) == 0:
        return None
    weights, _, _ = tet10.tri6_load_weights(mesh.nodes[chosen])
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, chosen.ravel(), weights.ravel())
    nodes = np.flatnonzero(w)
    return nodes, w[nodes] / w[nodes].sum()


def build_bolts(project: Project, mesh: FEMesh) -> list[BoltConnector]:
    """Attach each bolt where it really pushes: the annulus under the nut on the far side of
    hole_a and under the head on the far side of hole_b (falls back to the hole wall when
    the hole does not end on a flat face, e.g. a tapped blind hole)."""
    out = []
    for bolt in project.bolts:
        size, grade = sizes()[bolt.size], grades()[bolt.grade]
        na, wa, ca = _face_weights(mesh, bolt.hole_a)
        nb, wb, cb = _face_weights(mesh, bolt.hole_b)
        _, direction, hole_d = hole_axis(project, bolt.hole_a)
        axis = direction if float((cb - ca) @ direction) >= 0 else -direction
        s_a = (mesh.nodes[na] - ca) @ axis
        s_b = (mesh.nodes[nb] - ca) @ axis
        r_in, r_out = hole_d / 2, max(HEAD_RADIUS * size.d, hole_d / 2 * 1.3)
        nut = _bearing_weights(mesh, bolt.hole_a.part_id, ca, axis, s_a.min(), r_in, r_out)
        head = _bearing_weights(mesh, bolt.hole_b.part_id, ca, axis, s_b.max(), r_in, r_out)
        end_a = s_a.min() if nut else 0.0
        end_b = s_b.max() if head else float((cb - ca) @ axis)
        attach_a = nut or (na, wa)
        attach_b = head or (nb, wb)
        # grip length between the bearing planes, at least one diameter
        length = max(end_b - end_a, size.d)
        area = size.stress_area
        inertia = np.pi * size.d**4 / 64
        k_axial = grade.E * area / length
        k_shear = min(12 * grade.E * inertia / length**3, grade.E / 2.6 * area / length)
        out.append(BoltConnector(bolt, attach_a, attach_b, axis, length, k_axial, k_shear))
    return out


def bolt_stiffness_and_preload(
    connectors: list[BoltConnector], n_dof: int, pulls: np.ndarray | None = None
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Bolt spring stiffness and the load vector of the bolt pulls (default: nominal P)."""
    K = sp.csr_matrix((n_dof, n_dof))
    f = np.zeros(n_dof)
    for i, c in enumerate(connectors):
        axial, s1, s2 = c.gradients(n_dof)
        K = K + c.k_axial * (axial.T @ axial) + c.k_shear * (s1.T @ s1 + s2.T @ s2)
        # a pull T draws the holes together: generalized force -T on the elongation
        pull = c.bolt.preload if pulls is None else pulls[i]
        f -= pull * axial.toarray().ravel()
    return K.tocsr(), f


def axial_vectors(connectors: list[BoltConnector], n_dof: int) -> np.ndarray:
    """(n_bolts, n_dof) elongation gradients."""
    return np.array([c.gradients(n_dof)[0].toarray().ravel() for c in connectors])


def locked_pulls(connectors, axial: np.ndarray, unit_response) -> np.ndarray:
    """Pulls T giving exactly the target preload P in every bolt of the preload-only state.

    Bolt i carries N_i = T_i + k_i c_i·u, and u = Σ_j T_j u_j with u_j the response to a
    unit pull on bolt j, so (I + diag(k) C U) T = P.
    """
    targets = np.array([c.bolt.preload for c in connectors])
    if not np.any(targets):
        return targets
    k = np.array([c.k_axial for c in connectors])
    U = np.column_stack([unit_response(-axial[j]) for j in range(len(connectors))])
    A = np.eye(len(connectors)) + k[:, None] * (axial @ U)
    return np.linalg.solve(A, targets)


@dataclass
class BoltResult:
    name: str
    size: str
    grade: str
    preload: float
    axial: float  # N (tension positive)
    shear: float  # N
    stress: float  # equivalent sqrt(σ² + 3τ²) on the stress area, MPa
    proof: float
    safety_factor: float


def evaluate(connectors, u: np.ndarray, pulls: np.ndarray) -> list[BoltResult]:
    out = []
    n_dof = len(u)
    for i, c in enumerate(connectors):
        axial, s1, s2 = c.gradients(n_dof)
        N = pulls[i] + c.k_axial * (axial @ u)[0]
        V = c.k_shear * float(np.hypot((s1 @ u)[0], (s2 @ u)[0]))
        size, grade = sizes()[c.bolt.size], grades()[c.bolt.grade]
        sigma, tau = max(N, 0.0) / size.stress_area, V / size.stress_area
        eq = float(np.sqrt(sigma**2 + 3 * tau**2))
        fos = grade.proof / eq if eq > 0 else 100.0
        out.append(
            BoltResult(
                c.bolt.name or tr("볼트 {n}", n=i + 1),
                c.bolt.size,
                c.bolt.grade,
                c.bolt.preload,
                float(N),
                V,
                eq,
                grade.proof,
                min(fos, 100.0),
            )
        )
    return out
