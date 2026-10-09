"""Boundary conditions: fixtures and bonded connections.

Exact DOF elimination is used where the constrained direction is a global axis (fixed
faces, axis-aligned rollers). Everything else (inclined rollers, pins, bonded ties between
non-matching part meshes) is enforced with stiff penalty springs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from scipy.spatial import cKDTree

from kstruct.fem import tet10
from kstruct.fem.mesh import FEMesh
from kstruct.geometry.tessellate import faces_of
from kstruct.model import Connection, FaceRef, FixtureKind, Project

PENALTY_FACTOR = 1e6  # relative to the largest stiffness diagonal
AXIS_TOL = 1e-6


class ConstraintError(Exception):
    pass


@dataclass
class Constraints:
    fixed: np.ndarray  # (3N,) bool, DOFs held at zero
    penalty_rows: list[np.ndarray] = field(default_factory=list)  # each (k,) dof indices
    penalty_coefs: list[np.ndarray] = field(default_factory=list)  # each (k,) coefficients
    # per fixture, for reactions: DOFs it eliminated, and its range of penalty equations
    fixture_dofs: list[np.ndarray] = field(default_factory=list)
    fixture_equations: list[range] = field(default_factory=list)

    def add_equation(self, dofs: np.ndarray, coefs: np.ndarray) -> None:
        """Penalty-enforce sum(coefs * u[dofs]) == 0."""
        self.penalty_rows.append(np.asarray(dofs))
        self.penalty_coefs.append(np.asarray(coefs, dtype=float))

    def penalty_forces(self, u: np.ndarray, equations: range, stiffness_scale: float) -> np.ndarray:
        """(3,) total force the given penalty equations exert on the structure."""
        alpha = PENALTY_FACTOR * stiffness_scale
        total = np.zeros(3)
        for i in equations:
            dofs, c = self.penalty_rows[i], self.penalty_coefs[i]
            force = -alpha * c * (c @ u[dofs])
            np.add.at(total, dofs % 3, force)
        return total

    def penalty_matrix(self, n_dof: int, stiffness_scale: float) -> sp.csr_matrix:
        if not self.penalty_rows:
            return sp.csr_matrix((n_dof, n_dof))
        rows, cols, vals = [], [], []
        for dofs, c in zip(self.penalty_rows, self.penalty_coefs, strict=True):
            rows.append(np.repeat(dofs, len(dofs)))
            cols.append(np.tile(dofs, len(dofs)))
            vals.append(np.outer(c, c).ravel())
        alpha = PENALTY_FACTOR * stiffness_scale
        return sp.csr_matrix(
            (alpha * np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_dof, n_dof),
        )


def _node_normals(mesh: FEMesh, refs: list[FaceRef]) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted outward normals at the nodes of the given faces."""
    tri_idx = mesh.face_tris(refs)
    tris = mesh.tris[tri_idx]
    _, normal, area = tet10.tri6_load_weights(mesh.nodes[tris])
    acc = np.zeros((mesh.n_nodes, 3))
    np.add.at(acc, tris.ravel(), np.repeat(normal * area[:, None], 6, axis=0))
    nodes = np.unique(tris)
    n = acc[nodes]
    return nodes, n / np.linalg.norm(n, axis=1, keepdims=True)


def _surface_kind(project: Project, ref: FaceRef) -> BRepAdaptor_Surface:
    return BRepAdaptor_Surface(faces_of(project.part(ref.part_id).shape)[ref.face_index])


def build_constraints(project: Project, mesh: FEMesh) -> Constraints:
    if not project.fixtures:
        raise ConstraintError("고정이 하나도 없습니다. 부품이 공중에 떠 있어 해석할 수 없습니다.")
    c = Constraints(fixed=np.zeros(mesh.n_nodes * 3, dtype=bool))

    for fixture in project.fixtures:
        nodes = mesh.face_nodes(fixture.faces)
        if len(nodes) == 0:
            raise ConstraintError("고정할 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
        already_fixed = c.fixed.copy()
        first_equation = len(c.penalty_rows)

        if fixture.kind == FixtureKind.FIXED:
            c.fixed[(nodes[:, None] * 3 + np.arange(3)).ravel()] = True

        elif fixture.kind == FixtureKind.ROLLER:
            for ref in fixture.faces:
                surf = _surface_kind(project, ref)
                face_nodes, normals = _node_normals(mesh, [ref])
                if surf.GetType() == GeomAbs_Plane:
                    n = np.array(surf.Plane().Axis().Direction().Coord())
                    axis = int(np.argmax(np.abs(n)))
                    if 1 - abs(n[axis]) < AXIS_TOL:
                        c.fixed[face_nodes * 3 + axis] = True
                        continue
                for node, n in zip(face_nodes, normals, strict=True):
                    c.add_equation(node * 3 + np.arange(3), n)

        elif fixture.kind == FixtureKind.PIN:
            for ref in fixture.faces:
                surf = _surface_kind(project, ref)
                if surf.GetType() != GeomAbs_Cylinder:
                    raise ConstraintError("핀 고정은 원통면(구멍이나 축)에만 지정할 수 있습니다.")
                ax = surf.Cylinder().Axis()
                origin = np.array(ax.Location().Coord())
                direction = np.array(ax.Direction().Coord())
                face_nodes = mesh.face_nodes([ref])
                rel = mesh.nodes[face_nodes] - origin
                radial = rel - np.outer(rel @ direction, direction)
                radial /= np.linalg.norm(radial, axis=1, keepdims=True)
                for node, r in zip(face_nodes, radial, strict=True):
                    dofs = node * 3 + np.arange(3)
                    c.add_equation(dofs, r)
                    c.add_equation(dofs, direction)

        # a DOF shared by two fixtures (edge between faces) counts toward the first one
        c.fixture_dofs.append(np.flatnonzero(c.fixed & ~already_fixed))
        c.fixture_equations.append(range(first_equation, len(c.penalty_rows)))

    for connection in project.connections:
        add_bonded(c, mesh, connection)
    return c


# --- bonded (tie) connections -------------------------------------------------------


def _tri6_shape(l0: np.ndarray, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            l0 * (2 * l0 - 1),
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            4 * l1 * l2,
            4 * l0 * l2,
            4 * l0 * l1,
        ],
        axis=-1,
    )


def _project_to_triangles(p: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Closest point on triangles ABC (vectorized). Returns barycentrics and distance."""
    ab, ac, ap = B - A, C - A, p - A
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    # clamp into the triangle (approximate, good enough for choosing the closest triangle)
    v, w = np.clip(v, 0, 1), np.clip(w, 0, 1)
    s = v + w
    over = s > 1
    v[over], w[over] = v[over] / s[over], w[over] / s[over]
    u = 1 - v - w
    q = u[:, None] * A + v[:, None] * B + w[:, None] * C
    return u, v, w, np.linalg.norm(p - q, axis=1)


def add_bonded(c: Constraints, mesh: FEMesh, connection: Connection, tol: float | None = None):
    """Tie every node of faces_b to the surface of faces_a (u_b == interpolated u_a)."""
    master = mesh.tris[mesh.face_tris(connection.faces_a)]
    slave_nodes = mesh.face_nodes(connection.faces_b)
    if len(master) == 0 or len(slave_nodes) == 0:
        raise ConstraintError("접합할 면이 메시에 없습니다.")
    X = mesh.nodes
    A, B, C = X[master[:, 0]], X[master[:, 1]], X[master[:, 2]]
    edge = np.linalg.norm(B - A, axis=1).mean()
    tol = tol if tol is not None else 0.25 * edge

    tree = cKDTree((A + B + C) / 3)
    k = min(8, len(master))
    _, cand = tree.query(X[slave_nodes], k=k)
    cand = cand.reshape(len(slave_nodes), k)

    for node, tris in zip(slave_nodes, cand, strict=True):
        p = np.repeat(X[node][None], len(tris), axis=0)
        u, v, w, dist = _project_to_triangles(p, A[tris], B[tris], C[tris])
        best = int(np.argmin(dist))
        if dist[best] > tol:
            continue  # node not on the shared area
        N = _tri6_shape(u[best], v[best], w[best])
        tri_nodes = master[tris[best]]
        for d in range(3):
            c.add_equation(np.concatenate([[node * 3 + d], tri_nodes * 3 + d]), np.r_[1.0, -N])
