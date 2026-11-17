"""Nodal force vectors of the rotating-machinery loads: torque, bearing load, centrifugal.

Surface loads are integrated triangle by triangle with `tet10.tri6_load_weights` (uniform
traction per flat TRI6), so each distribution is scaled to give the requested resultant
exactly (Σ r × t dA = T for torque, Σ t dA = F for the bearing load).
"""

from __future__ import annotations

import numpy as np

from kstruct.fem import tet10
from kstruct.fem.assembly import SolverError, part_property
from kstruct.fem.mesh import FEMesh
from kstruct.geometry.face_info import common_axis
from kstruct.geometry.tessellate import faces_of
from kstruct.model import Load, LoadKind, Material, Project


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise SolverError("하중의 축 방향이 정해지지 않았습니다. 하중을 다시 추가해 주세요.")
    return v / n


def load_axis(project: Project, load: Load) -> tuple[np.ndarray, np.ndarray]:
    """(point, unit direction) of the load's axis. Bearing loads always use the axis of the
    selected cylindrical hole; torque and centrifugal loads use the axis stored on the load
    (point + vector) and fall back to a selected cylindrical face."""
    faces = [faces_of(project.part(r.part_id).shape)[r.face_index] for r in load.faces]
    from_faces = common_axis(faces) if faces else None
    if load.kind == LoadKind.BEARING:
        if from_faces is None:
            raise SolverError(
                "베어링 하중은 구멍(원통면)에만 줄 수 있습니다. 구멍 안쪽 면을 선택해 주세요."
            )
        return from_faces
    if load.point is not None and any(load.vector):
        return np.asarray(load.point, dtype=float), _unit(load.vector)
    if from_faces is None:
        raise SolverError(
            "토크·원심력의 회전축을 정할 수 없습니다. 원통면을 고르거나 축 방향을 지정하세요."
        )
    return from_faces


def _radial(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = points - origin
    return rel - np.outer(rel @ axis, axis)


def torque_forces(project: Project, mesh: FEMesh, load: Load) -> np.ndarray:
    """Tangential traction growing linearly with the radius (the elastic torsion shear
    distribution), scaled so the moment about the axis equals load.magnitude (N·mm).
    Positive torque follows the right-hand rule about load.vector."""
    origin, axis = load_axis(project, load)
    tris = mesh.tris[mesh.face_tris(load.faces)]
    if len(tris) == 0:
        raise SolverError("하중을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
    weights, _, area = tet10.tri6_load_weights(mesh.nodes[tris])
    centers = mesh.nodes[tris[:, :3]].mean(axis=1)
    radial = _radial(centers, origin, axis)
    r = np.linalg.norm(radial, axis=1)
    tangent = np.cross(axis, radial)  # |tangent| = r, direction of positive rotation
    second_moment = float((area * r**2).sum())
    if second_moment <= 0:
        raise SolverError("토크를 줄 면이 회전축 위에 있습니다. 축에서 떨어진 면을 고르세요.")
    k = load.magnitude / second_moment  # traction = k * r, tangential
    traction = k * tangent  # (S, 3): k * r * unit tangent
    f = np.zeros((mesh.n_nodes, 3))
    np.add.at(f, tris.ravel(), (weights[:, :, None] * traction[:, None, :]).reshape(-1, 3))
    return f


def bearing_forces(project: Project, mesh: FEMesh, load: Load) -> np.ndarray:
    """Pin-in-hole load: radial pressure p = p0·cosθ on the half of the hole that faces the
    load direction (zero on the other half), scaled so the resultant equals load.vector."""
    origin, axis = load_axis(project, load)
    force = np.asarray(load.vector, dtype=float)
    radial_force = force - (force @ axis) * axis  # only the radial component bears
    if np.linalg.norm(radial_force) < 0.995 * np.linalg.norm(force):
        # the part along the hole axis would be dropped silently while the condition
        # list and the report still showed the full vector
        raise SolverError(
            "베어링 하중의 방향이 구멍 축과 수직이 아닙니다. 베어링(핀)은 구멍 벽을 옆으로만 "
            "누를 수 있으므로, 축에 수직인 방향을 고르거나 축 방향 힘은 '힘' 하중으로 따로 "
            "지정하세요."
        )
    direction = _unit(radial_force)
    tris = mesh.tris[mesh.face_tris(load.faces)]
    if len(tris) == 0:
        raise SolverError("하중을 줄 면이 메시에 없습니다. 메시를 다시 만들어 주세요.")
    weights, _, area = tet10.tri6_load_weights(mesh.nodes[tris])
    centers = mesh.nodes[tris[:, :3]].mean(axis=1)
    radial = _radial(centers, origin, axis)
    rhat = radial / np.linalg.norm(radial, axis=1, keepdims=True)
    cos = np.maximum(rhat @ direction, 0.0)
    denominator = float((area * cos**2).sum())
    if denominator <= 0:
        raise SolverError("베어링 하중 방향을 마주 보는 구멍 면이 없습니다.")
    p0 = np.linalg.norm(radial_force) / denominator
    traction = (p0 * cos)[:, None] * rhat  # pushes the wall outward from the axis
    f = np.zeros((mesh.n_nodes, 3))
    np.add.at(f, tris.ravel(), (weights[:, :, None] * traction[:, None, :]).reshape(-1, 3))
    # the mesh is not symmetric about the load direction, so the side components of the
    # cosine distribution do not cancel exactly (well under 1 %); spread that remainder
    # over the loaded nodes so the resultant equals the requested force and the reported
    # reactions balance it
    loaded = np.linalg.norm(f, axis=1)
    error = f.sum(axis=0) - radial_force
    f -= np.outer(loaded / loaded.sum(), error)
    return f


def centrifugal_forces(
    project: Project, mesh: FEMesh, materials: dict[int, Material], load: Load
) -> np.ndarray:
    """Body force ρ ω² r directed away from the axis, on every part."""
    origin, axis = load_axis(project, load)
    omega = load.magnitude
    X = mesh.nodes[mesh.tets]  # (E, 10, 3)
    rho = part_property(mesh, materials, "density")
    w = tet10.mass_lumped_volume(X) * rho[:, None]  # (E, 10) nodal mass
    radial = _radial(X.reshape(-1, 3), origin, axis).reshape(X.shape)
    f = np.zeros((mesh.n_nodes, 3))
    np.add.at(f, mesh.tets.ravel(), (w[:, :, None] * omega**2 * radial).reshape(-1, 3))
    return f
