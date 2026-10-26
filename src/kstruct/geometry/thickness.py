"""Local wall thickness of a solid, per B-rep face.

Rays are shot inward from sample points on each face; the distance to where they leave
the solid again is the thickness there. The per-face median drives the mesh size so thin
walls get at least a few elements through the thickness (a single quadratic tetrahedron
through a wall under-predicts bending stress badly).
"""

from __future__ import annotations

import numpy as np
from OCP.TopoDS import TopoDS_Shape

from kstruct.geometry.tessellate import bounding_diagonal, tessellate

MAX_SAMPLES_PER_FACE = 60


def face_thickness(shape: TopoDS_Shape, quality: float = 2e-3) -> dict[int, float]:
    """face index -> median wall thickness under that face (mm). Faces whose rays never
    hit anything (degenerate tessellation) are left out."""
    surface = tessellate(shape, quality)
    if surface.n_cells == 0:
        return {}
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    points = np.asarray(surface.points, dtype=float)
    face_id = np.asarray(surface.cell_data["face_id"])
    triangles = points[faces]  # (T, 3, 3)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > 0
    triangles, areas, face_id, cross = triangles[keep], areas[keep], face_id[keep], cross[keep]
    normals = cross / (2 * areas[:, None])  # outward: tessellate() orients the triangles
    centers = triangles.mean(axis=1)
    diag = bounding_diagonal(shape)
    eps = 1e-4 * diag
    rng = np.random.default_rng(0)

    origins, directions, owners = [], [], []
    for fi in np.unique(face_id):
        cells = np.flatnonzero(face_id == fi)
        if len(cells) > MAX_SAMPLES_PER_FACE:
            p = areas[cells] / areas[cells].sum()
            cells = rng.choice(cells, MAX_SAMPLES_PER_FACE, replace=False, p=p)
        origins.append(centers[cells] - normals[cells] * eps)
        directions.append(-normals[cells])
        owners.append(np.full(len(cells), fi))
    origins = np.vstack(origins)
    directions = np.vstack(directions)
    owners = np.concatenate(owners)

    distance = _first_hit(origins, directions, triangles, eps) + eps
    out = {}
    for fi in np.unique(owners):
        d = distance[owners == fi]
        d = d[np.isfinite(d) & (d > 2 * eps)]
        if len(d):
            out[int(fi)] = float(np.median(d))
    return out


def _first_hit(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray, eps: float):
    """Möller–Trumbore ray/triangle test, vectorised over rays × triangles in chunks.
    Returns the distance to the nearest hit further than eps along each ray (nan: none)."""
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    out = np.full(len(origins), np.nan)
    chunk = max(1, 4_000_000 // max(len(triangles), 1))
    for start in range(0, len(origins), chunk):
        o = origins[start : start + chunk, None, :]  # (R, 1, 3)
        d = directions[start : start + chunk, None, :]
        p = np.cross(d, e2[None])  # (R, T, 3)
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0[None]
        u = np.einsum("rtk,rtk->rt", s, p) * inv
        q = np.cross(s, e1[None])
        v = np.einsum("rtk,rk->rt", q, directions[start : start + chunk]) * inv
        t = np.einsum("rtk,tk->rt", q, e2) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > eps)
        t = np.where(hit, t, np.inf)
        nearest = t.min(axis=1)
        out[start : start + chunk] = np.where(np.isfinite(nearest), nearest, np.nan)
    return out
