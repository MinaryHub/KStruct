"""Where a linear FE stress cannot be trusted: stress singularities.

At a sharp re-entrant (concave) edge and along the boundary of a fully fixed face the
elastic stress of a linear model grows without bound as the mesh is refined. Reporting
the raw peak there makes the verdict depend on the mesh. This module marks those nodes
and a small geometric zone around them, so the summary can also report a "hot-spot"
value taken just outside the zone (as weld/notch assessment methods do).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from kstruct.fem import tet10
from kstruct.fem.mesh import FEMesh
from kstruct.model import Project

NONE, FIXTURE_EDGE, CONCAVE_EDGE = 0, 1, 2
KIND_NAMES = {FIXTURE_EDGE: "완전 고정면 가장자리", CONCAVE_EDGE: "오목 모서리(필렛 없음)"}
MIN_BEND = np.cos(np.radians(15.0))  # faces meeting at less than 15° are smooth, not a corner
ZONE_FRACTION = 0.03  # exclusion radius as a fraction of the part's bounding diagonal


@dataclass
class Singularities:
    kind: np.ndarray  # (N,) NONE / FIXTURE_EDGE / CONCAVE_EDGE per node
    radius: dict[int, float]  # part id -> exclusion radius (mm)

    def zone(self, mesh: FEMesh, part_id: int) -> np.ndarray:
        """Boolean mask over all nodes: within the exclusion radius of a singular node of
        this part (False everywhere when the part has none)."""
        nodes = mesh.part_nodes(part_id)
        singular = nodes[self.kind[nodes] != NONE]
        mask = np.zeros(mesh.n_nodes, dtype=bool)
        if len(singular) == 0:
            return mask
        tree = cKDTree(mesh.nodes[singular])
        d, _ = tree.query(mesh.nodes[nodes])
        mask[nodes] = d <= self.radius[part_id]
        return mask

    def nearest_kind(self, mesh: FEMesh, part_id: int, node: int) -> int:
        """Kind of the singular node of the part closest to `node` (NONE if none)."""
        nodes = mesh.part_nodes(part_id)
        singular = nodes[self.kind[nodes] != NONE]
        if len(singular) == 0:
            return NONE
        d = np.linalg.norm(mesh.nodes[singular] - mesh.nodes[node], axis=1)
        return int(self.kind[singular[int(np.argmin(d))]])


def find_singularities(project: Project, mesh: FEMesh) -> Singularities:
    kind = np.zeros(mesh.n_nodes, dtype=int)
    _mark_concave_edges(mesh, kind)
    _mark_fixture_edges(project, mesh, kind)
    radius = {}
    for part in project.parts:
        nodes = mesh.part_nodes(part.id)
        if len(nodes) == 0:
            continue
        diag = float(np.linalg.norm(np.ptp(mesh.nodes[nodes], axis=0)))
        tets = mesh.tets[mesh.tet_part == part.id]
        h = float(np.linalg.norm(mesh.nodes[tets[:, 1]] - mesh.nodes[tets[:, 0]], axis=1).mean())
        # at least one element away from the singular node, else a fixed fraction of the
        # part size so the hot-spot value does not move with the mesh
        radius[part.id] = max(1.5 * h, ZONE_FRACTION * diag)
    return Singularities(kind, radius)


def _mark_concave_edges(mesh: FEMesh, kind: np.ndarray) -> None:
    tris = mesh.tris
    corners = tris[:, :3]
    _, normals, _ = tet10.tri6_load_weights(mesh.nodes[tris])
    centers = mesh.nodes[corners].mean(axis=1)
    # every corner edge of every boundary triangle, keyed so both orientations match
    a = np.concatenate([corners[:, 0], corners[:, 1], corners[:, 2]])
    b = np.concatenate([corners[:, 1], corners[:, 2], corners[:, 0]])
    mids = np.concatenate([tris[:, 5], tris[:, 3], tris[:, 4]])  # (0,1) (1,2) (2,0)
    tri_of = np.tile(np.arange(len(tris)), 3)
    # The surface normal of each triangle *at the shared edge*, from its quadratic shape.
    # Flat facet normals differ by the angle an element spans on any curved face, which
    # marked the seam of a split cylinder and the tangent line of a fillet as sharp
    # corners (and so excused a real stress concentration as a singularity).
    edge_normal = np.concatenate(
        [
            _tri6_normal(mesh.nodes[tris], L)
            for L in ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5))
        ]
    )
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((hi, lo))
    lo, hi, mids, tri_of = lo[order], hi[order], mids[order], tri_of[order]
    edge_normal = edge_normal[order]
    same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
    i = np.flatnonzero(same)  # pairs (i, i+1) share an edge
    t1, t2 = tri_of[i], tri_of[i + 1]
    sharp = (mesh.tri_face[t1] != mesh.tri_face[t2]) | (mesh.tri_part[t1] != mesh.tri_part[t2])
    bend = np.einsum("ij,ij->i", edge_normal[i], edge_normal[i + 1]) < MIN_BEND
    # the neighbour's centre lies outside the first triangle's plane -> the solid bends
    # inward (re-entrant); for a convex edge it lies behind the plane
    edge_len = np.linalg.norm(mesh.nodes[hi[i]] - mesh.nodes[lo[i]], axis=1)
    concave = np.einsum("ij,ij->i", centers[t2] - centers[t1], normals[t1]) > 1e-3 * edge_len
    concave &= np.einsum("ij,ij->i", centers[t1] - centers[t2], normals[t2]) > 1e-3 * edge_len
    pick = sharp & bend & concave & (mesh.tri_part[t1] == mesh.tri_part[t2])
    for nodes in (lo[i][pick], hi[i][pick], mids[i][pick]):
        kind[nodes] = CONCAVE_EDGE


def _tri6_normal(X: np.ndarray, L: tuple[float, float, float]) -> np.ndarray:
    """Unit normals of quadratic triangles X (S, 6, 3) at barycentric point L (same side
    as the flat normal (B - A) x (C - A)). Midside nodes 3..5 sit on edges (1,2) (0,2) (0,1)."""
    l0, l1, l2 = L
    d1 = np.array([-(4 * l0 - 1), 4 * l1 - 1, 0.0, 4 * l2, -4 * l2, 4 * (l0 - l1)])
    d2 = np.array([-(4 * l0 - 1), 0.0, 4 * l2 - 1, 4 * l1, 4 * (l0 - l2), -4 * l1])
    n = np.cross(np.einsum("k,ski->si", d1, X), np.einsum("k,ski->si", d2, X))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def _mark_fixture_edges(project: Project, mesh: FEMesh, kind: np.ndarray) -> None:
    from kstruct.model import FixtureKind

    refs = [r for fx in project.fixtures if fx.kind == FixtureKind.FIXED for r in fx.faces]
    if not refs:
        return
    on_fixture = np.zeros(len(mesh.tris), dtype=bool)
    on_fixture[mesh.face_tris(refs)] = True
    fixed_nodes = np.unique(mesh.tris[on_fixture])
    free_nodes = np.unique(mesh.tris[~on_fixture])
    ring = np.intersect1d(fixed_nodes, free_nodes)
    kind[ring] = np.where(kind[ring] == NONE, FIXTURE_EDGE, kind[ring])
