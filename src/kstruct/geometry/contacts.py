"""Detect faces of different parts that touch (candidates for bonded connections)."""

from __future__ import annotations

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_REVERSED

from kstruct.geometry.face_info import face_info
from kstruct.geometry.tessellate import bounding_diagonal, faces_of
from kstruct.model import Connection, ConnectionKind, FaceRef, Project


def _bbox(shape, gap: float) -> Bnd_Box:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    box.Enlarge(gap)
    return box


def _coincident(fa, fb, tol: float) -> bool:
    sa, sb = BRepAdaptor_Surface(fa), BRepAdaptor_Surface(fb)
    ka, kb = sa.GetType(), sb.GetType()
    if ka != kb:
        return False
    if ka == GeomAbs_Plane:
        pa, pb = sa.Plane(), sb.Plane()
        na = np.array(pa.Axis().Direction().Coord())
        nb = np.array(pb.Axis().Direction().Coord())
        # outward normals of mating faces point at each other
        if fa.Orientation() == TopAbs_REVERSED:
            na = -na
        if fb.Orientation() == TopAbs_REVERSED:
            nb = -nb
        if na @ nb > -1 + 1e-6:
            return False
        offset = np.array(pb.Location().Coord()) - np.array(pa.Location().Coord())
        return abs(offset @ na) < tol
    if ka == GeomAbs_Cylinder:
        ca, cb = sa.Cylinder(), sb.Cylinder()
        if abs(ca.Radius() - cb.Radius()) > tol:
            return False
        da = np.array(ca.Axis().Direction().Coord())
        db = np.array(cb.Axis().Direction().Coord())
        if abs(abs(da @ db) - 1) > 1e-6:
            return False
        rel = np.array(cb.Location().Coord()) - np.array(ca.Location().Coord())
        return np.linalg.norm(rel - (rel @ da) * da) < tol
    return False


def _overlap_area(fa, fb) -> float:
    common = BRepAlgoAPI_Common(fa, fb)
    if not common.IsDone():
        return 0.0
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(common.Shape(), props)
    return props.Mass()


def find_touching_faces(project: Project, tol: float | None = None) -> list[Connection]:
    """Bonded connections for every pair of coincident planar or cylindrical faces of
    different parts. The smaller face is tied onto the larger one."""
    if len(project.parts) < 2:
        return []
    diag = max(bounding_diagonal(p.shape) for p in project.parts)
    tol = tol if tol is not None else 1e-4 * diag
    faces = {p.id: faces_of(p.shape) for p in project.parts}
    boxes = {p.id: [_bbox(f, tol) for f in faces[p.id]] for p in project.parts}

    connections = []
    parts = project.parts
    for i, pa in enumerate(parts):
        box_a = _bbox(pa.shape, tol)
        for pb in parts[i + 1 :]:
            if box_a.IsOut(_bbox(pb.shape, tol)):
                continue
            for ia, fa in enumerate(faces[pa.id]):
                for ib, fb in enumerate(faces[pb.id]):
                    if boxes[pa.id][ia].IsOut(boxes[pb.id][ib]):
                        continue
                    if not _coincident(fa, fb, tol):
                        continue
                    dist = BRepExtrema_DistShapeShape(fa, fb)
                    if not dist.IsDone() or dist.Value() > tol:
                        continue
                    area_a, area_b = face_info(fa).area, face_info(fb).area
                    if _overlap_area(fa, fb) < 1e-3 * min(area_a, area_b):
                        continue  # only an edge or corner touches
                    a, b = FaceRef(pa.id, ia), FaceRef(pb.id, ib)
                    if area_a < area_b:
                        a, b = b, a
                    connections.append(Connection(ConnectionKind.BONDED, [a], [b]))
    return connections
