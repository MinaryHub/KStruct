"""Convert B-rep shapes to display meshes that remember which B-rep face each
triangle came from (cell_data["face_id"]), so clicks in the viewer map back to faces.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeIndexMap
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape


def faces_of(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    """Faces in a stable order; the list index is the face_index used by FaceRef."""
    face_map = ShapeIndexMap()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    return [TopoDS.Face(face_map.FindKey(i)) for i in range(1, face_map.Extent() + 1)]


def bounding_diagonal(shape: TopoDS_Shape) -> float:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box.CornerMin().Distance(box.CornerMax())


def tessellate(shape: TopoDS_Shape, quality: float = 1e-3) -> pv.PolyData:
    """Triangulate a shape. `quality` is the chord deflection relative to the model size."""
    deflection = max(bounding_diagonal(shape) * quality, 1e-6)
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.3, True)

    points: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    face_ids: list[np.ndarray] = []
    offset = 0
    for face_index, face in enumerate(faces_of(shape)):
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        if tri is None:
            continue
        trsf = location.Transformation()
        pts = np.array([tri.Node(i).Transformed(trsf).Coord() for i in range(1, tri.NbNodes() + 1)])
        tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
        if face.Orientation() == TopAbs_REVERSED:
            tris = tris[:, [0, 2, 1]]
        points.append(pts)
        triangles.append(tris + offset)
        face_ids.append(np.full(len(tris), face_index))
        offset += len(pts)

    if not points:
        return pv.PolyData()
    tris = np.vstack(triangles)
    cells = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    mesh = pv.PolyData(np.vstack(points), cells)
    mesh.cell_data["face_id"] = np.concatenate(face_ids)
    return mesh
