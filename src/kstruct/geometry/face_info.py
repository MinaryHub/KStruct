"""Plain-language description of a B-rep face (shown when the user clicks a face)."""

from __future__ import annotations

from dataclasses import dataclass

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Face

_KIND_NAMES = {
    GeomAbs_Plane: "평면",
    GeomAbs_Cylinder: "원통면",
    GeomAbs_Cone: "원뿔면",
    GeomAbs_Sphere: "구면",
}


@dataclass(frozen=True)
class FaceInfo:
    kind: str
    area: float  # mm²
    radius: float | None = None  # mm, for cylinders (e.g. bolt holes)

    def describe(self) -> str:
        text = f"{self.kind}, 면적 {self.area:,.1f} mm²"
        if self.radius is not None:
            text += f", 지름 {self.radius * 2:,.2f} mm"
        return text


def face_info(face: TopoDS_Face) -> FaceInfo:
    surface = BRepAdaptor_Surface(face)
    kind = surface.GetType()
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    radius = surface.Cylinder().Radius() if kind == GeomAbs_Cylinder else None
    return FaceInfo(_KIND_NAMES.get(kind, "곡면"), props.Mass(), radius)


def face_at(shape, point: tuple[float, float, float]) -> int:
    """Index (faces_of order) of the face closest to a point; used by scripts and tests."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    from kstruct.geometry.tessellate import faces_of

    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
    distances = [BRepExtrema_DistShapeShape(vertex, f).Value() for f in faces_of(shape)]
    return min(range(len(distances)), key=distances.__getitem__)


def outward_normal(face: TopoDS_Face) -> tuple[float, float, float] | None:
    """Outward unit normal of a planar face, None for curved faces."""
    from OCP.TopAbs import TopAbs_REVERSED

    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Plane:
        return None
    x, y, z = surface.Plane().Axis().Direction().Coord()
    sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    return (sign * x, sign * y, sign * z)


def common_axis(faces: list[TopoDS_Face], tol: float = 1e-3):
    """(point, unit direction) of the axis shared by cylindrical faces, or None when a face
    is not a cylinder or the axes differ (used for torque, bearing and centrifugal loads)."""
    import numpy as np

    axis = None
    for face in faces:
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Cylinder:
            return None
        ax = surface.Cylinder().Axis()
        point = np.array(ax.Location().Coord())
        direction = np.array(ax.Direction().Coord())
        direction /= np.linalg.norm(direction)
        if axis is None:
            axis = (point, direction)
            continue
        p0, d0 = axis
        if abs(abs(d0 @ direction) - 1) > 1e-6:
            return None
        offset = point - p0
        if np.linalg.norm(offset - (offset @ d0) * d0) > tol * max(1.0, np.linalg.norm(offset)):
            return None
    return axis


def faces_centroid(faces: list[TopoDS_Face]):
    """Area-weighted centroid of faces (numpy array)."""
    import numpy as np

    total, acc = 0.0, np.zeros(3)
    for face in faces:
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        acc += area * np.array(props.CentreOfMass().Coord())
        total += area
    return acc / total if total > 0 else acc
