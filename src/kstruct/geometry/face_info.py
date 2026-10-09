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
