"""Built-in sample models (used by tests and the "예제 열기" menu)."""

from __future__ import annotations

from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape


def _box(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def _hole(x: float, y: float, z: float, direction: gp_Dir, r: float, h: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), direction), r, h).Shape()


def l_bracket() -> TopoDS_Shape:
    """100 x 60 L-bracket, 8 mm thick, two mounting holes in each flange (mm)."""
    base = _box(0, 0, 0, 100, 60, 8)
    wall = _box(0, 0, 0, 8, 60, 80)
    shape = BRepAlgoAPI_Fuse(base, wall).Shape()
    for y in (15, 45):
        shape = BRepAlgoAPI_Cut(shape, _hole(70, y, -1, gp_Dir(0, 0, 1), 5, 10)).Shape()
        shape = BRepAlgoAPI_Cut(shape, _hole(-1, y, 55, gp_Dir(1, 0, 0), 5, 10)).Shape()
    # merge coplanar faces split by the boolean ops so each flat side is one clickable face
    unify = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    unify.Build()
    return unify.Shape()


def two_plates() -> TopoDS_Shape:
    """Two stacked plates as separate solids (assembly test case)."""
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, _box(0, 0, 0, 100, 40, 5))
    builder.Add(compound, _box(20, 0, 5, 60, 40, 5))
    return compound


def write_step(shape: TopoDS_Shape, path: str | Path) -> Path:
    path = Path(path)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise OSError(f"STEP 저장 실패: {path}")
    return path


def bolted_plates() -> TopoDS_Shape:
    """Two 80x60x5 plates, one on the other, each with three 11 mm (M10 clearance) holes."""
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for z0 in (0.0, 5.0):
        plate = _box(0, 0, z0, 80, 60, 5)
        for x, y in ((10, 10), (70, 10), (40, 50)):
            plate = BRepAlgoAPI_Cut(plate, _hole(x, y, z0 - 1, gp_Dir(0, 0, 1), 5.5, 7)).Shape()
        builder.Add(compound, plate)
    return compound
