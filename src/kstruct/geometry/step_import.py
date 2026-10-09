"""STEP / IGES import via OpenCASCADE.

Every solid found in the file becomes a separate Part, so multi-body files and
assemblies exported as STEP load as several parts.
"""

from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape

from kstruct.model import Part, Project

STEP_SUFFIXES = {".step", ".stp"}
IGES_SUFFIXES = {".iges", ".igs"}


class GeometryImportError(Exception):
    pass


def _read_shape(path: Path) -> TopoDS_Shape:
    suffix = path.suffix.lower()
    if suffix in STEP_SUFFIXES:
        reader = STEPControl_Reader()
    elif suffix in IGES_SUFFIXES:
        reader = IGESControl_Reader()
    else:
        raise GeometryImportError(f"지원하지 않는 파일 형식입니다: {suffix}")

    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise GeometryImportError(f"파일을 읽을 수 없습니다: {path.name}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise GeometryImportError(f"파일에 형상이 없습니다: {path.name}")
    return shape


def load_cad(path: str | Path) -> Project:
    """Read a STEP/IGES file into a new Project (lengths in mm)."""
    path = Path(path)
    shape = _read_shape(path)

    solids: list[TopoDS_Shape] = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if not solids:
        raise GeometryImportError(
            "솔리드(속이 찬 형상)가 없습니다. 곡면만 있는 모델은 해석할 수 없습니다."
        )

    project = Project(source_path=path)
    for i, solid in enumerate(solids):
        name = path.stem if len(solids) == 1 else f"{path.stem} - {i + 1}"
        project.parts.append(Part(id=i, name=name, shape=solid))
    return project
