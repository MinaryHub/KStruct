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

from kstruct.i18n import tr
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
        raise GeometryImportError(tr("지원하지 않는 파일 형식입니다: {suffix}", suffix=suffix))

    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise GeometryImportError(tr("파일을 읽을 수 없습니다: {name}", name=path.name))
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise GeometryImportError(tr("파일에 형상이 없습니다: {name}", name=path.name))
    return shape


def _decode_name(text: str) -> str:
    """STEP stores names as bytes; UTF-8 names come back as one code point per byte."""
    try:
        if text and all(ord(c) < 256 for c in text):
            return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    return text


_PLACEHOLDER_NAMES = ("open cascade step translator", "unnamed", "none", "nauo", "=>[")


def _label_name(label) -> str:
    from OCP.TDataStd import TDataStd_Name

    name = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name):
        text = _decode_name(name.Get().ToExtString()).strip()
        # exporters that were given no name write their own: not a part name
        if text.lower().startswith(_PLACEHOLDER_NAMES):
            return ""
        return text
    return ""


def read_named_solids(path: Path) -> list[tuple[str, TopoDS_Shape]]:
    """Solids of a STEP file with the names CAD systems write (XCAF), assemblies
    flattened with their placements. Empty when the file carries no product structure."""
    from OCP.collections import Sequence_TDF_Label
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(doc):
        return []
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    out: list[tuple[str, TopoDS_Shape]] = []

    def visit(label, location: TopLoc_Location, inherited: str) -> None:
        name = _label_name(label) or inherited
        if tool.IsReference_s(label):
            referred = tool.GetReferredShape_s(label)
            visit(referred, location.Multiplied(tool.GetLocation_s(label)), name)
            return
        if tool.IsAssembly_s(label):
            components = Sequence_TDF_Label()
            tool.GetComponents_s(label, components)
            for i in range(1, components.Length() + 1):
                visit(components.Value(i), location, "")
            return
        shape = tool.GetShape_s(label).Moved(location)
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solids = []
        while explorer.More():
            solids.append(explorer.Current())
            explorer.Next()
        for k, solid in enumerate(solids):
            out.append((name if len(solids) == 1 else f"{name} - {k + 1}", solid))

    roots = Sequence_TDF_Label()
    tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        visit(roots.Value(i), TopLoc_Location(), "")
    return out


def _unique(names: list[str], stem: str) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        base = name or (stem if len(names) == 1 else f"{stem} - {i + 1}")
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return out


def load_cad(path: str | Path) -> Project:
    """Read a CAD (STEP/IGES) or mesh (STL/OBJ/glTF/3MF/FBX) file into a new Project (mm)."""
    path = Path(path)
    from kstruct.geometry.mesh_import import MESH_SUFFIXES, load_mesh

    if path.suffix.lower() in MESH_SUFFIXES:
        return load_mesh(path)

    named: list[tuple[str, TopoDS_Shape]] = []
    if path.suffix.lower() in STEP_SUFFIXES:
        try:
            named = read_named_solids(path)
        except Exception:  # fall back to the plain reader on any XCAF trouble
            named = []
    if named:
        solids = [shape for _, shape in named]
        names = _unique([name for name, _ in named], path.stem)
    else:
        shape = _read_shape(path)
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids.append(explorer.Current())
            explorer.Next()
        names = _unique([""] * len(solids), path.stem)
    if not solids:
        raise GeometryImportError(
            tr("솔리드(속이 찬 형상)가 없습니다. 곡면만 있는 모델은 해석할 수 없습니다.")
        )

    project = Project(source_path=path)
    for i, (name, solid) in enumerate(zip(names, solids, strict=True)):
        project.parts.append(Part(id=i, name=name, shape=solid))
    return project
