import numpy as np
import pytest

from kstruct.geometry import GeometryImportError, faces_of, load_cad, tessellate
from kstruct.geometry.samples import l_bracket, two_plates, write_step


def test_box_round_trip(tmp_path):
    from kstruct.geometry.samples import _box

    project = load_cad(write_step(_box(0, 0, 0, 10, 20, 30), tmp_path / "box.step"))
    assert len(project.parts) == 1
    shape = project.parts[0].shape
    assert len(faces_of(shape)) == 6

    mesh = tessellate(shape)
    assert set(np.unique(mesh.cell_data["face_id"])) == set(range(6))
    assert mesh.bounds_size == pytest.approx((10, 20, 30))
    # closed solid: triangulated surface encloses the box volume
    assert abs(mesh.volume) == pytest.approx(6000, rel=1e-6)


def test_bracket_has_holes(tmp_path):
    project = load_cad(write_step(l_bracket(), tmp_path / "bracket.stp"))
    assert len(project.parts) == 1
    # 4 holes add one cylindrical face each to the 8 faces of the fused L
    assert len(faces_of(project.parts[0].shape)) == 12


def test_multiple_solids_become_parts(tmp_path):
    project = load_cad(write_step(two_plates(), tmp_path / "plates.step"))
    assert [p.id for p in project.parts] == [0, 1]
    assert project.parts[0].name == "plates - 1"


def test_unsupported_format(tmp_path):
    bad = tmp_path / "model.sldprt"
    bad.write_bytes(b"")
    with pytest.raises(GeometryImportError):
        load_cad(bad)


def test_touching_faces_are_detected(tmp_path):
    from kstruct.geometry.contacts import find_touching_faces

    project = load_cad(write_step(two_plates(), tmp_path / "plates.step"))
    connections = find_touching_faces(project)
    assert len(connections) == 1
    c = connections[0]
    # the upper plate's bottom (smaller) is tied onto the lower plate's top
    assert c.faces_a[0].part_id == 0 and c.faces_b[0].part_id == 1


def _write_named_step(path, named_shapes):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for name, shape in named_shapes:
        label = tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def test_step_part_names_are_read(tmp_path):
    from kstruct.geometry import load_cad
    from kstruct.geometry.samples import _box

    path = _write_named_step(
        tmp_path / "named.step",
        [
            ("베이스 플레이트", _box(0, 0, 0, 50, 50, 5)),
            ("Upper-Block", _box(0, 0, 5, 20, 20, 20)),
            ("Upper-Block", _box(30, 0, 5, 20, 20, 20)),  # same name twice: numbered
        ],
    )
    project = load_cad(path)
    assert [p.name for p in project.parts] == ["베이스 플레이트", "Upper-Block", "Upper-Block (2)"]
    assert len(project.parts) == 3
