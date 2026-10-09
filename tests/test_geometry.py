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
