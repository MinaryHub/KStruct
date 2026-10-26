"""Mesh sizing: wall-thickness estimate, thickness-aware presets, per-face refinement."""

import numpy as np
import pytest

from kstruct.fem import generate_mesh
from kstruct.fem.mesh import MESH_PRESETS, face_sizes
from kstruct.geometry.face_info import face_at, face_info
from kstruct.geometry.samples import _box, l_bracket
from kstruct.geometry.tessellate import faces_of
from kstruct.geometry.thickness import face_thickness
from kstruct.model import FaceRef, MeshRefinement, Part, Project
from kstruct.project_io import load_project, save_project


def test_face_thickness_of_a_plate_and_the_bracket():
    plate = _box(0, 0, 0, 100, 60, 8)
    t = face_thickness(plate)
    assert len(t) == 6
    big = [i for i, f in enumerate(faces_of(plate)) if face_info(f).area == pytest.approx(6000)]
    assert all(t[i] == pytest.approx(8.0, rel=1e-3) for i in big)
    bracket = l_bracket()
    t = face_thickness(bracket)
    flats = [i for i, f in enumerate(faces_of(bracket)) if face_info(f).area > 4000]
    assert flats and all(t[i] == pytest.approx(8.0, rel=1e-3) for i in flats)


def _elements_through(mesh, part_id, z_lo, z_hi, x_band):
    """Distinct node z-levels in a thin band of a plate: layers + 1 corner levels at least."""
    nodes = mesh.part_nodes(part_id)
    inside = nodes[np.abs(mesh.nodes[nodes, 0] - x_band) < 3.0]
    z = np.round(mesh.nodes[inside, 2], 3)
    return len(np.unique(z[(z >= z_lo - 1e-6) & (z <= z_hi + 1e-6)]))


def test_presets_put_elements_through_thin_walls():
    plate = _box(0, 0, 0, 200, 60, 6)  # diagonal ≈ 209: global sizes 21 / 10 / 7 / 5 mm
    project = Project(parts=[Part(0, "plate", plate, "s235")])
    maxh, per_part, notes = face_sizes(project, "normal")
    assert maxh > 6 and max(per_part[0].values()) <= 3.0 + 1e-9
    assert notes and "두께 방향으로 2개" in notes[0]
    mesh = generate_mesh(project, "normal")
    assert "두께" in mesh.notes[0]
    # at least 2 elements (quadratic: 5 node levels) through the 6 mm plate
    assert _elements_through(mesh, 0, 0, 6, 100) >= 5
    coarse = generate_mesh(project, "coarse")
    assert len(coarse.tets) < len(mesh.tets)


def test_face_refinement_is_applied_and_saved(tmp_path):
    shape = l_bracket()
    hole = FaceRef(0, face_at(shape, (75, 15, 4)))
    project = Project(parts=[Part(0, "b", shape, "s235")])
    base = generate_mesh(project, "coarse")
    project.mesh_refinements = [MeshRefinement([hole], 0.8)]
    fine = generate_mesh(project, "coarse")
    assert len(fine.face_tris([hole])) > 5 * len(base.face_tris([hole]))
    assert any("사용자가 지정한 면 1개" in n for n in fine.notes)
    loaded = load_project(save_project(project, tmp_path / "m.kst"))
    assert loaded.mesh_refinements == project.mesh_refinements


def test_presets_are_ordered():
    order = ["coarse", "normal", "fine", "very_fine"]
    divisions = [MESH_PRESETS[k].divisions for k in order]
    layers = [MESH_PRESETS[k].layers for k in order]
    assert divisions == sorted(divisions) and layers == sorted(layers)
