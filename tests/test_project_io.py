from kstruct.geometry.samples import l_bracket
from kstruct.geometry.tessellate import faces_of
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Part, Project
from kstruct.project_io import load_project, save_project


def test_round_trip(tmp_path):
    project = Project(parts=[Part(0, "브래킷", l_bracket(), "s235")], standard="jis")
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, 3)])]
    project.loads = [Load(LoadKind.FORCE, [FaceRef(0, 5)], vector=(0, 0, -100.0))]
    loaded = load_project(save_project(project, tmp_path / "a.kst"))

    assert loaded.standard == "jis"
    assert loaded.parts[0].name == "브래킷"
    assert loaded.parts[0].material_id == "s235"
    assert len(faces_of(loaded.parts[0].shape)) == len(faces_of(project.parts[0].shape))
    assert loaded.fixtures == project.fixtures
    assert loaded.loads == project.loads


def test_mesh_is_stored_and_reused_only_while_valid(tmp_path):
    import numpy as np

    from kstruct.fem import generate_mesh
    from kstruct.model import MeshRefinement
    from kstruct.project_io import load_project_mesh

    project = Project(parts=[Part(0, "b", l_bracket(), "s235")])
    mesh = generate_mesh(project, "coarse")
    path = save_project(project, tmp_path / "m.kst", mesh)
    loaded = load_project(path)
    again = load_project_mesh(path, loaded)
    assert again is not None
    assert np.array_equal(again.nodes, mesh.nodes) and np.array_equal(again.tets, mesh.tets)
    assert again.notes == mesh.notes
    # a different preset or refinement would give another mesh: the stored one is ignored
    loaded.mesh_quality = "fine"
    assert load_project_mesh(path, loaded) is None
    loaded.mesh_quality = "coarse"
    loaded.mesh_refinements = [MeshRefinement([FaceRef(0, 1)], 0.5)]
    assert load_project_mesh(path, loaded) is None
    # files without a mesh simply give None
    plain = save_project(project, tmp_path / "p.kst")
    assert load_project_mesh(plain, project) is None
