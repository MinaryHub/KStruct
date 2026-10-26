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


def test_radiation_load_and_conductivity_table_survive_the_file(tmp_path):
    from dataclasses import replace

    from kstruct import standards
    from kstruct.project_io import load_project, save_project

    project = Project(parts=[Part(0, "p", l_bracket())])
    project.loads = [
        Load(LoadKind.RADIATION, [FaceRef(0, 0)], magnitude=0.85, ambient=30.0),
    ]
    steel = standards.materials()["s235"]
    custom = replace(
        steel, id="user_k", name="온도 의존 강", conductivity_table=((0.0, 50.0), (400.0, 40.0))
    )
    project.custom_materials = {custom.id: custom}
    project.parts[0].material_id = custom.id
    path = save_project(project, tmp_path / "r.kst")
    loaded = load_project(path)
    ld = loaded.loads[0]
    assert ld.kind == LoadKind.RADIATION and ld.magnitude == 0.85 and ld.ambient == 30.0
    table = loaded.custom_materials["user_k"].conductivity_table
    assert table == ((0.0, 50.0), (400.0, 40.0))
    assert loaded.custom_materials["user_k"].thermal_conductivity == steel.thermal_conductivity
