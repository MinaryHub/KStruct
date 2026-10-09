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
