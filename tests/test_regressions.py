"""Regression tests for defects found in the 2026-11 review (each one reproduced first)."""

import zipfile

import numpy as np
import pytest

from kstruct import standards
from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.thermal import solve_thermal
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box, _hole, l_bracket, two_plates, write_step
from kstruct.model import (
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Part,
    Project,
)
from kstruct.project_io import ProjectFileError, load_project, save_project
from kstruct.report.summary import summarize
from kstruct.standards import Verdict

MATERIALS = standards.materials()
STEEL = MATERIALS["s235"]


def _plate_with_hole(split: bool):
    """40x100x5 plate with a 12 mm hole; `split` cuts the hole wall into two half
    cylinders, which is how many CAD systems export a closed face."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed

    shape = BRepAlgoAPI_Cut(_box(0, 0, 0, 40, 100, 5), _hole(20, 50, -1, gp_Dir(0, 0, 1), 6, 7))
    shape = shape.Shape()
    if split:
        divide = ShapeUpgrade_ShapeDivideClosed(shape)
        divide.SetNbSplitPoints(1)
        divide.Perform()
        shape = divide.Result()
    return shape


@pytest.mark.parametrize("split", [False, True])
def test_seam_of_a_curved_face_is_not_a_sharp_corner(split):
    """The stress concentration at a hole is real. With the hole wall in two faces the
    seam used to be marked as a sharp concave corner, the peak was excused as a
    singularity and the verdict flipped from "caution" to "safe"."""
    from kstruct.fem.singularity import CONCAVE_EDGE, find_singularities

    shape = _plate_with_hole(split)
    project = Project(parts=[Part(0, "plate", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (20, 0, 2.5)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (20, 100, 2.5)))], vector=(0, 12000.0, 0))
    ]
    mesh = generate_mesh(project, "coarse")
    assert not np.any(find_singularities(project, mesh).kind == CONCAVE_EDGE)
    result = solve_static(project, mesh, {0: STEEL})
    summary = summarize(project, result, standards.effective_pack(project))
    assert summary.verdict == Verdict.WARN
    assert summary.parts[0].singular_kind is None
    assert summary.parts[0].min_fos == pytest.approx(1.3, abs=0.1)


def test_sharp_corner_is_still_found_and_fatigue_does_not_follow_the_mesh():
    """The sample bracket has a true re-entrant corner. Static and fatigue verdicts are
    taken outside it, so neither changes with the mesh grade (fatigue used to go
    1.50 -> 1.36 -> 1.19 and from "safe" to "caution")."""
    from kstruct.fem.singularity import CONCAVE_EDGE, find_singularities

    shape = l_bracket()
    project = Project(parts=[Part(0, "bracket", shape, "s235")], load_cycle="pulsating")
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -2000.0))
    ]
    fatigue = []
    for quality in ("coarse", "normal"):
        mesh = generate_mesh(project, quality)
        assert np.any(find_singularities(project, mesh).kind == CONCAVE_EDGE)
        summary = summarize(
            project, solve_static(project, mesh, {0: STEEL}), standards.effective_pack(project)
        )
        assert summary.verdict == Verdict.OK
        assert summary.parts[0].fatigue_excludes_singularity
        assert any("피로 안전율과 수명은 응력 특이점" in n for n in summary.notes)
        fatigue.append(summary.parts[0].fatigue_fos)
    assert fatigue[1] == pytest.approx(fatigue[0], rel=0.08)


def test_brittle_part_can_fail_in_compression():
    """Cast iron crushed at 1000 MPa was reported "safe, 100x margin": only the tensile
    principal stress was checked."""
    shape = _box(0, 0, 0, 10, 10, 50)
    project = Project(parts=[Part(0, "bar", shape, "fc250")])

    def face(point):
        return FaceRef(0, face_at(shape, point))

    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [face((0, 5, 25))]),
        Fixture(FixtureKind.ROLLER, [face((5, 0, 25))]),
        Fixture(FixtureKind.ROLLER, [face((5, 5, 0))]),
    ]
    project.loads = [Load(LoadKind.FORCE, [face((5, 5, 50))], vector=(0, 0, -100_000.0))]
    cast_iron = MATERIALS["fc250"]
    assert cast_iron.compressive_strength == 840
    mesh = generate_mesh(project, "coarse")
    result = solve_static(project, mesh, {0: cast_iron})
    assert result.safety_factor.min() == pytest.approx(0.84, rel=1e-3)
    summary = summarize(project, result, standards.effective_pack(project))
    assert summary.verdict == Verdict.FAIL
    assert summary.parts[0].crushing and summary.parts[0].capacity == 840
    assert any("압축 응력" in n for n in summary.notes)
    # rollers do not clamp: no "fully fixed boundary" remark
    assert not any("완전 고정" in n for n in summary.notes)
    # without a compressive strength the tensile strength is used (safe side)
    from dataclasses import replace

    unknown = replace(cast_iron, compressive_strength=None)
    assert solve_static(project, mesh, {0: unknown}).safety_factor.min() == pytest.approx(
        0.25, rel=1e-3
    )


def _bar(length=100.0):
    shape = _box(0, 0, 0, length, 10, 10)
    return Project(parts=[Part(0, "bar", shape, "s235")]), shape


def test_radiation_converges_beyond_a_few_watts():
    """5 W and 20 W into a radiating bar did not converge (secant coefficient, plain
    fixed point); the tangent linearisation does, and the heat balance closes."""
    project, shape = _bar()
    for watts in (5.0, 20.0):
        project.loads = [
            Load(LoadKind.HEAT_FLOW, [FaceRef(0, face_at(shape, (0, 5, 5)))], magnitude=watts),
            Load(
                LoadKind.RADIATION,
                [FaceRef(0, face_at(shape, (50, 5, 10)))],
                magnitude=0.9,
                ambient=20.0,
            ),
        ]
        mesh = generate_mesh(project, "coarse")
        r = solve_thermal(project, mesh, {0: STEEL})
        assert r.iterations < 20
        assert r.heat_in == pytest.approx(watts, rel=1e-6)
        assert r.heat_out == pytest.approx(watts, rel=1e-3)


def test_part_without_a_temperature_reference_is_reported(tmp_path):
    """An unconnected second part silently came out at 0 °C."""
    from kstruct.geometry.step_import import load_cad

    project = load_cad(write_step(two_plates(), tmp_path / "plates.step"))
    for part in project.parts:
        part.material_id = "s235"
    lower = project.parts[0].shape
    project.loads = [
        Load(
            LoadKind.FACE_TEMPERATURE, [FaceRef(0, face_at(lower, (0, 20, 2.5)))], magnitude=100.0
        ),
        Load(LoadKind.CONVECTION, [FaceRef(0, face_at(lower, (50, 20, 0)))], magnitude=10.0),
    ]
    mesh = generate_mesh(project, "coarse")
    with pytest.raises(SolverError, match="온도의 기준이 없습니다"):
        solve_thermal(project, mesh, {0: STEEL, 1: STEEL})
    from kstruct.geometry.contacts import find_touching_faces

    project.connections = find_touching_faces(project)
    r = solve_thermal(project, mesh, {0: STEEL, 1: STEEL})
    assert r.temperature[mesh.part_nodes(1)].min() > 50.0


def test_failed_save_keeps_the_previous_file(tmp_path, monkeypatch):
    """Saving wrote into the target in place, so a failure half-way destroyed it (and
    the autosave uses the same function)."""
    import kstruct.project_io as io

    project, _ = _bar()
    path = save_project(project, tmp_path / "bar.kst")
    before = path.read_bytes()
    monkeypatch.setattr(io.BRepTools, "Write_s", lambda *a: False)
    with pytest.raises(ProjectFileError):
        save_project(project, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]  # no stray partial file
    assert len(load_project(path).parts) == 1


def test_damaged_project_file_is_a_project_file_error(tmp_path):
    project, _ = _bar()
    path = save_project(project, tmp_path / "bar.kst")
    data = bytearray(path.read_bytes())
    data[60:68] = b"\xff" * 8  # inside the compressed project.json
    (tmp_path / "bad.kst").write_bytes(bytes(data))
    with pytest.raises(ProjectFileError):
        load_project(tmp_path / "bad.kst")
    # valid zip, wrong content types
    with zipfile.ZipFile(tmp_path / "odd.kst", "w") as z:
        z.writestr("project.json", '{"format": "kstruct", "version": 2, "cases": null}')
    with pytest.raises(ProjectFileError):
        load_project(tmp_path / "odd.kst")


def test_report_covers_only_the_analysed_parts(tmp_path):
    """A hidden part's fixture crashed the report; its load was listed as applied."""
    from kstruct.geometry.step_import import load_cad
    from kstruct.model import active_project
    from kstruct.report.pdf import build_html

    project = load_cad(write_step(two_plates(), tmp_path / "plates.step"))
    for part in project.parts:
        part.material_id = "s235"
    lower, upper = project.parts[0].shape, project.parts[1].shape
    project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (0, 20, 2.5)))]),
        Fixture(FixtureKind.FIXED, [FaceRef(1, face_at(upper, (20, 20, 7.5)))]),
    ]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(lower, (100, 20, 2.5)))], vector=(0, 0, -100.0)),
        Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (80, 20, 7.5)))], vector=(0, 0, -5000.0)),
    ]
    project.parts[1].visible = False
    active = active_project(project)
    mesh = generate_mesh(active, "coarse")
    result = solve_static(active, mesh, {0: STEEL})
    summary = summarize(active, result, standards.effective_pack(project))
    text = build_html(project, result, summary, [], {0: STEEL})
    assert "100.0 N" in text and "5,000.0 N" not in text


def test_units_are_never_shown_in_exponent_notation():
    from kstruct.report.describe import describe_load
    from kstruct.units import fmt

    assert fmt("US", "stress", 100.0) == "14,504 psi"
    assert fmt("SI", "torque", 25_000_000.0) == "25,000 N·m"
    assert fmt("SI", "length", 0.4054) == "0.4054 mm"
    project, shape = _bar()
    face = [FaceRef(0, face_at(shape, (100, 5, 5)))]
    oblique = Load(LoadKind.TORQUE, face, vector=(0.9, 0.436, 0.0), magnitude=1000.0)
    assert "X 축" not in describe_load(project, oblique)
    along = Load(LoadKind.TORQUE, face, vector=(0.0, 0.0, -1.0), magnitude=1000.0)
    assert "−Z 축" in describe_load(project, along)


def test_notes_follow_the_unit_system():
    shape = l_bracket()
    project = Project(parts=[Part(0, "bracket", shape, "s235")], units="US")
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -2000.0))
    ]
    mesh = generate_mesh(project, "coarse")
    summary = summarize(
        project, solve_static(project, mesh, {0: STEEL}), standards.effective_pack(project)
    )
    note = next(n for n in summary.notes if "응력 특이점" in n)
    assert "psi" in note and "MPa" not in note and " in " in note


def test_bearing_load_must_be_radial():
    """The component along the hole axis was dropped while the condition list kept the
    full vector."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir

    shape = BRepAlgoAPI_Cut(_box(0, 0, 0, 40, 40, 10), _hole(20, 20, -1, gp_Dir(0, 0, 1), 6, 12))
    shape = shape.Shape()
    project = Project(parts=[Part(0, "lug", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 20, 5)))])]
    hole = [FaceRef(0, face_at(shape, (26, 20, 5)))]
    project.loads = [Load(LoadKind.BEARING, hole, vector=(0.0, 1000.0, 1000.0))]
    mesh = generate_mesh(project, "coarse")
    with pytest.raises(SolverError, match="구멍 축과 수직이 아닙니다"):
        solve_static(project, mesh, {0: STEEL})
    project.loads = [Load(LoadKind.BEARING, hole, vector=(0.0, 1000.0, 0.0))]
    result = solve_static(project, mesh, {0: STEEL})
    assert result.applied_load == pytest.approx([0.0, 1000.0, 0.0], abs=1e-6)


def test_touching_faces_within_the_tolerance_are_found():
    """Faces a micrometre apart passed every check but the exact boolean overlap."""
    from kstruct.geometry.contacts import find_touching_faces

    for gap in (0.0, 1e-6, 1e-3):
        project = Project(
            parts=[
                Part(0, "lower", _box(0, 0, 0, 100, 40, 5)),
                Part(1, "upper", _box(20, 0, 5 + gap, 60, 40, 5)),
            ]
        )
        assert len(find_touching_faces(project)) == 1, gap
    apart = Project(
        parts=[
            Part(0, "lower", _box(0, 0, 0, 100, 40, 5)),
            Part(1, "upper", _box(20, 0, 5.5, 60, 40, 5)),
        ]
    )
    assert find_touching_faces(apart) == []


def test_exporter_placeholder_is_not_a_part_name(tmp_path):
    """A STEP file written without names carries the exporter's own product name; the
    sample bracket showed up as "Open CASCADE STEP translator 8.0 1"."""
    from kstruct.geometry.step_import import load_cad

    project = load_cad(write_step(l_bracket(), tmp_path / "bracket.step"))
    assert [p.name for p in project.parts] == ["bracket"]


def test_3mf_units(tmp_path):
    """Inch, metre, centimetre and foot files were read as millimetres (the unit names
    were compared in the plural)."""
    from kstruct.geometry.mesh_import import load_mesh

    def write(unit: str):
        corners = [(x, y, z) for x in (0, 10) for y in (0, 20) for z in (0, 30)]
        faces = [
            (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ]  # fmt: skip
        v = "".join(f'<vertex x="{a}" y="{b}" z="{c}"/>' for a, b, c in corners)
        t = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
        ns = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        model = (
            f'<?xml version="1.0" encoding="UTF-8"?><model unit="{unit}" xmlns="{ns}">'
            f'<resources><object id="1" type="model"><mesh><vertices>{v}</vertices>'
            f"<triangles>{t}</triangles></mesh></object></resources>"
            '<build><item objectid="1"/></build></model>'
        )
        types = "http://schemas.openxmlformats.org/package/2006/content-types"
        rels = "http://schemas.openxmlformats.org/package/2006/relationships"
        path = tmp_path / f"{unit}.3mf"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(
                "[Content_Types].xml",
                f'<?xml version="1.0"?><Types xmlns="{types}"><Default Extension="rels" '
                'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="model" ContentType='
                '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>',
            )
            z.writestr(
                "_rels/.rels",
                f'<?xml version="1.0"?><Relationships xmlns="{rels}"><Relationship '
                'Target="/3D/3dmodel.model" Id="rel0" Type='
                '"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
            )
            z.writestr("3D/3dmodel.model", model)
        return path

    from kstruct.geometry.tessellate import bounding_diagonal

    base = np.linalg.norm([10, 20, 30])
    for unit, factor in (("millimeter", 1), ("inch", 25.4), ("meter", 1000), ("foot", 304.8)):
        project = load_mesh(write(unit))
        assert bounding_diagonal(project.parts[0].shape) == pytest.approx(base * factor, rel=1e-6)
