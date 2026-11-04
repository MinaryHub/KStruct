"""Load cases: model, file format, per-case solve (superposition) and the comparison."""

import numpy as np
import pytest

from kstruct import standards
from kstruct.fem import generate_mesh
from kstruct.fem.static import solve_cases, solve_static
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import (
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadCase,
    LoadKind,
    Part,
    Project,
    active_project,
    with_case,
)
from kstruct.project_io import load_project, save_project
from kstruct.report.summary import compare_cases, summarize
from kstruct.standards import Verdict

STEEL = standards.materials()["s235"]


def _beam():
    L, b = 200.0, 20.0
    shape = _box(0, 0, 0, L, b, b)
    project = Project(parts=[Part(0, "beam", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 10, 10)))])]
    tip = FaceRef(0, face_at(shape, (L, 10, 10)))
    return project, tip


def test_loads_property_follows_the_active_case():
    project = Project()
    assert len(project.cases) == 1 and project.loads == []
    project.loads = [Load(LoadKind.GRAVITY, vector=(0, 0, -9810.0))]
    project.cases.append(LoadCase("충격", [Load(LoadKind.TEMPERATURE, magnitude=10.0)]))
    assert len(project.all_loads()) == 2
    project.active_case = 1
    assert project.loads[0].kind == LoadKind.TEMPERATURE
    view = with_case(project, 0)
    assert view.loads[0].kind == LoadKind.GRAVITY and project.active_case == 1  # untouched
    project.active_case = 99  # out of range is clamped, never an IndexError
    assert project.case.name == "충격"


def test_cases_round_trip_and_old_files(tmp_path):
    project, tip = _beam()
    project.loads = [Load(LoadKind.FORCE, [tip], vector=(0, 0, -100.0))]
    project.cases.append(LoadCase("옆 하중", [Load(LoadKind.FORCE, [tip], vector=(0, 50.0, 0))]))
    project.active_case = 1
    loaded = load_project(save_project(project, tmp_path / "c.kst"))
    assert [c.name for c in loaded.cases] == ["하중 케이스 1", "옆 하중"]
    assert loaded.cases == project.cases and loaded.active_case == 1
    # a hidden part drops its loads from every case, not only the active one
    other = _box(0, 0, 50, 10, 10, 10)
    project.parts.append(Part(1, "other", other, "s235", visible=False))
    project.cases[0].loads.append(Load(LoadKind.PRESSURE, [FaceRef(1, 0)], magnitude=1.0))
    active = active_project(project)
    assert len(active.cases[0].loads) == 1 and len(active.cases[1].loads) == 1
    assert len(project.cases[0].loads) == 2  # the original is not modified


def test_old_format_without_cases_loads(tmp_path):
    import json
    import zipfile

    project, tip = _beam()
    project.loads = [Load(LoadKind.FORCE, [tip], vector=(0, 0, -100.0))]
    path = save_project(project, tmp_path / "v1.kst")
    # rewrite project.json in the version-1 layout (flat "loads", no "cases")
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read("project.json"))
        parts = {n: z.read(n) for n in z.namelist() if n != "project.json"}
    doc["version"] = 1
    doc["loads"] = doc.pop("cases")[0]["loads"]
    doc.pop("active_case")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("project.json", json.dumps(doc))
        for n, data in parts.items():
            z.writestr(n, data)
    loaded = load_project(path)
    assert len(loaded.cases) == 1 and loaded.loads[0].vector == (0, 0, -100.0)


def test_cases_solve_independently_and_superpose():
    project, tip = _beam()
    project.cases = [
        LoadCase("아래", [Load(LoadKind.FORCE, [tip], vector=(0, 0, -1000.0))]),
        LoadCase("옆", [Load(LoadKind.FORCE, [tip], vector=(0, 500.0, 0))]),
        LoadCase("둘 다", [Load(LoadKind.FORCE, [tip], vector=(0, 500.0, -1000.0))]),
    ]
    mesh = generate_mesh(project, "coarse")
    mats = {0: STEEL}
    messages = []
    results = solve_cases(project, mesh, mats, lambda f, m: messages.append((f, m)))
    assert len(results) == 3
    assert any(m.startswith("[옆]") for _, m in messages)
    down, side, both = results
    # linear: the combined case is the sum of the two
    assert both.displacement == pytest.approx(down.displacement + side.displacement, abs=1e-9)
    assert np.abs(side.displacement[:, 1]).max() > np.abs(side.displacement[:, 2]).max()
    # solve_static alone still uses the active case
    project.active_case = 1
    alone = solve_static(project, mesh, mats)
    assert alone.displacement == pytest.approx(side.displacement, abs=1e-12)

    pack = standards.effective_pack(project)
    summaries = [summarize(with_case(project, i), r, pack) for i, r in enumerate(results)]
    overview = compare_cases([c.name for c in project.cases], summaries)
    assert overview.worst == 2  # the combined load governs
    assert "둘 다" in overview.headline
    assert summaries[2].governing_fos < summaries[0].governing_fos < summaries[1].governing_fos
    assert all(s.verdict in (Verdict.OK, Verdict.WARN, Verdict.FAIL) for s in summaries)


def test_cases_share_one_assembly_and_factorization(monkeypatch):
    """K and fixtures are the same in every case, so only the first case factorizes."""
    import kstruct.fem.static as static

    project, tip = _beam()
    project.cases = [
        LoadCase(f"{i}", [Load(LoadKind.FORCE, [tip], vector=(0, 0, -100.0 * (i + 1)))])
        for i in range(3)
    ]
    mesh = generate_mesh(project, "coarse")
    calls = {"factorize": 0, "assemble": 0}
    real_solver, real_stiffness = static.make_solver, static.stiffness_matrix

    def counting_solver(*args):
        calls["factorize"] += 1
        return real_solver(*args)

    def counting_stiffness(*args):
        calls["assemble"] += 1
        return real_stiffness(*args)

    monkeypatch.setattr(static, "make_solver", counting_solver)
    monkeypatch.setattr(static, "stiffness_matrix", counting_stiffness)
    results = solve_cases(project, mesh, {0: STEEL})
    assert calls == {"factorize": 1, "assemble": 1}
    tips = [r.displacement[:, 2].min() for r in results]
    assert tips[1] == pytest.approx(2 * tips[0], rel=1e-9)
    assert tips[2] == pytest.approx(3 * tips[0], rel=1e-9)


def test_report_has_a_case_table():
    from kstruct.report.pdf import build_html

    project, tip = _beam()
    project.cases = [
        LoadCase("A", [Load(LoadKind.FORCE, [tip], vector=(0, 0, -100.0))]),
        LoadCase("B", [Load(LoadKind.FORCE, [tip], vector=(0, 0, -300.0))]),
    ]
    mesh = generate_mesh(project, "coarse")
    results = solve_cases(project, mesh, {0: STEEL})
    pack = standards.effective_pack(project)
    summaries = [summarize(with_case(project, i), r, pack) for i, r in enumerate(results)]
    overview = compare_cases(["A", "B"], summaries)
    project.active_case = overview.worst
    html = build_html(
        project, results[overview.worst], summaries[overview.worst], [], {0: STEEL}, overview
    )
    assert "하중 케이스 비교" in html and "B ◀ 가장 불리" in html
    assert "'B' 케이스의 것입니다" in html
    # a single case keeps the old layout
    html1 = build_html(project, results[0], summaries[0], [], {0: STEEL}, None)
    assert "하중 케이스 비교" not in html1
