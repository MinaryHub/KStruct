"""Release-readiness items: version sync, hidden parts, safety-factor overrides,
singularity hot-spots, report wording/units, custom materials, bug report bundle."""

import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

import kstruct
from kstruct import standards
from kstruct.fem import generate_mesh, solve_static
from kstruct.fem.singularity import CONCAVE_EDGE, FIXTURE_EDGE, find_singularities
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box, l_bracket
from kstruct.model import (
    Connection,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Material,
    Part,
    Project,
    active_project,
    excluded_by_hidden_parts,
)
from kstruct.project_io import load_project, save_project
from kstruct.report.describe import describe_load
from kstruct.report.summary import summarize
from kstruct.standards import Verdict

ROOT = Path(__file__).parents[1]


def test_version_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert kstruct.RELEASE_VERSION == pyproject["project"]["version"]
    assert kstruct.__version__.startswith(kstruct.RELEASE_VERSION)


def _kbuild():
    import importlib.util

    spec = importlib.util.spec_from_file_location("kbuild", ROOT / "packaging" / "build.py")
    kbuild = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kbuild)
    return kbuild


def test_signing_is_optional_and_configured_by_environment(tmp_path):
    """No certificate: no signing commands (the unsigned build of before). A .pfx or a
    custom command signs both the program and the installer; macOS signs the bundle and
    notarizes the disk image only with an identity and a notary profile."""
    kbuild = _kbuild()
    exe, setup = tmp_path / "KStruct.exe", tmp_path / "KStruct-Setup-1.exe"
    assert kbuild.windows_sign_commands([exe], env={}) == []
    pfx = kbuild.windows_sign_commands(
        [exe, setup],
        env={"KSTRUCT_SIGN_PFX": "c.pfx", "KSTRUCT_SIGN_PASSWORD": "pw", "KSTRUCT_SIGNTOOL": "st"},
    )
    assert [cmd[-1] for cmd in pfx] == [str(exe), str(setup)]
    assert pfx[0][:2] == ["st", "sign"] and "/tr" in pfx[0] and ["/p", "pw"] == pfx[0][-3:-1]
    custom = kbuild.windows_sign_commands([exe], env={"KSTRUCT_SIGN_COMMAND": "azsign {file}"})
    assert custom == [["azsign", str(exe)]]

    app, dmg = tmp_path / "KStruct.app", tmp_path / "KStruct.dmg"
    assert kbuild.macos_sign_commands(app, env={}) == []
    signed = kbuild.macos_sign_commands(app, env={"KSTRUCT_MACOS_IDENTITY": "Developer ID X"})
    assert "--options" in signed[0] and signed[0][-1] == str(app)
    assert (ROOT / "packaging" / "macos" / "entitlements.plist").is_file()
    assert kbuild.macos_notarize_commands(dmg, env={"KSTRUCT_NOTARY_PROFILE": "p"}) == []
    notarize = kbuild.macos_notarize_commands(
        dmg, env={"KSTRUCT_NOTARY_PROFILE": "p", "KSTRUCT_MACOS_IDENTITY": "Developer ID X"}
    )
    assert notarize[0][:3] == ["xcrun", "notarytool", "submit"] and notarize[1][-1] == str(dmg)

    (tmp_path / "a.exe").write_bytes(b"abc")
    sums = kbuild.write_checksums([tmp_path / "a.exe"])
    assert sums.read_text() == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a.exe\n"
    )


def test_build_stamp_raises_the_version_every_build(tmp_path, monkeypatch):
    """packaging/build.py: the counter file goes 1, 2, 3 … and the stamp module turns the
    release version into release.build; an explicit number (CI run number) wins."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kbuild", ROOT / "packaging" / "build.py")
    kbuild = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kbuild)
    monkeypatch.delenv("KSTRUCT_BUILD_NUMBER", raising=False)
    counter = tmp_path / "build_number.txt"
    assert [kbuild.next_build_number(counter) for _ in range(3)] == [1, 2, 3]
    assert counter.read_text() == "3\n"
    assert kbuild.next_build_number(counter, explicit=57) == 57
    monkeypatch.setenv("KSTRUCT_BUILD_NUMBER", "99")
    assert kbuild.next_build_number(counter) == 99 and counter.read_text() == "3\n"

    # the stamp is a module next to kstruct/__init__.py: write it into a copy of the package
    pkg = tmp_path / "src" / "kstruct"
    pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text("utf-8"), "utf-8")
    (pkg / "__init__.py").write_text((ROOT / "src/kstruct/__init__.py").read_text("utf-8"), "utf-8")
    version = kbuild.write_stamp(57, pkg / "_build.py", commit="abc1234")
    assert version == kstruct.RELEASE_VERSION + ".57"
    stamp = (pkg / "_build.py").read_text("utf-8")
    assert "BUILD_NUMBER = 57" in stamp and 'GIT_COMMIT = "abc1234"' in stamp
    out = subprocess.run(
        [sys.executable, "-c", "import kstruct; print(kstruct.__version__, kstruct.build_info())"],
        cwd=tmp_path / "src",
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[0] == version and "abc1234" in out.stdout


# --- hidden parts -----------------------------------------------------------------------


def _two_blocks() -> Project:
    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = Project(parts=[Part(0, "lower", lower, "s235"), Part(1, "upper", upper, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (5, 5, 0)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (5, 5, 100)))], vector=(0, 0, -100.0)),
        Load(LoadKind.GRAVITY, vector=(0, 0, -9810.0)),
    ]
    project.connections = [
        Connection(
            ConnectionKind.BONDED,
            [FaceRef(0, face_at(lower, (5, 5, 50)))],
            [FaceRef(1, face_at(upper, (5, 5, 50)))],
        )
    ]
    return project


def test_hidden_part_drops_its_conditions():
    project = _two_blocks()
    assert active_project(project) is project  # nothing hidden: same object
    project.parts[1].visible = False
    active = active_project(project)
    assert [p.id for p in active.parts] == [0]
    assert len(active.fixtures) == 1
    assert [ld.kind for ld in active.loads] == [
        LoadKind.GRAVITY
    ]  # the force was on the hidden part
    assert active.connections == []
    assert excluded_by_hidden_parts(project) == ["하중 1개", "연결 1개"]
    # and the reduced project still solves (previously: "고정할 면이 메시에 없습니다")
    mesh = generate_mesh(active, "coarse")
    r = solve_static(active, mesh, {0: standards.materials()["s235"]})
    assert r.reactions[0][2] == pytest.approx(10 * 10 * 50 * 7.85e-9 * 9810, rel=1e-6)


# --- safety-factor overrides ------------------------------------------------------------


def test_target_override_changes_verdict_and_report_flag():
    project = Project(parts=[Part(0, "p", _box(0, 0, 0, 10, 10, 10), "s235")])
    pack = standards.effective_pack(project)
    assert pack.target_ductile == 1.5
    project.target_overrides = {"ductile": 4.0}
    pack = standards.effective_pack(project)
    assert pack.target_ductile == 4.0 and pack.target_brittle == 3.0
    m = standards.materials()["s235"]
    assert pack.target(m) == 4.0
    assert standards.verdict(2.0, pack.target(m)) == Verdict.WARN


# --- singularities / hot-spot -----------------------------------------------------------


def _bracket() -> Project:
    shape = l_bracket()
    project = Project(parts=[Part(0, "bracket", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -2000.0))
    ]
    return project


def test_concave_edge_and_fixture_ring_are_found():
    project = _bracket()
    mesh = generate_mesh(project, "coarse")
    sg = find_singularities(project, mesh)
    concave = mesh.nodes[sg.kind == CONCAVE_EDGE]
    assert len(concave) > 5
    # the inner corner of the L runs along y at x = 8, z = 8
    assert np.allclose(concave[:, 0], 8.0, atol=1e-6) and np.allclose(concave[:, 2], 8.0, atol=1e-6)
    ring = mesh.nodes[sg.kind == FIXTURE_EDGE]
    assert len(ring) > 5 and np.allclose(ring[:, 0], 0.0, atol=1e-6)
    # a plain box has no concave edge
    box = Project(parts=[Part(0, "b", _box(0, 0, 0, 10, 10, 10), "s235")])
    sg_box = find_singularities(box, generate_mesh(box, "coarse"))
    assert not np.any(sg_box.kind == CONCAVE_EDGE)


def test_hotspot_verdict_is_mesh_independent():
    """The raw peak at the sharp inner corner grows with refinement; the hot-spot value and
    the verdict based on it do not."""
    project = _bracket()
    mats = {0: standards.materials()["s235"]}
    pack = standards.effective_pack(project)
    summaries = []
    for quality in ("coarse", "fine"):
        mesh = generate_mesh(project, quality)
        summaries.append(summarize(project, solve_static(project, mesh, mats), pack))
    coarse, fine = (s.parts[0] for s in summaries)
    assert coarse.singular_kind == fine.singular_kind == "오목 모서리(필렛 없음)"
    assert fine.max_stress > coarse.max_stress  # the singular peak keeps rising
    assert fine.hotspot_fos == pytest.approx(coarse.hotspot_fos, rel=0.08)
    assert coarse.verdict == fine.verdict == Verdict.OK
    assert summaries[0].headline.startswith("✅") and "특이점 제외" in summaries[0].headline
    assert any("응력 특이점" in n for n in summaries[0].notes)


def test_brittle_material_keeps_the_peak():
    project = _bracket()
    mats = {0: standards.materials()["fc250"]}
    mesh = generate_mesh(project, "coarse")
    s = summarize(project, solve_static(project, mesh, mats), standards.effective_pack(project))
    part = s.parts[0]
    assert part.singular_kind is not None
    assert part.governing_fos == part.min_fos
    assert any("취성재" in n for n in s.notes)


# --- report wording and units -----------------------------------------------------------


def test_report_lists_thermal_conditions_in_display_units(tmp_path):
    from kstruct.report.pdf import build_html

    project = _bracket()
    project.standard = "astm"  # US units
    shape = project.parts[0].shape
    project.loads += [
        Load(LoadKind.FACE_TEMPERATURE, [FaceRef(0, face_at(shape, (0, 30, 40)))], magnitude=100.0),
        Load(LoadKind.CONVECTION, [FaceRef(0, face_at(shape, (90, 30, 8)))], magnitude=20.0),
    ]
    project.connections = []
    project.mesh_quality = "coarse"
    mats = {0: standards.materials()["s235"]}
    mesh = generate_mesh(project, "coarse")
    result = solve_static(project, mesh, mats)
    html = build_html(
        project, result, summarize(project, result, standards.effective_pack(project)), []
    )
    assert "고정 온도 212.0 °F" in html  # 100 °C, named as what it is
    assert "대류 h=20 W/m²K" in html
    assert "온도 변화" not in html  # the old report called every thermal condition this
    assert "lbf" in html and "psi" in html
    assert re.search(r"힘 [\d,.]+ lbf", html)
    assert "KStruct " + kstruct.__version__ in html
    assert "미국 단위" in html and "거침" in html
    assert "열응력 포함" in html


def test_report_disclaimer_names_the_analysis_kind():
    from kstruct.report.pdf import disclaimer

    assert "정상상태 열전도" in disclaimer("정상상태 열전도 유한요소해석")
    assert "선형 정적" in disclaimer()


def test_describe_load_covers_every_kind():
    project = Project(parts=[Part(0, "p", _box(0, 0, 0, 10, 10, 10))])
    face = [FaceRef(0, 0)]
    texts = {
        LoadKind.FORCE: describe_load(project, Load(LoadKind.FORCE, face, vector=(0, 0, -10.0))),
        LoadKind.PRESSURE: describe_load(project, Load(LoadKind.PRESSURE, face, magnitude=2.0)),
        LoadKind.GRAVITY: describe_load(project, Load(LoadKind.GRAVITY, vector=(0, 0, -9810.0))),
        LoadKind.TEMPERATURE: describe_load(project, Load(LoadKind.TEMPERATURE, magnitude=30.0)),
        LoadKind.FACE_TEMPERATURE: describe_load(
            project, Load(LoadKind.FACE_TEMPERATURE, face, magnitude=80.0)
        ),
        LoadKind.HEAT_FLOW: describe_load(project, Load(LoadKind.HEAT_FLOW, face, magnitude=5.0)),
        LoadKind.CONVECTION: describe_load(
            project, Load(LoadKind.CONVECTION, face, magnitude=10.0, ambient=25.0)
        ),
    }
    assert texts[LoadKind.FORCE].startswith("힘 10.0 N")
    assert texts[LoadKind.PRESSURE].startswith("압력 2 MPa")
    assert texts[LoadKind.GRAVITY] == "자중 1 g"
    assert texts[LoadKind.TEMPERATURE].startswith("온도 변화 +30.0 °C")
    assert texts[LoadKind.FACE_TEMPERATURE].startswith("고정 온도 80.0 °C")
    assert texts[LoadKind.HEAT_FLOW].startswith("열 입력 5 W")
    assert "공기 25 °C" in texts[LoadKind.CONVECTION]


# --- custom materials -------------------------------------------------------------------


def _custom() -> Material:
    base = standards.materials()["s355"]
    rec = standards.material_record(base)
    rec.update({"id": "user_test0001", "ko": "특수강 (성적서)", "yield": 420, "names": {}})
    return standards.material_from_record(rec)


def test_material_record_round_trip():
    m = standards.materials()["al6061t6"]
    again = standards.material_from_record(standards.material_record(m))
    assert again == m


def test_custom_material_is_saved_in_project_and_used(tmp_path):
    project = Project(parts=[Part(0, "p", _box(0, 0, 0, 10, 10, 100))])
    m = _custom()
    project.custom_materials[m.id] = m
    project.parts[0].material_id = m.id
    project.target_overrides = {"ductile": 2.5}
    loaded = load_project(save_project(project, tmp_path / "c.kst"))
    assert loaded.custom_materials[m.id] == m
    assert loaded.parts[0].material_id == m.id
    assert loaded.target_overrides == {"ductile": 2.5}
    mats = standards.project_materials(loaded)
    assert mats[m.id].yield_strength == 420 and "s235" in mats
    # old files without the new keys still load
    assert (
        load_project(
            save_project(Project(parts=[Part(0, "p", _box(0, 0, 0, 1, 1, 1))]), tmp_path / "o.kst")
        ).custom_materials
        == {}
    )


# --- bug report bundle ------------------------------------------------------------------


def test_bug_report_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("KSTRUCT_DATA_DIR", str(tmp_path / "data"))
    import logging

    from kstruct.paths import data_dir, setup_logging
    from kstruct.ui.dialogs import write_bug_report

    setup_logging()
    logging.getLogger("kstruct").warning("hello from the test")
    for h in logging.getLogger().handlers:
        h.flush()
    assert (data_dir() / "kstruct.log").read_text("utf-8").count("hello from the test") == 1
    project_file = save_project(
        Project(parts=[Part(0, "p", _box(0, 0, 0, 1, 1, 1))]), tmp_path / "p.kst"
    )
    out = write_bug_report(tmp_path / "r.zip", project_file, {"error.txt": "boom"})
    names = set(zipfile.ZipFile(out).namelist())
    assert {"environment.txt", "error.txt", "kstruct.log", "p.kst"} <= names
