import re
from pathlib import Path

import numpy as np
import pytest

from kstruct import standards
from kstruct.geometry.face_info import face_info
from kstruct.geometry.samples import _box
from kstruct.geometry.tessellate import faces_of
from kstruct.ui import help as kshelp

UI_SRC = Path(__file__).parents[1] / "src" / "kstruct" / "ui"  # help keys live in UI code
PREFIXES = "step|view|file|help|project|connection|material|fixture|load|mesh|result|analysis|modal|fatigue|bolt"
HELP_KEY = rf'"((?:{PREFIXES})\.[a-z_]+)"'


def test_every_help_key_used_in_code_exists():
    topics = kshelp._topics()
    used = set()
    for py in UI_SRC.rglob("*.py"):
        used |= set(re.findall(HELP_KEY, py.read_text("utf-8")))
    assert used, "no help keys found — regex out of date?"
    assert used <= topics.keys(), f"missing help text: {sorted(used - topics.keys())}"
    for t in topics.values():
        assert set(t.related) <= topics.keys(), f"{t.key} links to unknown topic"


def test_face_info_plane():
    faces = faces_of(_box(0, 0, 0, 10, 20, 30))
    info = face_info(faces[0])
    assert info.kind == "평면"
    assert info.area in (pytest.approx(200), pytest.approx(300), pytest.approx(600))


def _window(qtbot):
    from kstruct.ui.main_window import MainWindow

    window = MainWindow()
    window.ask_before_close = False  # teardown closes the window: no modal prompt
    qtbot.addWidget(window)
    window.show()
    return window


def test_viewer_picks_upper_plate(qtbot):
    window = _window(qtbot)
    window.open_sample("two_plates")
    assert window.panels[0].parts.count() == 2
    assert len(window.project.connections) == 1  # auto-bonded on load

    viewer = window.viewer
    viewer.plotter.view_xy()  # look down -z: the upper plate covers the center
    viewer.plotter.render()
    w, h = viewer.plotter.window_size
    ref = viewer._pick(w // 2, h // 2)
    assert ref is not None and ref.part_id == 1


def test_full_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef
    from kstruct.report.pdf import build_html, write_pdf
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    shape = window.project.parts[0].shape

    window._go(mw.STEP_MATERIAL)
    material = window.panels[mw.STEP_MATERIAL]
    material.material.setCurrentIndex(material.material.findData("s235"))
    material._material_chosen()
    assert window.project.parts[0].material_id == "s235"

    window._go(mw.STEP_FIXTURE)
    window.viewer.set_selection([FaceRef(0, face_at(shape, (0, 30, 40)))])  # back of the wall
    window.panels[mw.STEP_FIXTURE]._add()
    assert len(window.project.fixtures) == 1

    window._go(mw.STEP_LOAD)
    loads = window.panels[mw.STEP_LOAD]
    window.viewer.set_selection([FaceRef(0, face_at(shape, (90, 30, 8)))])  # top of the base
    loads.value.setValue(2000)
    loads.direction.setCurrentIndex(0)  # push the face
    loads._add()
    load = window.project.loads[0]
    assert load.vector == pytest.approx((0, 0, -2000))

    window._go(mw.STEP_SOLVE)
    assert all(ok for ok, _ in window.readiness())
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
    assert window.nav.currentRow() == mw.STEP_RESULT
    assert window.summary.parts[0].min_fos > 0
    assert window.panels[mw.STEP_RESULT].card_title.text()

    for i in range(window.panels[mw.STEP_RESULT].field.count()):
        window.panels[mw.STEP_RESULT].field.setCurrentIndex(i)

    images = window.capture_result_images()
    assert len(images) == 3 and all(png[1:4] == b"PNG" for _, png in images)
    pdf = write_pdf(
        build_html(window.project, window.result, window.summary, images), tmp_path / "r.pdf"
    )
    assert pdf.stat().st_size > 10_000


def _bracket_with_material(qtbot):
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    for part in window.project.parts:
        part.material_id = "al6061t6"
    window._go(mw.STEP_SOLVE)
    return window


def _run(qtbot, window, tmp_path):
    from kstruct.report.pdf import build_html, write_pdf
    from kstruct.ui import main_window as mw

    assert all(ok for ok, _ in window.readiness()), window.readiness()
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
    assert window.nav.currentRow() == mw.STEP_RESULT
    images = window.capture_result_images()
    html = build_html(window.project, window.result, window.summary, images, window.materials)
    assert write_pdf(html, tmp_path / "r.pdf").stat().st_size > 10_000
    return window.result


def test_modal_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import AnalysisKind, FaceRef, Fixture, FixtureKind

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    window.project.analysis = AnalysisKind.MODAL
    window.project.operating_frequency = 50.0
    window.panels[5].refresh()
    result = _run(qtbot, window, tmp_path)
    assert len(result.frequencies) == 6
    assert np.all(np.diff(result.frequencies) >= 0)
    panel = window.panels[6]
    assert panel.mode.count() == 6
    panel.mode.setCurrentIndex(2)
    window.viewer._anim_step()  # animation advances without error


def test_thermal_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import AnalysisKind, FaceRef, Load, LoadKind

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.loads = [
        Load(LoadKind.HEAT_FLOW, [FaceRef(0, face_at(shape, (90, 30, 8)))], magnitude=5.0),
        Load(LoadKind.CONVECTION, [FaceRef(0, face_at(shape, (0, 30, 40)))], magnitude=50.0),
        Load(
            LoadKind.RADIATION,
            [FaceRef(0, face_at(shape, (50, 30, 16)))],
            magnitude=0.9,
            ambient=20.0,
        ),
    ]
    window.project.analysis = AnalysisKind.THERMAL
    result = _run(qtbot, window, tmp_path)
    assert result.temperature.min() > 20.0
    assert result.heat_out == pytest.approx(5.0, rel=0.02)
    assert result.iterations > 1  # radiation makes the problem nonlinear


def test_fatigue_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    window.project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -1500.0))
    ]
    window.project.load_cycle = "reversed"
    _run(qtbot, window, tmp_path)
    part = window.summary.parts[0]
    # fatigue governs: reversed loading is far harsher than one static application
    assert part.fatigue_fos < part.min_fos
    keys = [window.panels[6].field.itemData(i) for i in range(window.panels[6].field.count())]
    assert "ffos" in keys and "life" in keys


def test_buckling_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import AnalysisKind, FaceRef, Fixture, FixtureKind, Load, LoadKind

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    # stand the bracket on its base and push down on the top edge of the wall
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (50, 30, 0)))])
    ]
    window.project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (4, 30, 80)))], vector=(0, 0, -5000.0))
    ]
    window.project.analysis = AnalysisKind.BUCKLING
    window.project.n_modes = 3
    result = _run(qtbot, window, tmp_path)
    assert result.factors[0] > 0
    assert window.panels[6].mode.count() == 3
    assert "좌굴" in window.summary.headline


def test_radiation_load_asks_for_emissivity_and_surroundings(qtbot):
    from kstruct.model import LoadKind

    window = _window(qtbot)
    window.open_sample("bracket")
    panel = window.panels[3]
    panel.kind.setCurrentIndex(panel.kind.findData(LoadKind.RADIATION))
    assert panel.value.value() == pytest.approx(0.8)
    assert "방사율" in panel.value_label.text()
    assert panel.ambient.isVisibleTo(panel) and panel.ambient_label.text().startswith("주변 온도")
    panel.kind.setCurrentIndex(panel.kind.findData(LoadKind.CONVECTION))
    assert panel.ambient_label.text().startswith("주변 공기 온도")
    panel.kind.setCurrentIndex(panel.kind.findData(LoadKind.FORCE))
    assert not panel.ambient.isVisibleTo(panel)


def test_connection_kind_can_be_switched(qtbot):
    from kstruct.model import ConnectionKind

    window = _window(qtbot)
    window.open_sample("two_plates")
    panel = window.panels[0]
    panel.refresh()
    assert panel.connections.currentRow() == 0
    panel.conn_kind.setCurrentIndex(panel.conn_kind.findData(ConnectionKind.CONTACT))
    panel._kind_chosen()
    assert window.project.connections[0].kind == ConnectionKind.CONTACT
    assert panel.connections.item(0).text().startswith("접촉 (마찰 없음)")
    assert panel.friction_row.isVisibleTo(panel)
    panel.friction.setValue(0.2)
    assert window.project.connections[0].friction == pytest.approx(0.2)
    assert "마찰 μ=0.2" in panel.connections.item(0).text()
    panel.conn_kind.setCurrentIndex(panel.conn_kind.findData(ConnectionKind.BONDED))
    panel._kind_chosen()
    assert not panel.friction_row.isVisibleTo(panel)


def test_bolt_workflow(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import ConnectionKind, FaceRef, Fixture, FixtureKind, Load, LoadKind
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bolted")
    project = window.project
    lower, upper = (p.shape for p in project.parts)
    for part in project.parts:
        part.material_id = "s235"
    project.connections[0].kind = ConnectionKind.CONTACT  # plates only pressed together
    project.mesh_quality = "coarse"  # contact + preload iterations: keep the test quick

    window._go(mw.STEP_FIXTURE)
    panel = window.panels[mw.STEP_FIXTURE]
    for x, y in ((10, 10), (70, 10), (40, 50)):
        window.viewer.set_selection(
            [
                FaceRef(0, face_at(lower, (x + 5.5, y, 2.5))),
                FaceRef(1, face_at(upper, (x + 5.5, y, 7.5))),
            ]
        )
        assert panel.bolt_add.isEnabled()
        assert panel.bolt_size.currentData() == "M10"  # suggested from the 11 mm hole
        panel._add_bolt()
    assert len(project.bolts) == 3
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (40, 30, 0)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (40, 30, 10)))], vector=(0, 0, 3000.0))
    ]

    window._go(mw.STEP_SOLVE)
    _run(qtbot, window, tmp_path)
    bolts = window.result.bolts
    assert len(bolts) == 3
    # preloaded joint: the clamped interface absorbs most of the 3 kN pull
    change = sum(b.axial - b.preload for b in bolts)
    assert abs(change) < 0.5 * 3000.0
    assert "볼트" in window.panels[mw.STEP_RESULT].summary.text()


def test_us_units_input_and_results(qtbot):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    window.project.standard = "astm"  # US pack -> inches, lbf, psi
    shape = window.project.parts[0].shape
    window._go(mw.STEP_LOAD)
    loads = window.panels[mw.STEP_LOAD]
    assert "lbf" in loads.value_label.text()
    window.viewer.set_selection([FaceRef(0, face_at(shape, (90, 30, 8)))])
    loads.direction.setCurrentIndex(7)  # -Z
    loads.value.setValue(100.0)  # lbf
    loads._add()
    assert window.project.loads[0].vector[2] == pytest.approx(-444.822, rel=1e-5)
    assert "lbf" in loads.items.item(0).text()


def test_hidden_part_is_excluded_from_checklist(qtbot):
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("two_plates")
    project = window.project
    for part in project.parts:
        part.material_id = "s235"
    lower, upper = (p.shape for p in project.parts)
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, 0)])]
    project.loads = [Load(LoadKind.FORCE, [FaceRef(1, 0)], vector=(0, 0, -10.0))]
    project.parts[1].visible = False  # hide the loaded plate
    window._go(mw.STEP_SOLVE)
    texts = [t for _, t in window.readiness()]
    assert any("숨긴 부품" in t and "하중 1개" in t for t in texts)
    assert any(t.startswith("모델: 부품 1개") for t in texts)
    # with nothing loaded on the visible part the run button stays disabled, with a reason
    assert any(not ok and "하중" in t for ok, t in window.readiness())


def test_custom_material_dialog_and_override_ui(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("KSTRUCT_DATA_DIR", str(tmp_path))
    from kstruct.ui import main_window as mw
    from kstruct.ui.dialogs import MaterialDialog, load_user_library

    window = _window(qtbot)
    window.open_sample("bracket")
    window._go(mw.STEP_MATERIAL)
    panel = window.panels[mw.STEP_MATERIAL]
    base = window.project.custom_materials  # the user library is merged on refresh
    assert base == {}

    dialog = MaterialDialog(standards.materials()["s355"], "SI")
    dialog.name.setText("특수강")
    dialog.yield_.setValue(420)
    m = dialog.material()
    assert m.id.startswith("user_") and m.yield_strength == 420 and m.tensile_strength == 470
    window.project.custom_materials[m.id] = m
    panel.refresh()
    assert panel.material.findData(m.id) >= 0
    panel.material.setCurrentIndex(panel.material.findData(m.id))
    panel._material_chosen()
    assert window.project.parts[0].material_id == m.id
    assert "사용자 정의 재료" in panel.props.text()
    # US display units convert the entered stress back to MPa
    us = MaterialDialog(None, "US")
    us.yield_.setValue(36_000)  # psi
    assert us.material().yield_strength == pytest.approx(248.2, abs=0.1)
    assert load_user_library() == {}  # nothing written until the user saves through the UI

    window._go(mw.STEP_SOLVE)
    solve = window.panels[mw.STEP_SOLVE]
    assert solve.target_default.isChecked() and not solve.targets["ductile"].isEnabled()
    solve.target_default.setChecked(False)
    solve.targets["ductile"].setValue(2.5)
    assert window.project.target_overrides["ductile"] == 2.5
    assert standards.effective_pack(window.project).target_ductile == 2.5
    solve.target_default.setChecked(True)
    assert window.project.target_overrides == {}


def test_load_value_survives_step_changes(qtbot):
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    window._go(mw.STEP_LOAD)
    loads = window.panels[mw.STEP_LOAD]
    loads.value.setValue(2345.0)
    window._go(mw.STEP_FIXTURE)
    window._go(mw.STEP_LOAD)
    assert loads.value.value() == 2345.0
    loads.kind.setCurrentIndex(1)  # pressure: a new kind gets its default
    assert loads.value.value() == 1.0
    window.project.standard = "astm"  # units changed: the number must be reset
    window._go(mw.STEP_FIXTURE)
    window._go(mw.STEP_LOAD)
    assert "psi" in loads.value_label.text()


def test_rotating_loads_from_the_panel(qtbot):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, LoadKind
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    warnings = []
    window.warn = warnings.append  # never open a modal box in the test
    shape = window.project.parts[0].shape
    hole = FaceRef(0, face_at(shape, (75, 15, 4)))  # inside wall of a base hole (x=70, r=5)
    window._go(mw.STEP_LOAD)
    loads = window.panels[mw.STEP_LOAD]

    loads.kind.setCurrentIndex(loads.kind.findData(LoadKind.BEARING))
    assert "N" in loads.value_label.text() and loads.direction.isVisible()
    window.viewer.set_selection([hole])
    loads.value.setValue(500)
    loads.direction.setCurrentIndex(2)  # +X
    loads._add()
    assert window.project.loads[-1].kind == LoadKind.BEARING
    assert window.project.loads[-1].vector == pytest.approx((500, 0, 0))
    assert "베어링 하중 500.0 N" in loads.items.item(0).text()

    loads.kind.setCurrentIndex(loads.kind.findData(LoadKind.TORQUE))
    assert "N·m" in loads.value_label.text()
    window.viewer.set_selection([hole])
    loads.axis.setCurrentIndex(0)  # the hole's own axis (z)
    loads.value.setValue(2.5)  # N·m
    loads._add()
    torque = window.project.loads[-1]
    assert torque.kind == LoadKind.TORQUE and torque.magnitude == pytest.approx(2500.0)
    assert abs(torque.vector[2]) == pytest.approx(1.0) and torque.point is not None
    assert "토크 2.5 N·m" in loads.items.item(1).text()

    loads.kind.setCurrentIndex(loads.kind.findData(LoadKind.CENTRIFUGAL))
    assert "rpm" in loads.value_label.text()
    window.viewer.set_selection([])
    loads.axis.setCurrentIndex(3)  # Z axis through the model centre
    loads.value.setValue(1500)
    loads._add()
    spin = window.project.loads[-1]
    assert spin.kind == LoadKind.CENTRIFUGAL and spin.magnitude == pytest.approx(
        1500 * 2 * np.pi / 60
    )
    assert spin.vector == (0.0, 0.0, 1.0) and spin.point is not None
    assert "원심력 1,500 rpm, Z 축" in loads.items.item(2).text()
    # a flat face cannot take a bearing load: refused with a message, nothing added
    loads.kind.setCurrentIndex(loads.kind.findData(LoadKind.BEARING))
    window.viewer.set_selection([FaceRef(0, face_at(shape, (90, 30, 8)))])
    loads._add()
    assert len(window.project.loads) == 3 and len(warnings) == 1
    window.viewer.annotate(window.project.fixtures, window.project.loads, [])  # axis lines draw


def test_load_cases_in_the_panel(qtbot):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, LoadKind
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    for part in window.project.parts:
        part.material_id = "s235"
    shape = window.project.parts[0].shape
    window._go(mw.STEP_LOAD)
    loads = window.panels[mw.STEP_LOAD]
    assert loads.case.count() == 1 and not loads.case_remove.isEnabled()
    window.viewer.set_selection([FaceRef(0, face_at(shape, (90, 30, 8)))])
    loads.value.setValue(1000)
    loads._add()
    loads._add_case("충격")
    assert window.project.active_case == 1 and loads.case.currentIndex() == 1
    assert loads.items.count() == 0  # the new case starts empty
    window.viewer.set_selection([FaceRef(0, face_at(shape, (90, 30, 8)))])
    loads.kind.setCurrentIndex(loads.kind.findData(LoadKind.GRAVITY))
    loads.value.setValue(3)
    loads._add()
    assert [len(c.loads) for c in window.project.cases] == [1, 1]
    loads.case.setCurrentIndex(0)
    assert window.project.active_case == 0 and "힘" in loads.items.item(0).text()

    window._go(mw.STEP_SOLVE)
    assert any("케이스 2개 모두" in t for _, t in window.readiness())
    window.project.cases[1].loads.clear()
    assert any(not ok and "하중이 없는 케이스: 충격" in t for ok, t in window.readiness())
    loads.case.setCurrentIndex(1)
    loads._remove_case()
    assert len(window.project.cases) == 1 and window.project.active_case == 0


def test_multi_case_strength_run(qtbot):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadCase, LoadKind
    from kstruct.ui import main_window as mw

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    top = FaceRef(0, face_at(shape, (90, 30, 8)))
    window.project.cases = [
        LoadCase("가벼움", [Load(LoadKind.FORCE, [top], vector=(0, 0, -500.0))]),
        LoadCase("무거움", [Load(LoadKind.FORCE, [top], vector=(0, 0, -4000.0))]),
    ]
    window.panels[mw.STEP_SOLVE].refresh()
    assert all(ok for ok, _ in window.readiness()), window.readiness()
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
    assert len(window.case_results) == 2 and window.overview is not None
    assert window.overview.worst == 1 and window.project.active_case == 1  # heavy governs
    panel = window.panels[mw.STEP_RESULT]
    assert panel.case.count() == 2 and panel.case.currentIndex() == 1
    assert "가장 불리" in panel.case.itemText(1)
    assert "2개 케이스 중 가장 불리한 것: 무거움" in panel.summary.text()
    heavy = window.summary.governing_fos
    window.select_case(0)
    assert window.project.active_case == 0 and window.summary.governing_fos > heavy
    assert "지금 보는 케이스: 가벼움" in panel.summary.text()


def test_mesh_refinement_from_the_panel(qtbot):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef
    from kstruct.ui import main_window as mw

    window = _window(qtbot)
    window.open_sample("bracket")
    shape = window.project.parts[0].shape
    window._go(mw.STEP_MESH)
    assert window.viewer.selection_enabled
    panel = window.panels[mw.STEP_MESH]
    assert not panel.refine_add.isEnabled() and panel.refine_size.value() > 0
    hole = FaceRef(0, face_at(shape, (75, 15, 4)))
    window.viewer.set_selection([hole])
    assert panel.refine_add.isEnabled() and "원통면" in panel.selection_label.text()
    panel.refine_size.setValue(1.5)
    panel._add_refinement()
    assert window.project.mesh_refinements[0].size == 1.5
    assert panel.refinements.count() == 1 and "1.5 mm" in panel.refinements.item(0).text()
    panel.refinements.setCurrentRow(0)
    panel._remove_refinement()
    assert window.project.mesh_refinements == []


def test_section_view_and_previous_comparison(qtbot, tmp_path):
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind
    from kstruct.report.export import export_csv
    from kstruct.ui import main_window as mw

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    window.project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -1000.0))
    ]
    window.project.mesh_quality = "coarse"
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
    panel = window.panels[mw.STEP_RESULT]
    assert window.previous is None and "이전 해석과 비교" not in panel.summary.text()
    first = window.summary.governing_fos
    # section view: controls appear, animation is off, display updates without error
    panel.clip.setChecked(True)
    assert panel.clip_row.isVisible() and not panel.animate.isEnabled()
    panel.clip_axis.setCurrentIndex(2)
    panel.clip_pos.setValue(30)
    assert panel.clip_fraction() == (2, 0.3)
    panel.clip.setChecked(False)
    # export from the window's result
    out = export_csv(tmp_path / "e.csv", window.project, window.result, window.summary)
    assert out.stat().st_size > 1000
    # second run: the comparison line names the previous mesh and the change
    window.project.mesh_quality = "normal"
    window.invalidate_mesh()
    window._go(mw.STEP_SOLVE)
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
    assert window.previous is not None and window.previous["fos"] == pytest.approx(first)
    assert "이전 해석과 비교" in panel.summary.text() and "거침 메시" in panel.summary.text()


def test_recent_files_autosave_and_recovery(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("KSTRUCT_DATA_DIR", str(tmp_path / "data"))
    from kstruct.geometry.samples import l_bracket, write_step
    from kstruct.ui import prefs

    assert prefs.recent_files() == [] and prefs.pending_autosave() is None
    model = write_step(l_bracket(), tmp_path / "my_bracket.step")
    window = _window(qtbot)
    window.load(model)
    assert prefs.recent_files() == [model] and prefs.last_dir() == str(tmp_path)
    assert window._dirty  # imported, not yet saved anywhere
    window.project.parts[0].material_id = "s235"
    window._autosave()
    pending = prefs.pending_autosave()
    assert pending is not None and pending["source"] == str(model)
    # a new window recovers the autosaved work
    other = _window(qtbot)
    assert other.recover()
    assert other.project is not None and other.project.parts[0].material_id == "s235"
    assert other.project.source_path == model and other._dirty
    # saving the project clears the autosave and remembers the .kst
    kst = tmp_path / "saved.kst"
    other.project_path = kst
    other.save_project()
    assert kst.is_file() and not other._dirty and prefs.pending_autosave() is None
    assert prefs.recent_files()[0] == kst
    # window layout round trip
    other._save_window_state()
    geometry, sizes = prefs.window_state()
    assert geometry and sizes and len(sizes) == 2
    # the recent menu lists the files
    other._fill_recent_menu()
    assert [a.text() for a in other.recent_menu.actions()] == ["saved.kst", "my_bracket.step"]


def test_solve_can_be_cancelled(qtbot):
    """The cancel button shows while solving; cancelling stops the job without a result or
    an error dialog, and the window can solve again afterwards."""
    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind
    from kstruct.ui import main_window as mw

    window = _bracket_with_material(qtbot)
    shape = window.project.parts[0].shape
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    window.project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -1000.0))
    ]
    window.project.mesh_quality = "coarse"
    window.warn = lambda *a, **k: pytest.fail("cancelling must not show an error")
    window._go(mw.STEP_SOLVE)
    panel = window.panels[mw.STEP_SOLVE]
    window.solve()  # no mesh yet: meshing runs first, in worker processes
    assert panel.cancel.isVisible()
    window.cancel_job()
    qtbot.waitUntil(lambda: not window._busy, timeout=60_000)
    assert window.result is None
    assert "취소했습니다" in panel.status.text()
    assert not panel.cancel.isVisible() and panel.run.isEnabled()
    window.solve()
    qtbot.waitUntil(lambda: window.result is not None, timeout=120_000)
