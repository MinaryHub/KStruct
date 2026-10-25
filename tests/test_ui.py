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
    ]
    window.project.analysis = AnalysisKind.THERMAL
    result = _run(qtbot, window, tmp_path)
    assert result.temperature.min() > 20.0
    assert result.heat_out == pytest.approx(5.0, rel=0.02)


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
    assert panel.connections.item(0).text().startswith("접촉")


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
