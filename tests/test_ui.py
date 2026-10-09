import re
from pathlib import Path

import numpy as np
import pytest

from kstruct.geometry.face_info import face_info
from kstruct.geometry.samples import _box
from kstruct.geometry.tessellate import faces_of
from kstruct.ui import help as kshelp

UI_SRC = Path(__file__).parents[1] / "src" / "kstruct" / "ui"  # help keys live in UI code
PREFIXES = "step|view|file|help|project|connection|material|fixture|load|mesh|result|analysis|modal|fatigue"
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
