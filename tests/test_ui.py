import re
from pathlib import Path

import pytest

from kstruct.geometry.face_info import face_info
from kstruct.geometry.samples import _box, two_plates
from kstruct.geometry.tessellate import faces_of
from kstruct.ui import help as kshelp

SRC = Path(__file__).parents[1] / "src" / "kstruct"


def test_every_help_key_used_in_code_exists():
    topics = kshelp._topics()
    used = set()
    for py in SRC.rglob("*.py"):
        used |= set(re.findall(r'"((?:step|view|file|help)\.[a-z_]+)"', py.read_text("utf-8")))
    assert used, "no help keys found — regex out of date?"
    assert used <= topics.keys(), f"missing help text: {sorted(used - topics.keys())}"
    for t in topics.values():
        assert set(t.related) <= topics.keys(), f"{t.key} links to unknown topic"


def test_face_info_plane():
    faces = faces_of(_box(0, 0, 0, 10, 20, 30))
    info = face_info(faces[0])
    assert info.kind == "평면"
    assert info.area in (pytest.approx(200), pytest.approx(300), pytest.approx(600))


def test_main_window_loads_sample_and_picks_face(qtbot):
    from kstruct.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._open_sample(two_plates, "")

    assert window.parts.count() == 2
    viewer = window.viewer
    viewer.plotter.view_xy()  # look down -z: the upper plate covers the center
    viewer.plotter.render()
    w, h = viewer.plotter.window_size
    ref = viewer._pick(w // 2, h // 2)
    assert ref is not None and ref.part_id == 1

    window._on_face_picked(ref)
    assert "면 #" in window.face_label.text()
