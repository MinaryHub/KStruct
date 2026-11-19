"""Store-listing screenshots (docs/screenshots): one scene per run, e.g.
    QT_SCALE_FACTOR=1.5 uv run python packaging/store_screenshots.py 03_stress out.png 1280 720
(1920x1080), QT_SCALE_FACTOR=2 ... 1440 900 (2880x1800). Needs a display (xvfb-run on CI)."""

import os
import sys
import tempfile


def main():
    scene, out, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.environ["KSTRUCT_DATA_DIR"] = tempfile.mkdtemp()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    from kstruct.geometry.face_info import face_at
    from kstruct.model import (
        AnalysisKind,
        ConnectionKind,
        FaceRef,
        Fixture,
        FixtureKind,
        Load,
        LoadKind,
    )
    from kstruct.ui import main_window as mw

    win = mw.MainWindow()
    win.ask_before_close = False
    win.resize(w, h)
    win.show()
    dark = scene.endswith("_dark")
    if dark:
        win.toggle_theme()
    base = scene.removesuffix("_dark")

    def shoot(delay=1200):
        def go():
            app.processEvents()
            win.statusBar().clearMessage()
            win.grab().save(out)
            print(
                "saved",
                out,
                win.grab().size(),
                getattr(win, "summary", None) and win.summary.headline,
            )
            app.quit()

        QTimer.singleShot(delay, go)

    def after_result(prepare):
        def poll():
            if win.result is None:
                QTimer.singleShot(300, poll)
                return
            prepare()
            shoot()

        QTimer.singleShot(300, poll)

    def bracket(material="s355"):
        win.open_sample("bracket")
        for p in win.project.parts:
            p.material_id = material
        s = win.project.parts[0].shape
        return s, FaceRef(0, face_at(s, (0, 30, 40))), FaceRef(0, face_at(s, (90, 30, 8)))

    def field(key):
        v = win.panels[mw.STEP_RESULT]
        v.field.setCurrentIndex(v.field.findData(key))

    if base == "01_welcome":
        shoot(800)
    elif base == "02_setup":
        s, back, top = bracket()
        win.project.fixtures = [Fixture(FixtureKind.FIXED, [back])]
        win.project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, -3000.0))]
        win._annotate()
        win._go(mw.STEP_LOAD)
        win.panels[mw.STEP_LOAD].refresh()
        shoot()
    elif base in ("03_stress", "04_safety"):
        s, back, top = bracket()
        win.project.fixtures = [Fixture(FixtureKind.FIXED, [back])]
        win.project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, -3000.0))]
        win.project.mesh_quality = "fine"
        win._go(mw.STEP_SOLVE)
        win.solve()
        after_result(lambda: field("vm" if base == "03_stress" else "fos"))
    elif base == "05_bolts":
        win.open_sample("bolted")
        pr = win.project
        lower, upper = (p.shape for p in pr.parts)
        for p in pr.parts:
            p.material_id = os.environ.get("BOLT_MAT", "s355")
        pr.connections[0].kind = ConnectionKind.CONTACT
        win._go(mw.STEP_FIXTURE)
        panel = win.panels[mw.STEP_FIXTURE]
        for x, y in ((10, 10), (70, 10), (40, 50)):
            win.viewer.set_selection(
                [
                    FaceRef(0, face_at(lower, (x + 5.5, y, 2.5))),
                    FaceRef(1, face_at(upper, (x + 5.5, y, 7.5))),
                ]
            )
            panel._add_bolt()
        for b in pr.bolts:
            b.grade = os.environ.get("BOLT_GRADE", "10.9")
        pr.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (40, 30, 0)))])]
        pr.loads = [
            Load(
                LoadKind.FORCE,
                [FaceRef(1, face_at(upper, (40, 30, 10)))],
                vector=(0, 0, float(os.environ.get("BOLT_LOAD", "3000"))),
            )
        ]
        win._go(mw.STEP_SOLVE)
        win.solve()
        after_result(lambda: field("vm"))
    elif base == "06_modal":
        s, back, top = bracket()
        win.project.fixtures = [Fixture(FixtureKind.FIXED, [back])]
        win.project.analysis = AnalysisKind.MODAL
        win.project.operating_frequency = 50.0
        win._go(mw.STEP_SOLVE)
        win.panels[mw.STEP_SOLVE].refresh()
        win.solve()

        def modal():
            v = win.panels[mw.STEP_RESULT]
            v.animate.setChecked(False)
            v.mode.setCurrentIndex(1)

        after_result(modal)
    elif base == "07_thermal":
        s, back, top = bracket("al6061t6")
        win.project.loads = [
            Load(LoadKind.HEAT_FLOW, [top], magnitude=15.0),
            Load(LoadKind.CONVECTION, [back], magnitude=50.0),
            Load(
                LoadKind.RADIATION,
                [FaceRef(0, face_at(s, (50, 30, 16)))],
                magnitude=0.9,
                ambient=20.0,
            ),
        ]
        win.project.analysis = AnalysisKind.THERMAL
        win._go(mw.STEP_SOLVE)
        win.panels[mw.STEP_SOLVE].refresh()
        win.solve()
        after_result(lambda: None)
    elif base == "08_nonlinear":
        s, back, top = bracket()
        win.project.fixtures = [Fixture(FixtureKind.FIXED, [back])]
        win.project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, -9000.0))]
        win.project.plasticity = True
        win._go(mw.STEP_SOLVE)
        win.panels[mw.STEP_SOLVE].refresh()
        win.solve()
        after_result(lambda: field("peeq"))
    QTimer.singleShot(600_000, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
