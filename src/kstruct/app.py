from __future__ import annotations

import sys
from pathlib import Path


def selftest() -> int:
    """Headless end-to-end check (used to verify packaged builds): mesh and solve the
    sample bracket and compare against a reference value."""
    from kstruct import standards
    from kstruct.fem import generate_mesh, solve_static
    from kstruct.geometry.face_info import face_at
    from kstruct.geometry.samples import l_bracket
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Part, Project

    shape = l_bracket()
    project = Project(parts=[Part(0, "bracket", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -2000.0))
    ]
    mesh = generate_mesh(project, "coarse")
    result = solve_static(project, mesh, {0: standards.materials()["s235"]})
    peak = float(result.von_mises.max())
    ok = 80 < peak < 200 and abs(result.reactions[0][2] - 2000) < 1e-3
    print(f"selftest: elements={len(mesh.tets)} peak_vm={peak:.1f} MPa -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def selftest_gui(screenshot: Path) -> int:
    """Open the main window, solve the sample bracket through the UI and save a screenshot."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from kstruct.geometry.face_info import face_at
    from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind
    from kstruct.ui.main_window import STEP_SOLVE, MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 860)
    window.show()
    window.open_sample("bracket")
    shape = window.project.parts[0].shape
    window.project.parts[0].material_id = "s235"
    window.project.fixtures = [
        Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])
    ]
    window.project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, -2000.0))
    ]
    window._go(STEP_SOLVE)
    window.solve()
    status = {"code": 1}

    def poll():
        if window.result is None:
            QTimer.singleShot(200, poll)
            return
        QTimer.singleShot(800, finish)  # let the result panel lay out first

    def finish():
        window.grab().save(str(screenshot))
        status["code"] = 0
        print(f"selftest-gui: {window.summary.headline!a} -> {screenshot}")
        app.quit()

    QTimer.singleShot(200, poll)
    QTimer.singleShot(180_000, app.quit)  # never hang a CI job
    app.exec()
    return status["code"]


def main() -> None:
    import multiprocessing

    # meshing runs in spawned worker processes; a frozen app must handle their start-up
    multiprocessing.freeze_support()
    from kstruct.paths import install_excepthook, setup_logging

    setup_logging()
    install_excepthook()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--selftest-gui" in sys.argv:
        i = sys.argv.index("--selftest-gui")
        target = Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 else Path("kstruct_selftest.png")
        sys.exit(selftest_gui(target))

    from PySide6.QtWidgets import QApplication

    from kstruct.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("KStruct")
    window = MainWindow()
    window.install_crash_dialog()
    window.show()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        window.load(Path(args[0]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
