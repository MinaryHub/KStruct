from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from kstruct.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("KStruct")
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
