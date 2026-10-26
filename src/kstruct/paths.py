"""Per-user folders (logs, user material library) and the application log.

Everything the app writes outside the project file lives under one folder:
  Windows  %APPDATA%\\KStruct
  macOS    ~/Library/Application Support/KStruct
  Linux    $XDG_DATA_HOME/kstruct (default ~/.local/share/kstruct)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

APP_NAME = "KStruct"
LOG_NAME = "kstruct.log"


def data_dir() -> Path:
    override = os.environ.get("KSTRUCT_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "kstruct"
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_path() -> Path:
    return data_dir() / LOG_NAME


def user_materials_path() -> Path:
    return data_dir() / "materials.json"


def setup_logging() -> Path:
    """Rotating log file in the data folder (plus stderr when there is a console).
    Returns the log path. Safe to call more than once."""
    root = logging.getLogger()
    path = log_path()
    if any(getattr(h, "kstruct_file", False) for h in root.handlers):
        return path
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(fmt)
        handler.kstruct_file = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    except OSError:  # read-only home etc.: keep going without a file
        pass
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        root.addHandler(console)
    from kstruct import __version__

    logging.getLogger("kstruct").info(
        "KStruct %s, Python %s, %s %s, frozen=%s",
        __version__,
        platform.python_version(),
        platform.system(),
        platform.release(),
        getattr(sys, "frozen", False),
    )
    return path


def install_excepthook() -> None:
    """Log every unhandled exception; the UI adds a dialog on top of this."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        logging.getLogger("kstruct").critical("unhandled exception", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = hook


def environment_summary() -> str:
    from kstruct import __version__

    return (
        f"KStruct {__version__}\n"
        f"Python {platform.python_version()}\n"
        f"{platform.system()} {platform.release()} ({platform.machine()})\n"
        f"frozen={getattr(sys, 'frozen', False)}\n"
        f"data folder: {data_dir()}\n"
    )


def notices_text() -> str:
    """Third-party notices shipped with the app (repo root in development)."""
    candidates = [
        Path(getattr(sys, "_MEIPASS", "")) / "THIRD_PARTY_NOTICES.md",
        Path(sys.executable).parent / "THIRD_PARTY_NOTICES.md",
        Path(__file__).resolve().parents[2] / "THIRD_PARTY_NOTICES.md",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c.read_text("utf-8")
        except OSError:
            continue
    return "THIRD_PARTY_NOTICES.md 파일을 찾을 수 없습니다."
