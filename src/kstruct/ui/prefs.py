"""Per-user preferences (recent files, last folder, window layout) as a small JSON file
in the data folder, plus the autosave used for crash recovery."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from kstruct.paths import data_dir

log = logging.getLogger(__name__)
MAX_RECENT = 8
AUTOSAVE_MINUTES = 3


def prefs_path() -> Path:
    return data_dir() / "prefs.json"


def autosave_path() -> Path:
    return data_dir() / "autosave.kst"


def load_prefs() -> dict:
    path = prefs_path()
    try:
        return json.loads(path.read_text("utf-8")) if path.is_file() else {}
    except (OSError, ValueError) as e:
        log.warning("prefs unreadable (%s): %s", path, e)
        return {}


def save_prefs(prefs: dict) -> None:
    try:
        prefs_path().write_text(json.dumps(prefs, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        log.warning("cannot write prefs: %s", e)


def recent_files() -> list[Path]:
    return [Path(p) for p in load_prefs().get("recent", []) if Path(p).is_file()]


def remember_file(path: Path) -> None:
    prefs = load_prefs()
    items = [str(path)] + [p for p in prefs.get("recent", []) if p != str(path)]
    prefs["recent"] = items[:MAX_RECENT]
    prefs["last_dir"] = str(path.parent)
    save_prefs(prefs)


def last_dir() -> str:
    d = load_prefs().get("last_dir", "")
    return d if d and Path(d).is_dir() else ""


def set_window_state(geometry: bytes, splitter: list[int]) -> None:
    prefs = load_prefs()
    prefs["geometry"] = geometry.hex()
    prefs["splitter"] = list(splitter)
    save_prefs(prefs)


def window_state() -> tuple[bytes | None, list[int] | None]:
    prefs = load_prefs()
    geometry = prefs.get("geometry")
    try:
        return (bytes.fromhex(geometry) if geometry else None), prefs.get("splitter")
    except ValueError:
        return None, prefs.get("splitter")


def theme() -> str:
    """ "light" or "dark" (the look the user chose last)."""
    name = load_prefs().get("theme", "light")
    return name if name in ("light", "dark") else "light"


def set_theme(name: str) -> None:
    prefs = load_prefs()
    prefs["theme"] = name
    save_prefs(prefs)


# --- autosave ---------------------------------------------------------------------------


def write_autosave(project, source: Path | None) -> Path:
    from kstruct.project_io import save_project

    path = save_project(project, autosave_path())
    meta = {"source": str(source) if source else None, "time": dt.datetime.now().isoformat()}
    autosave_path().with_suffix(".json").write_text(json.dumps(meta), "utf-8")
    return path


def pending_autosave() -> dict | None:
    """{"path", "source", "time"} when an autosave from an unclean exit is present."""
    path = autosave_path()
    if not path.is_file():
        return None
    meta_path = path.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text("utf-8")) if meta_path.is_file() else {}
    except (OSError, ValueError):
        meta = {}
    return {"path": path, "source": meta.get("source"), "time": meta.get("time", "")}


def clear_autosave() -> None:
    for p in (autosave_path(), autosave_path().with_suffix(".json")):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("cannot remove autosave %s: %s", p, e)
