"""User-facing language: Korean (the source text) or English.

Every user-facing string is written in Korean in the code and passed through `tr`, which
returns it unchanged in Korean and looks it up in the English catalog otherwise:

    tr("해석 완료")                         -> "Analysis complete"
    LABELS = [N_("고정"), ...]  ...  tr(LABELS[i])   (constants made at import time)
    tr("부품 {name}: {fos:.2f}", name=n, fos=f)   (str.format placeholders, also in Korean)

The catalog is `resources/i18n/en/*.json` ({korean: english}, one file per area, merged).
Data files (materials, standards) keep Korean labels and are looked up the same way.

The language is chosen once at start-up (`set_language`, from prefs or KSTRUCT_LANG) and
changing it needs a restart: widgets are built with their texts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import cache
from importlib import resources

log = logging.getLogger(__name__)

LANGUAGES = {"ko": "한국어", "en": "English"}
DEFAULT = "ko"
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")

_language = (
    os.environ.get("KSTRUCT_LANG", "") if os.environ.get("KSTRUCT_LANG") in LANGUAGES else ""
)
_missing: set[str] = set()


def language() -> str:
    return _language or DEFAULT


def set_language(lang: str) -> None:
    """Called at start-up; KSTRUCT_LANG (tests, screenshots) wins over the saved choice."""
    global _language
    env = os.environ.get("KSTRUCT_LANG", "")
    _language = env if env in LANGUAGES else (lang if lang in LANGUAGES else DEFAULT)


def is_korean() -> bool:
    return language() == "ko"


@cache
def catalog(lang: str = "en") -> dict[str, str]:
    """{korean source: translation} merged from resources/i18n/<lang>/*.json."""
    folder = resources.files("kstruct.resources.i18n").joinpath(lang)
    merged: dict[str, str] = {}
    if not folder.is_dir():
        return merged
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            data = json.loads(entry.read_text("utf-8"))
            merged.update({k: v for k, v in data.items() if not k.startswith("_")})
    return merged


def tr(text: str, /, **values) -> str:
    """Translate `text` (Korean source) and fill its {placeholders} from `values`.

    Without `values` the text is returned as is (no .format), so literal braces are safe;
    with `values`, literal braces must be doubled as in str.format.
    """
    lang = language()
    if lang != "ko" and text:
        translated = catalog(lang).get(text)
        if translated is None:
            if text not in _missing:  # keep going in Korean rather than fail
                _missing.add(text)
                log.warning("no %s translation: %r", lang, text)
        else:
            text = translated
    return text.format(**values) if values else text


def N_(text: str) -> str:  # noqa: N802 (gettext convention)
    """Mark a Korean text for translation without translating it now: for module-level
    constants (built before the language is set). Show them through tr(value)."""
    return text


def tr_data(text: str) -> str:
    """Translate a label from data (built-in material, standards pack) if the catalog has
    it; user-entered names (custom materials, part names) stay as they are, silently."""
    lang = language()
    if lang == "ko" or not text:
        return text
    return catalog(lang).get(text, text)


def missing() -> set[str]:
    """Source texts that were shown without a translation in this session."""
    return set(_missing)
