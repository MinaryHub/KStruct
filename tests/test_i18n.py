"""English UI: every Korean source text has a translation, nothing Korean is left
untranslated in the code, placeholders match, and help exists for every topic."""

import ast
import json
import string
from importlib import resources
from pathlib import Path

import pytest

from kstruct import i18n

SRC = Path(__file__).resolve().parents[1] / "src" / "kstruct"
HANGUL = i18n.HANGUL

# Korean literals that are not shown through tr(): the language names in the language
# menu and its bilingual restart notice (they must read the same in both languages).
ALLOWED = {
    "한국어",
    "언어 / Language",
    "KStruct를 다시 시작하면 바뀐 언어로 표시됩니다.\nRestart KStruct to switch the language.",
}
SKIP_FILES = {"i18n.py"}


def _is_docstring(node: ast.AST, parent: ast.AST | None) -> bool:
    return (
        isinstance(parent, ast.Expr)
        and isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    )


def _scan():
    """(tr literals with their file, untranslated Korean literals with file:line)."""
    used: dict[str, str] = {}
    bare: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        if py.name in SKIP_FILES or "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text("utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        translated: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("tr", "tr_data", "N_")
                and node.args
            ):
                arg = node.args[0]
                translated.add(id(arg))
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if node.func.id in ("tr", "N_"):
                        used.setdefault(arg.value, f"{py.relative_to(SRC)}:{node.lineno}")
        for node in ast.walk(tree):
            if id(node) in translated:
                continue
            parent = parents.get(node)
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _is_docstring(node, parent) or isinstance(parent, ast.JoinedStr):
                    continue
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                if id(node) in translated:
                    continue
                text = "".join(
                    v.value
                    for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
            if text and HANGUL.search(text) and text not in ALLOWED:
                # logger messages are for developers, not users
                call = parent
                while call is not None and not isinstance(call, ast.Call):
                    call = parents.get(call)
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "log"
                ):
                    continue
                bare.append(f"{py.relative_to(SRC)}:{node.lineno}: {text[:60]!r}")
    return used, bare


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}


def test_no_korean_text_outside_tr():
    _, bare = _scan()
    assert not bare, "Korean text not passed through tr():\n" + "\n".join(bare)


def test_every_tr_text_has_an_english_translation():
    used, _ = _scan()
    assert used, "no tr() calls found"
    en = i18n.catalog("en")
    missing = [f"{where}: {text!r}" for text, where in used.items() if text not in en]
    assert not missing, "missing English:\n" + "\n".join(sorted(missing))


def test_translations_keep_placeholders_and_are_english():
    for source, english in i18n.catalog("en").items():
        assert not HANGUL.search(english), f"Korean left in translation of {source!r}"
        assert _fields(source) == _fields(english), f"placeholders differ: {source!r}"


def test_english_help_covers_every_topic():
    folder = resources.files("kstruct.resources.help")
    ko = json.loads(folder.joinpath("ko.json").read_text("utf-8"))
    en = json.loads(folder.joinpath("en.json").read_text("utf-8"))
    assert en.keys() == ko.keys()
    for key, topic in en.items():
        assert topic.get("related", []) == ko[key].get("related", []), key
        for field in ("title", "short", "detail"):
            assert topic[field].strip(), (key, field)
            assert not HANGUL.search(topic[field]), (key, field)


@pytest.fixture
def english(monkeypatch):
    monkeypatch.setenv("KSTRUCT_LANG", "en")
    i18n.set_language("en")
    yield
    monkeypatch.delenv("KSTRUCT_LANG")
    i18n.set_language("ko")


def test_tr_formats_and_falls_back(english):
    assert i18n.tr("해석 중…") != "해석 중…"
    assert i18n.tr_data("사용자가 입력한 이름") == "사용자가 입력한 이름"


def test_korean_is_unchanged_by_tr():
    i18n.set_language("ko")
    assert i18n.tr("부품 {n}개", n=3) == "부품 3개"
    assert i18n.tr("{ 그대로 }") == "{ 그대로 }"  # no values: no formatting
