"""Help system: every user-facing control references a key in resources/help/<lang>.json.

- hover  -> `short` text as a tooltip
- click "?" -> popup with `detail` (rich text) and links to `related` topics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kstruct.i18n import tr


@dataclass(frozen=True)
class HelpTopic:
    key: str
    title: str
    short: str
    detail: str
    related: tuple[str, ...] = field(default_factory=tuple)


@cache
def _topics(lang: str = "ko") -> dict[str, HelpTopic]:
    text = resources.files("kstruct.resources.help").joinpath(f"{lang}.json").read_text("utf-8")
    return {
        key: HelpTopic(key, v["title"], v["short"], v["detail"], tuple(v.get("related", ())))
        for key, v in json.loads(text).items()
    }


def topic(key: str) -> HelpTopic:
    from kstruct.i18n import language

    lang = language()
    if lang != "ko":
        try:
            return _topics(lang)[key]
        except (KeyError, FileNotFoundError):
            pass  # fall back to Korean
    try:
        return _topics()[key]
    except KeyError:
        # missing help text is a bug, but must not crash the app
        return HelpTopic(
            key,
            key,
            tr("(도움말 없음: {key})", key=key),
            tr("도움말이 아직 없습니다: {key}", key=key),
        )


class HelpDialog(QDialog):
    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        t = topic(key)
        self.setWindowTitle(t.title)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        title = QLabel(f"<h3>{t.title}</h3>")
        body = QLabel(t.detail)
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)

        for related in t.related:
            link = QPushButton(f"→ {topic(related).title}")
            link.setFlat(True)
            link.clicked.connect(lambda _=False, k=related: HelpDialog(k, self).exec())
            layout.addWidget(link)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr("닫기"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class HelpButton(QToolButton):
    """Small "?" button: tooltip on hover, detailed popup on click."""

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setText("?")
        self.setAutoRaise(True)
        self.setToolTip(topic(key).short)
        self.setFixedSize(20, 20)
        self.setProperty("role", "help")  # styled by ui/theme.py
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(lambda: HelpDialog(self.key, self.window()).exec())


def with_help(widget: QWidget, key: str) -> QWidget:
    """Wrap a control with its "?" button and set its tooltip from the same help key."""
    widget.setToolTip(topic(key).short)
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget, 1)
    row.addWidget(HelpButton(key))
    return box
