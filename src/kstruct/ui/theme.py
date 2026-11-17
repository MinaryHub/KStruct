"""Look of the application: colour tokens (light and dark), the Qt style sheet built from
them, toolbar icons painted in code (no image assets to ship) and the step navigator.

Everything visual reads its colours from the active Theme, so a widget never hard-codes
one; `apply()` installs palette, font and style sheet on the QApplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleFactory,
)

from kstruct.standards import Verdict


@dataclass(frozen=True)
class Theme:
    name: str
    window: str  # behind everything
    surface: str  # panels, cards, inputs
    raised: str  # hovered rows, secondary buttons
    border: str
    text: str
    muted: str  # hints, secondary text
    accent: str
    accent_hover: str
    accent_text: str  # text on the accent colour
    accent_soft: str  # tinted background (current step, info card)
    ok: str
    ok_soft: str
    warn: str
    warn_soft: str
    fail: str
    fail_soft: str
    viewer_bottom: str
    viewer_top: str
    viewer_text: str  # scalar bar, edges in the 3D view

    def verdict(self, verdict: Verdict | None) -> tuple[str, str]:
        """(foreground, soft background) of a verdict card; None: informational."""
        return {
            Verdict.OK: (self.ok, self.ok_soft),
            Verdict.WARN: (self.warn, self.warn_soft),
            Verdict.FAIL: (self.fail, self.fail_soft),
        }.get(verdict, (self.accent, self.accent_soft))


LIGHT = Theme(
    name="light",
    window="#F3F5F8",
    surface="#FFFFFF",
    raised="#EEF1F5",
    border="#D5DAE2",
    text="#18212C",
    muted="#5F6B7A",
    accent="#1F5FD1",
    accent_hover="#184CAA",
    accent_text="#FFFFFF",
    accent_soft="#E6EEFB",
    ok="#177245",
    ok_soft="#E5F4EC",
    warn="#A35A00",
    warn_soft="#FDF1DC",
    fail="#B3261E",
    fail_soft="#FBE9E7",
    viewer_bottom="#F4F6F9",
    viewer_top="#DCE3EC",
    viewer_text="#263040",
)
DARK = Theme(
    name="dark",
    window="#14181E",
    surface="#1C222A",
    raised="#262E38",
    border="#36404D",
    text="#E6EAF0",
    muted="#98A3B3",
    accent="#5B9BFF",
    accent_hover="#7FB1FF",
    accent_text="#0B1220",
    accent_soft="#1D2F4D",
    ok="#4CC38A",
    ok_soft="#163326",
    warn="#F0B255",
    warn_soft="#3A2C12",
    fail="#FF7A6E",
    fail_soft="#3D1B18",
    viewer_bottom="#1B2027",
    viewer_top="#2B333F",
    viewer_text="#DDE3EB",
)
THEMES = {"light": LIGHT, "dark": DARK}
_current = LIGHT


def current() -> Theme:
    return _current


def _check_image(t: Theme) -> str:
    """Path of the tick drawn inside a checked box (style sheets take images by file)."""
    from kstruct.paths import data_dir

    path = data_dir() / f"check_{t.name}.png"
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(t.accent_text), 4.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    tick = QPainterPath(QPointF(7, 17))
    tick.lineTo(13.5, 23)
    tick.lineTo(25, 9.5)
    painter.drawPath(tick)
    painter.end()
    pixmap.save(str(path))
    return path.as_posix()


def _stylesheet(t: Theme) -> str:
    check = _check_image(t)
    return f"""
QMainWindow, QDialog {{ background: {t.window}; }}
QToolTip {{
    background: {t.text}; color: {t.surface}; border: 0; padding: 6px 8px; border-radius: 4px;
}}

/* header bar */
QToolBar {{
    background: {t.surface}; border: 0; border-bottom: 1px solid {t.border};
    padding: 6px 10px; spacing: 2px;
}}
QToolBar::separator {{ background: {t.border}; width: 1px; margin: 6px 8px; }}
QToolBar QToolButton {{
    color: {t.text}; background: transparent; border: 0; border-radius: 6px; padding: 6px 10px;
}}
QToolBar QToolButton:hover {{ background: {t.raised}; }}
QToolBar QToolButton:pressed {{ background: {t.border}; }}
QToolBar QToolButton::menu-indicator {{ image: none; }}
QLabel#appName {{ font-size: 15pt; font-weight: 700; color: {t.text}; padding: 0 12px 0 2px; }}
QLabel#appName[accent="true"] {{ color: {t.accent}; }}

/* side panel */
QWidget#sidePanel {{ background: {t.surface}; border-right: 1px solid {t.border}; }}
QListWidget#stepNav {{ background: transparent; border: 0; outline: 0; }}
QWidget#stepFooter {{ border-top: 1px solid {t.border}; }}
QLabel#stepTitle {{ font-size: 14pt; font-weight: 700; color: {t.text}; }}
QLabel[role="hint"] {{ color: {t.muted}; }}
QLabel[role="section"] {{ font-weight: 700; color: {t.text}; padding-top: 6px; }}
QLabel[role="disclaimer"] {{ color: {t.warn}; font-size: 9pt; }}
QScrollArea {{ background: transparent; border: 0; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* buttons */
QPushButton {{
    background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
    border-radius: 6px; padding: 6px 14px; min-height: 20px;
}}
QPushButton:hover {{ background: {t.raised}; }}
QPushButton:pressed {{ background: {t.border}; }}
QPushButton:disabled {{ color: {t.muted}; background: {t.window}; border-color: {t.border}; }}
QPushButton:focus {{ border-color: {t.accent}; }}
QPushButton[primary="true"] {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent}; font-weight: 700;
}}
QPushButton[primary="true"]:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; }}
QPushButton[primary="true"]:disabled {{
    background: {t.raised}; color: {t.muted}; border-color: {t.border};
}}
QPushButton[compact="true"] {{ padding: 4px 8px; }}
QPushButton:flat {{
    border: 0; background: transparent; color: {t.accent}; text-align: left; padding: 4px 2px;
}}
QPushButton:flat:hover {{ text-decoration: underline; background: transparent; }}
QToolButton[role="help"] {{
    color: {t.muted}; background: transparent; border: 1px solid {t.border};
    border-radius: 10px; font-weight: 700; font-size: 9pt;
}}
QToolButton[role="help"]:hover {{ color: {t.accent}; border-color: {t.accent}; }}

/* inputs */
QLineEdit, QAbstractSpinBox, QComboBox {{ min-height: 26px; }}
QCheckBox, QRadioButton {{ spacing: 8px; min-height: 22px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1.5px solid {t.muted}; background: {t.surface};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{
    background: {t.accent}; border-color: {t.accent}; image: url("{check}");
}}
QRadioButton::indicator:checked {{
    width: 8px; height: 8px; border: 5px solid {t.accent}; background: {t.accent_text};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {t.border}; background: {t.raised};
}}
QListWidget {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 6px; padding: 2px;
}}
QListWidget::indicator {{
    width: 15px; height: 15px; border: 1.5px solid {t.muted}; border-radius: 4px;
    background: {t.surface};
}}
QListWidget::indicator:checked {{
    background: {t.accent}; border-color: {t.accent}; image: url("{check}");
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}
QGroupBox {{
    border: 1px solid {t.border}; border-radius: 8px; margin-top: 12px; padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.muted}; font-weight: 700;
}}
QProgressBar {{
    background: {t.raised}; border: 0; border-radius: 4px; min-height: 8px; max-height: 8px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 4px; }}

/* cards */
QFrame#verdictCard {{ border-radius: 8px; }}
QFrame#welcomeCard {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px;
}}
QLabel#welcomeTitle {{ font-size: 20pt; font-weight: 700; color: {t.text}; }}

/* chrome */
QStatusBar {{ background: {t.surface}; color: {t.muted}; border-top: 1px solid {t.border}; }}
QSplitter::handle {{ background: {t.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QMenu {{
    background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 4px;
}}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t.accent_soft}; }}
QMenu::item:disabled {{ color: {t.muted}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {t.muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t.border}; border-radius: 3px; min-width: 28px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def _palette(t: Theme) -> QPalette:
    p = QPalette()
    roles = QPalette.ColorRole
    for role, colour in (
        (roles.Window, t.window),
        (roles.WindowText, t.text),
        (roles.Base, t.surface),
        (roles.AlternateBase, t.raised),
        (roles.Text, t.text),
        (roles.Button, t.surface),
        (roles.ButtonText, t.text),
        (roles.BrightText, t.accent_text),
        (roles.Highlight, t.accent),
        (roles.HighlightedText, t.accent_text),
        (roles.Link, t.accent),
        (roles.PlaceholderText, t.muted),
        (roles.ToolTipBase, t.text),
        (roles.ToolTipText, t.surface),
        (roles.Mid, t.border),
        (roles.Dark, t.border),
        (roles.Light, t.surface),
    ):
        p.setColor(role, QColor(colour))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.Text, roles.WindowText, roles.ButtonText):
        p.setColor(disabled, role, QColor(t.muted))
    return p


_font_family: str | None = None


def _ui_font() -> QFont:
    """The bundled Noto Sans KR, so Hangul looks the same on every platform."""
    global _font_family
    if _font_family is None:
        path = str(resources.files("kstruct.resources").joinpath("fonts/NotoSansKR.ttf"))
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(path))
        _font_family = families[0] if families else ""
    font = QFont(_font_family) if _font_family else QFont()
    font.setPointSize(10)
    weighted(font, False)
    return font


def weighted(font: QFont, bold: bool) -> QFont:
    """Set the weight of the bundled font. It is a variable font whose default instance
    is Thin (100); the weight axis has to be set explicitly, QFont.setBold alone does
    not move it."""
    font.setVariableAxis(QFont.Tag("wght"), 700.0 if bold else 420.0)
    font.setBold(bold)
    return font


def apply(app: QApplication, name: str = "light") -> Theme:
    """Install the theme on the application and make it the current one."""
    global _current
    _current = THEMES.get(name, LIGHT)
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setFont(_ui_font())
    app.setPalette(_palette(_current))
    app.setStyleSheet(_stylesheet(_current))
    return _current


def mark(widget, **properties) -> None:
    """Set dynamic properties the style sheet selects on (primary="true", role="hint")
    and re-polish, so it also works on a widget that is already shown."""
    for key, value in properties.items():
        widget.setProperty(key, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


# --- icons ----------------------------------------------------------------------------


def _paths() -> dict[str, list]:
    """Line icons on a 20x20 grid: lists of polylines (closed when the last point equals
    the first) and ("circle", cx, cy, r) entries."""
    return {
        "new": [
            [(5, 2.5), (12, 2.5), (15.5, 6), (15.5, 17.5), (5, 17.5), (5, 2.5)],
            [(12, 2.5), (12, 6), (15.5, 6)],
            [(10.2, 9), (10.2, 14)],
            [(7.7, 11.5), (12.7, 11.5)],
        ],
        "open": [
            [(2.5, 15.5), (2.5, 4.5), (8, 4.5), (9.8, 6.5), (16.5, 6.5), (16.5, 8.5)],
            [(2.5, 15.5), (5, 8.5), (18, 8.5), (15.5, 15.5), (2.5, 15.5)],
        ],
        "project": [
            [(3, 6), (10, 2.5), (17, 6), (10, 9.5), (3, 6)],
            [(3, 6), (3, 14), (10, 17.5), (17, 14), (17, 6)],
            [(10, 9.5), (10, 17.5)],
        ],
        "save": [
            [(3, 3), (14, 3), (17, 6), (17, 17), (3, 17), (3, 3)],
            [(6, 3), (6, 7.5), (13, 7.5), (13, 3)],
            [(6, 17), (6, 11.5), (14, 11.5), (14, 17)],
        ],
        "recent": [("circle", 10, 10, 7.2), [(10, 5.8), (10, 10), (13, 12)]],
        "fit": [
            [(3, 7), (3, 3), (7, 3)],
            [(13, 3), (17, 3), (17, 7)],
            [(17, 13), (17, 17), (13, 17)],
            [(7, 17), (3, 17), (3, 13)],
            [(7.5, 7.5), (12.5, 7.5), (12.5, 12.5), (7.5, 12.5), (7.5, 7.5)],
        ],
        "help": [
            ("circle", 10, 10, 7.2),
            [(7.8, 8), (8.3, 6.6), (10, 6), (11.8, 6.7), (12.2, 8.3), (10, 10.2), (10, 11.6)],
            [(10, 13.8), (10, 14.2)],
        ],
        "report": [
            [(10, 3), (17.5, 16.5), (2.5, 16.5), (10, 3)],
            [(10, 8), (10, 12)],
            [(10, 14), (10, 14.4)],
        ],
        "info": [("circle", 10, 10, 7.2), [(10, 9.2), (10, 14)], [(10, 6.2), (10, 6.6)]],
        "theme": [
            ("circle", 10, 10, 3.6),
            [(10, 2), (10, 4)],
            [(10, 16), (10, 18)],
            [(2, 10), (4, 10)],
            [(16, 10), (18, 10)],
            [(4.4, 4.4), (5.8, 5.8)],
            [(14.2, 14.2), (15.6, 15.6)],
            [(4.4, 15.6), (5.8, 14.2)],
            [(14.2, 5.8), (15.6, 4.4)],
        ],
    }


def icon(name: str, colour: str | None = None, size: int = 20) -> QIcon:
    """A toolbar icon in the theme's text colour (2x pixmap for sharp HiDPI rendering)."""
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(colour or _current.text))
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.scale(size * scale / 20.0, size * scale / 20.0)
    for shape in _paths()[name]:
        if shape[0] == "circle":
            _, cx, cy, r = shape
            painter.drawEllipse(QPointF(cx, cy), r, r)
            continue
        path = QPainterPath(QPointF(*shape[0]))
        for point in shape[1:]:
            path.lineTo(QPointF(*point))
        painter.drawPath(path)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


# --- step navigator -------------------------------------------------------------------

STEP_DONE = Qt.ItemDataRole.UserRole + 1  # bool on each navigator item: step completed


class StepDelegate(QStyledItemDelegate):
    """Paints one step of the horizontal navigator: a numbered badge (filled for the
    current step, a tick for a completed one, grey when not reachable yet), the step's
    short name under it and a connector line to its neighbours."""

    HEIGHT = 62
    BADGE = 26

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 (Qt override)
        view = option.widget
        count = max(index.model().rowCount(), 1)
        width = view.viewport().width() // count if view is not None else 52
        return QSize(max(width, 40), self.HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        t = _current
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect)
        enabled = bool(index.flags() & Qt.ItemFlag.ItemIsEnabled)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        done = bool(index.data(STEP_DONE))
        row, last = index.row(), index.model().rowCount() - 1
        cx, cy, r = rect.center().x(), rect.top() + 8 + self.BADGE / 2, self.BADGE / 2

        # connectors: green up to the last completed step
        previous_done = row > 0 and bool(index.sibling(row - 1, 0).data(STEP_DONE))
        for present, x0, x1, green in (
            (row > 0, rect.left(), cx - r - 3, previous_done),
            (row < last, cx + r + 3, rect.right(), done),
        ):
            if present:
                painter.setPen(QPen(QColor(t.ok if green else t.border), 2))
                painter.drawLine(QPointF(x0, cy), QPointF(x1, cy))

        if selected:
            fill, ring, ink = t.accent, t.accent, t.accent_text
        elif done and enabled:
            fill, ring, ink = t.ok_soft, t.ok, t.ok
        else:
            fill, ring, ink = t.surface, t.border, t.muted
        if hovered and enabled and not selected:
            ring = t.accent
        badge = QRectF(cx - r, cy - r, self.BADGE, self.BADGE)
        painter.setBrush(QColor(fill))
        painter.setPen(QPen(QColor(ring), 1.6))
        painter.drawEllipse(badge)

        text = str(index.data(Qt.ItemDataRole.DisplayRole))
        number, _, title = text.partition(". ")
        if done and enabled and not selected:
            tick = QPainterPath(QPointF(cx - 5.2, cy + 0.4))
            tick.lineTo(cx - 1.4, cy + 4.2)
            tick.lineTo(cx + 5.4, cy - 3.8)
            pen = QPen(QColor(t.ok), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(tick)
        else:
            painter.setFont(weighted(QFont(option.font), True))
            painter.setPen(QColor(ink))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, number)

        font = weighted(QFont(option.font), selected)
        font.setPointSizeF(max(option.font.pointSizeF() - 1, 7))
        painter.setFont(font)
        colour = t.accent if selected else (t.text if enabled else t.muted)
        painter.setPen(QColor(colour))
        label = QRectF(
            rect.left(), badge.bottom() + 3, rect.width(), rect.bottom() - badge.bottom() - 4
        )
        short = (title or text).split()[0]  # "모델 열기" -> "모델": seven labels in a row
        painter.drawText(label, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, short)
        painter.restore()


class StepNav(QListWidget):
    """The seven steps side by side. A QListWidget, so the window keeps driving it by
    row (currentRow / setCurrentRow) as before."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("stepNav")
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setItemDelegate(StepDelegate(self))
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(StepDelegate.HEIGHT + 4)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.scheduleDelayedItemsLayout()  # cell width follows the panel width
