"""Main window: wizard (left), 3D viewer (center). Owns the Project, mesh and result."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kstruct import i18n, standards
from kstruct.fem import ConstraintError, MeshingError, SolverError, generate_mesh
from kstruct.fem.buckling import BucklingResult, solve_buckling
from kstruct.fem.mesh import FEMesh
from kstruct.fem.modal import ModalResult, solve_modal
from kstruct.fem.nonlinear import unsupported as nonlinear_unsupported
from kstruct.fem.static import StaticResult, solve_cases
from kstruct.fem.thermal import ThermalResult, solve_thermal
from kstruct.geometry import GeometryImportError, load_cad
from kstruct.geometry.contacts import find_touching_faces
from kstruct.geometry.samples import bolted_plates, l_bracket, two_plates, write_step
from kstruct.i18n import N_, tr, tr_data
from kstruct.model import (
    TEMPERATURE_REFERENCES,
    THERMAL_LOADS,
    AnalysisKind,
    Project,
    active_project,
    excluded_by_hidden_parts,
    with_case,
)
from kstruct.project_io import (
    SUFFIX,
    ProjectFileError,
    from_document,
    load_project,
    load_project_mesh,
    save_project,
    to_document,
)
from kstruct.report.summary import (
    VERDICT_STYLE,
    CaseOverview,
    Summary,
    compare_cases,
    format_duration,
    summarize,
    summarize_buckling,
    summarize_modal,
    summarize_thermal,
)
from kstruct.ui import prefs, steps, theme
from kstruct.ui.dialogs import AboutDialog, save_bug_report
from kstruct.ui.help import HelpDialog, topic
from kstruct.ui.updates import UpdateChecker, UpdateDialog, open_contact, open_sponsor
from kstruct.ui.viewer import ModelViewer, ResultField
from kstruct.ui.worker import Cancelled, JobHandle, run_job
from kstruct.units import fmt, quantity, system_of

CAD_FILTER = N_(
    "모든 3D 파일 (*.step *.stp *.iges *.igs *.stl *.obj *.fbx *.gltf *.glb *.3mf *.kst);;"
    "CAD 파일 (*.step *.stp *.iges *.igs);;"
    "메시 파일 (*.stl *.obj *.fbx *.gltf *.glb *.3mf);;"
    "모든 파일 (*)"
)
PROJECT_FILTER = N_("KStruct 프로젝트 (*{suffix})")  # tr(PROJECT_FILTER, suffix=SUFFIX)
MAX_UNDO = 50  # steps kept in the undo history
SAMPLES = {"bracket": l_bracket, "two_plates": two_plates, "bolted": bolted_plates}
SAMPLE_PART_NAMES = {
    "bracket": [N_("브래킷")],
    "two_plates": [N_("아래 판"), N_("위 판")],
    "bolted": [N_("아래 판"), N_("위 판")],
}

log = logging.getLogger("kstruct.ui")
STEP_OPEN, STEP_MATERIAL, STEP_FIXTURE, STEP_LOAD, STEP_MESH, STEP_SOLVE, STEP_RESULT = range(7)
STEP_NAMES = [
    N_("1. 모델 열기"),
    N_("2. 재료"),
    N_("3. 고정"),
    N_("4. 하중"),
    N_("5. 메시"),
    N_("6. 해석"),
    N_("7. 결과"),
]
THERMAL_FIELDS = [("temp", N_("온도")), ("flux", N_("열 흐름 (열유속)"))]
STEP_KEYS = [
    "step.open",
    "step.material",
    "step.fixture",
    "step.load",
    "step.mesh",
    "step.solve",
    "step.result",
]


def friendly_error(e: Exception) -> str:
    if isinstance(e, SolverError):
        if str(e) == "unstable":
            return tr(
                "부품이 충분히 고정되지 않아 공중에서 움직이거나 돌 수 있습니다.\n\n"
                "• 고정을 추가하세요 (예: 볼트 구멍이나 바닥면을 '완전 고정').\n"
                "• 여러 부품이면 각 부품이 고정되어 있거나, 다른 부품과 '접합'되어 있어야 합니다."
            )
        if str(e) == "bolt without contact":
            return tr(
                "볼트로 조인 두 부품 사이에 맞닿은 면(접촉 또는 접합)이 없습니다. 체결력은 부품끼리 "
                "서로 눌러 줄 때만 생기므로, 1단계에서 '맞닿은 면 자동으로 찾기'를 누르고 연결 종류를 "
                "'접촉'으로 지정하세요. (체결력을 0으로 하면 연결 없이도 계산할 수 있습니다.)"
            )
        if str(e) == "contact unstable":
            return tr(
                "접촉으로만 연결된 부품이 미끄러지거나 떨어져 나가서 붙잡혀 있지 않습니다.\n\n"
                "• 그 부품을 직접 고정하거나,\n"
                "• 실제로 볼트·용접으로 붙어 있다면 1단계에서 연결 종류를 '접합'으로 바꾸세요.\n"
                "• 접촉은 마찰이 없다고 가정하므로 옆으로 미는 힘은 버티지 못합니다."
            )
        if str(e) == "not converged":
            return tr(
                "계산이 수렴하지 않았습니다. 메시를 '보통'으로 바꾸거나 고정 조건을 확인하세요."
            )
        return str(e)
    if isinstance(e, (MeshingError, ConstraintError, GeometryImportError, ProjectFileError)):
        return str(e)
    return tr("예상하지 못한 오류가 발생했습니다:\n{kind}: {error}", kind=type(e).__name__, error=e)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        i18n.set_language(prefs.language())  # before any widget text is made
        self.setWindowTitle(tr("KStruct — 부품 강도 해석"))
        self.resize(1440, 900)
        self.project: Project | None = None
        self.project_path: Path | None = None
        self.mesh: FEMesh | None = None
        self.result: StaticResult | ModalResult | ThermalResult | None = None
        self.summary: Summary | None = None
        # strength runs: one result/summary per load case; result/summary follow the
        # case chosen in the result panel
        self.case_results: list[tuple[str, StaticResult, Summary]] | None = None
        self.overview: CaseOverview | None = None
        # key numbers of the previous strength run, for the "compared with last time" line
        self.previous: dict | None = None
        self._solved_quality: str | None = None  # mesh preset the current result used
        self._dirty = False  # unsaved changes (drives autosave and the close prompt)
        self.ask_before_close = True  # tests switch the "save changes?" prompt off
        self._sample_files: set[Path] = set()
        self.materials: dict = {}
        self._busy = False
        self._job: JobHandle | None = None  # the running mesh / solve job
        # undo history: project documents (project_io.to_document as JSON text; the part
        # shapes never change and stay with the parts). _baseline is the current state.
        self._history: list[str] = []
        self._future: list[str] = []
        self._baseline: str | None = None
        self._restoring = False

        app = QApplication.instance()
        self.theme = theme.apply(app, prefs.theme()) if app is not None else theme.current()
        self.viewer = ModelViewer()
        self.viewer.apply_theme(self.theme)
        self.viewer.selection_changed.connect(self._selection_changed)
        self.viewer.probed.connect(lambda text: self.panels[STEP_RESULT].probe.setText(text))

        self.panels: list[steps.StepPanel] = [
            steps.OpenStep(self),
            steps.MaterialStep(self),
            steps.FixtureStep(self),
            steps.LoadStep(self),
            steps.MeshStep(self),
            steps.SolveStep(self),
            steps.ResultStep(self),
        ]
        for panel in self.panels:
            panel.changed.connect(self._project_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_side())
        # right pane: a welcome page until a model is open, then the 3D view
        self.pane = QStackedWidget()
        self.pane.addWidget(self._build_welcome())
        self.pane.addWidget(self.viewer)
        splitter.addWidget(self.pane)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1040])
        splitter.setHandleWidth(1)
        self.setCentralWidget(splitter)
        self._build_toolbar()
        self._restore_window_state(splitter)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(prefs.AUTOSAVE_MINUTES * 60 * 1000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        self._go(STEP_OPEN)
        self.statusBar().showMessage(tr("'파일 열기'로 STEP 파일을 불러오거나 예제를 눌러 보세요."))

    # --- layout ----------------------------------------------------------------------

    def _build_side(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidePanel")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        side.setMinimumWidth(340)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(0)

        # the seven steps in a row: number badge, tick when done (theme.StepNav)
        self.nav = theme.StepNav()
        for name, key in zip(STEP_NAMES, STEP_KEYS, strict=True):
            item = QListWidgetItem(tr(name))
            item.setToolTip(f"{tr(name)} — {topic(key).short}")
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self._nav_changed)
        nav_box = QWidget()
        nav_layout = QHBoxLayout(nav_box)
        nav_layout.setContentsMargins(6, 0, 6, 4)
        nav_layout.addWidget(self.nav)
        layout.addWidget(nav_box)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet(f"color: {self.theme.border};")
        self._rule = rule
        layout.addWidget(rule)

        self.stack = QStackedWidget()
        for panel in self.panels:
            panel.layout_.setContentsMargins(16, 14, 12, 12)
            panel.layout_.setSpacing(8)
            # long option texts must not push the panel wider than the side bar
            for combo in panel.findChildren(QComboBox):
                combo.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                combo.setMinimumContentsLength(6)
            scroll = QScrollArea()
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(panel)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self.stack.addWidget(scroll)
        layout.addWidget(self.stack, 1)

        footer = QWidget()
        footer.setObjectName("stepFooter")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        buttons = QHBoxLayout(footer)
        buttons.setContentsMargins(16, 10, 16, 10)
        self.prev_btn = QPushButton(tr("이전"))
        self.next_btn = QPushButton(tr("다음 단계"))
        theme.mark(self.next_btn, primary="true")
        self.prev_btn.clicked.connect(lambda: self._go(self.nav.currentRow() - 1))
        self.next_btn.clicked.connect(lambda: self._go(self.nav.currentRow() + 1))
        buttons.addWidget(self.prev_btn)
        buttons.addWidget(self.next_btn, 1)
        layout.addWidget(footer)
        return side

    def _build_welcome(self) -> QWidget:
        """Shown instead of an empty 3D view: what the program does and how to start."""
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card = QFrame()
        card.setObjectName("welcomeCard")
        card.setFixedWidth(560)
        box = QVBoxLayout(card)
        box.setContentsMargins(36, 32, 36, 28)
        box.setSpacing(10)
        title = QLabel(tr("부품이 견디는지 확인해 보세요"))
        title.setObjectName("welcomeTitle")
        box.addWidget(title)
        intro = QLabel(
            tr(
                "CAD에서 만든 부품을 열고 재료·고정·하중을 고르면, 안전한지와 얼마나 휘는지를 "
                "색과 쉬운 말로 알려 드립니다. 위쪽의 7단계를 차례로 따라가면 됩니다."
            )
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "hint")
        box.addWidget(intro)
        box.addSpacing(8)
        row = QHBoxLayout()
        open_model = QPushButton(tr("모델 열기…"))
        theme.mark(open_model, primary="true")
        open_model.setMinimumHeight(36)
        open_model.clicked.connect(self.open_file)
        open_project = QPushButton(tr("프로젝트 열기…"))
        open_project.setMinimumHeight(36)
        open_project.clicked.connect(self.open_project)
        row.addWidget(open_model, 1)
        row.addWidget(open_project, 1)
        box.addLayout(row)
        formats = QLabel(tr("STEP · IGES · STL · OBJ · FBX · glTF · 3MF 파일을 열 수 있습니다."))
        formats.setProperty("role", "hint")
        box.addWidget(formats)

        box.addSpacing(10)
        section = QLabel(tr("예제로 시작하기"))
        section.setProperty("role", "section")
        box.addWidget(section)
        samples = QHBoxLayout()
        for label, name, tip in (
            (tr("브래킷"), "bracket", tr("부품 하나의 강도 확인")),
            (tr("판 2장"), "two_plates", tr("여러 부품: 접합과 접촉")),
            (tr("볼트 체결"), "bolted", tr("볼트 체결력과 안전율")),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.open_sample(n))
            samples.addWidget(b)
        box.addLayout(samples)

        self._welcome_recent_title = QLabel(tr("최근 파일"))
        self._welcome_recent_title.setProperty("role", "section")
        box.addWidget(self._welcome_recent_title)
        self._welcome_recent = QVBoxLayout()
        self._welcome_recent.setSpacing(0)
        box.addLayout(self._welcome_recent)
        outer.addWidget(card)
        self._fill_welcome_recent()
        return page

    def _fill_welcome_recent(self) -> None:
        while self._welcome_recent.count():
            item = self._welcome_recent.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        files = prefs.recent_files()[:5]
        self._welcome_recent_title.setVisible(bool(files))
        for path in files:
            link = QPushButton(path.name)
            link.setFlat(True)
            link.setToolTip(str(path))
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(lambda _=False, p=path: self.load(p))
            self._welcome_recent.addWidget(link)

    def _show_pane(self) -> None:
        """Welcome page without a project, the 3D view with one."""
        if self.project is None:
            self._fill_welcome_recent()
        self.pane.setCurrentIndex(0 if self.project is None else 1)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar(tr("기본"))
        bar.setMovable(False)
        bar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(18, 18))
        name = QLabel("KStruct")
        name.setObjectName("appName")
        bar.addWidget(name)
        self._icons: list[tuple[object, str]] = []  # (action or button, icon name)

        def action(text, slot, key=None, shortcut=None, icon=None, icon_only=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            if key:
                a.setToolTip(topic(key).short)
            if icon:
                a.setIcon(theme.icon(icon))
                self._icons.append((a, icon))
            a.triggered.connect(slot)
            bar.addAction(a)
            if icon_only:
                bar.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            return a

        action(tr("새로 시작"), self.new_project, "project.new", icon="new")
        action(tr("모델 열기"), self.open_file, "step.open", QKeySequence.StandardKey.Open, "open")
        action(tr("프로젝트 열기"), self.open_project, "project.open", icon="project")
        action(tr("저장"), self.save_project, "project.save", QKeySequence.StandardKey.Save, "save")
        self.undo_action = action(
            tr("실행 취소"), self.undo, "project.undo", QKeySequence.StandardKey.Undo, "undo", True
        )
        self.redo_action = action(
            tr("다시 실행"), self.redo, "project.undo", QKeySequence.StandardKey.Redo, "redo", True
        )
        self._update_undo_actions()
        self.recent_button = QToolButton()
        self.recent_button.setText(tr("최근 파일"))
        self.recent_button.setIcon(theme.icon("recent"))
        self.recent_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._icons.append((self.recent_button, "recent"))
        self.recent_button.setToolTip(topic("project.recent").short)
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_menu = QMenu(self.recent_button)
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.recent_button.setMenu(self.recent_menu)
        bar.addWidget(self.recent_button)
        bar.addSeparator()
        action(tr("화면 맞춤"), self.viewer.fit, "view.fit", "F", "fit")

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        # shown only when the start-up check found a newer release
        self.update_action = action(tr("업데이트"), self.show_update, "help.update", icon="update")
        self.update_action.setVisible(False)
        self._pending_release = None
        action(
            tr("사용 안내"),
            lambda: HelpDialog("help.mode", self).exec(),
            "help.mode",
            "F1",
            "help",
            True,
        )
        action(
            tr("문의하기"),
            lambda: open_contact(self),
            "help.contact",
            icon="contact",
            icon_only=True,
        )
        action(tr("문제 보고"), self.report_problem, "help.report", icon="report", icon_only=True)
        action(
            tr("후원하기"), lambda: open_sponsor(self), "help.sponsor", icon="heart", icon_only=True
        )
        action(tr("화면 색"), self.toggle_theme, "view.theme", icon="theme", icon_only=True)
        self.info_button = QToolButton()
        self.info_button.setIcon(theme.icon("info"))
        self._icons.append((self.info_button, "info"))
        self.info_button.setToolTip(topic("help.about").short)
        self.info_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        info_menu = QMenu(self.info_button)
        info_menu.addAction(tr("KStruct 정보"), lambda: AboutDialog(self).exec())
        info_menu.addAction(tr("사용 안내"), lambda: HelpDialog("help.mode", self).exec())
        info_menu.addAction(tr("업데이트 확인…"), self.show_update)
        info_menu.addSeparator()
        info_menu.addAction(tr("문의하기 (GitHub)"), lambda: open_contact(self))
        info_menu.addAction(tr("문제 보고 (로그·프로젝트 묶기)"), self.report_problem)
        info_menu.addAction(tr("후원하기 (GitHub Sponsors)"), lambda: open_sponsor(self))
        info_menu.addSeparator()
        language_menu = info_menu.addMenu("언어 / Language")
        for code, name in i18n.LANGUAGES.items():
            item = language_menu.addAction(name, lambda c=code: self.choose_language(c))
            item.setCheckable(True)
            item.setChecked(code == prefs.language())
        self.info_button.setMenu(info_menu)
        bar.addWidget(self.info_button)

    def choose_language(self, lang: str) -> None:
        """Saved now, applied at the next start (widgets are built with their texts)."""
        if lang == prefs.language():
            return
        prefs.set_language(lang)
        QMessageBox.information(
            self,
            "언어 / Language",
            "KStruct를 다시 시작하면 바뀐 언어로 표시됩니다.\n"
            "Restart KStruct to switch the language.",
        )

    # --- updates ---------------------------------------------------------------------

    def check_updates_quietly(self) -> None:
        """Start-up check (once a day, can be turned off in the update dialog)."""
        self._update_checker = UpdateChecker(self)
        self._update_checker.available.connect(self._update_available)
        self._update_checker.start()

    def _update_available(self, release) -> None:
        self._pending_release = release
        self.update_action.setText(tr("업데이트 {version}", version=release.version))
        self.update_action.setVisible(True)
        self.statusBar().showMessage(
            tr(
                "새 버전 {version}이 있습니다. 툴바의 '업데이트'를 누르세요.",
                version=release.version,
            ),
            15000,
        )

    def show_update(self) -> None:
        UpdateDialog(self, self._pending_release).exec()

    def toggle_theme(self) -> None:
        """Switch between the light and the dark look and remember the choice."""
        name = "dark" if self.theme.name == "light" else "light"
        prefs.set_theme(name)
        self.theme = theme.apply(QApplication.instance(), name)
        for holder, icon_name in self._icons:
            holder.setIcon(theme.icon(icon_name))
        self._rule.setStyleSheet(f"color: {self.theme.border};")
        self.viewer.apply_theme(self.theme)
        self.nav.viewport().update()
        if self.project is not None:
            self.panels[self.nav.currentRow()].refresh()
        if self.result is not None and self.nav.currentRow() == STEP_RESULT:
            self.show_result()

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        files = prefs.recent_files()
        if not files:
            self.recent_menu.addAction(tr("(최근에 연 파일이 없습니다)")).setEnabled(False)
        for path in files:
            self.recent_menu.addAction(path.name, lambda p=path: self.load(p)).setToolTip(str(path))

    def _restore_window_state(self, splitter: QSplitter) -> None:
        geometry, sizes = prefs.window_state()
        if geometry:
            self.restoreGeometry(geometry)
        if sizes and len(sizes) == 2:
            splitter.setSizes(sizes)
        self._splitter = splitter

    def _save_window_state(self) -> None:
        prefs.set_window_state(bytes(self.saveGeometry().data()), self._splitter.sizes())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._job is not None:  # don't leave a solve running behind a closed window
            self._job.cancel()
        if self.project is not None and self._dirty and self.ask_before_close:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(tr("저장하지 않은 변경"))
            box.setText(tr("저장하지 않은 변경이 있습니다. 프로젝트를 저장할까요?"))
            save = box.addButton(tr("저장"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("저장 안 함"), QMessageBox.ButtonRole.DestructiveRole)
            cancel = box.addButton(tr("취소"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is cancel:
                event.ignore()
                return
            if box.clickedButton() is save:
                self.save_project()
                if self._dirty:  # the save dialog was cancelled
                    event.ignore()
                    return
        self._autosave_timer.stop()
        prefs.clear_autosave()
        self._save_window_state()
        self.viewer.shutdown()
        super().closeEvent(event)

    # --- autosave / recovery ----------------------------------------------------------

    def _autosave(self) -> None:
        if self.project is None or not self._dirty or self._busy:
            return
        try:
            prefs.write_autosave(self.project, self.project_path or self.project.source_path)
            log.info("autosaved")
        except Exception:  # never let the timer kill the app
            log.exception("autosave failed")

    def offer_recovery(self) -> bool:
        """At start-up: load the autosave left by an unclean exit if the user wants it."""
        pending = prefs.pending_autosave()
        if pending is None:
            return False
        source = pending["source"] or tr("이름 없는 프로젝트")
        when = pending["time"][:16].replace("T", " ")
        answer = QMessageBox.question(
            self,
            tr("작업 복구"),
            tr(
                "지난번에 저장하지 않은 작업이 있습니다({when}, {source}).\n복구할까요?",
                when=when,
                source=source,
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            prefs.clear_autosave()
            return False
        return self.recover()

    def recover(self) -> bool:
        pending = prefs.pending_autosave()
        if pending is None:
            return False
        try:
            project = load_project(pending["path"])
        except (ProjectFileError, KeyError, ValueError) as e:
            self.warn(tr("복구 파일을 읽을 수 없습니다: {error}", error=e))
            prefs.clear_autosave()
            return False
        source = Path(pending["source"]) if pending["source"] else None
        self._set_project(project, source if source and source.suffix.lower() == SUFFIX else None)
        if source and source.suffix.lower() != SUFFIX:
            project.source_path = source
        self._dirty = True
        self.statusBar().showMessage(
            tr("저장하지 않았던 작업을 복구했습니다. 프로젝트를 저장해 두세요.")
        )
        return True

    def install_crash_dialog(self) -> None:
        """Unhandled exceptions in Qt slots: log (paths.install_excepthook) and tell the user
        instead of dying silently in a windowed build."""
        previous = sys.excepthook

        def hook(exc_type, exc, tb):
            previous(exc_type, exc, tb)
            detail = "".join(traceback.format_exception(exc_type, exc, tb))[-1500:]
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle(tr("예상하지 못한 오류"))
            box.setText(
                tr(
                    "프로그램 안에서 오류가 났습니다. 작업 내용은 '프로젝트 저장'으로 보관할 수 "
                    "있습니다.\n\n'문제 보고'를 누르면 로그와 프로젝트를 묶어 저장할 수 있고, "
                    "'문의하기'를 누르면 오류 내용을 채운 문의(GitHub 이슈)를 엽니다."
                )
            )
            box.setDetailedText(detail)
            report = box.addButton(tr("문제 보고"), QMessageBox.ButtonRole.ActionRole)
            contact = box.addButton(tr("문의하기"), QMessageBox.ButtonRole.ActionRole)
            box.addButton(tr("닫기"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is report:
                self.report_problem(detail)
            elif box.clickedButton() is contact:
                open_contact(self, detail)

        sys.excepthook = hook

    def report_problem(self, detail: str = "") -> None:
        project_file = None
        if self.project is not None:
            project_file = Path(tempfile.gettempdir()) / "kstruct_problem.kst"
            try:
                save_project(self.project, project_file)
            except Exception:  # the report must still be produced
                log.exception("could not save the project for the bug report")
                project_file = None
        extra = {"error.txt": detail} if detail else {}
        save_bug_report(self, project_file, extra)

    # --- navigation ------------------------------------------------------------------

    def _step_available(self, step: int) -> bool:
        if step == STEP_OPEN:
            return True
        if self.project is None:
            return False
        if step == STEP_RESULT:
            return self.result is not None
        return True

    def _step_done(self, step: int) -> bool:
        """Whether the step's input is complete (ticked in the navigator)."""
        p = self.project
        if p is None:
            return False
        if step == STEP_OPEN:
            return bool(p.parts)
        if step == STEP_MATERIAL:
            visible = p.visible_parts()
            return bool(visible) and all(part.material_id for part in visible)
        if step == STEP_FIXTURE:
            return bool(p.fixtures)
        if step == STEP_LOAD:
            return bool(p.loads) or any(b.preload for b in p.bolts)
        if step == STEP_MESH:
            return self.mesh is not None
        return self.result is not None

    def _update_nav(self) -> None:
        for i in range(self.nav.count()):
            item = self.nav.item(i)
            flags = item.flags()
            if self._step_available(i):
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
            item.setData(theme.STEP_DONE, self._step_done(i))
        row = self.nav.currentRow()
        self.prev_btn.setEnabled(row > 0)
        self.next_btn.setEnabled(row < STEP_RESULT and self._step_available(row + 1))

    def _go(self, step: int) -> None:
        if 0 <= step < len(self.panels) and self._step_available(step):
            self.nav.setCurrentRow(step)

    def _nav_changed(self, step: int) -> None:
        if step < 0:
            return
        if not self._step_available(step):
            self._go(STEP_OPEN)
            return
        self.stack.setCurrentIndex(step)
        if self.project is not None:
            self.panels[step].refresh()
        self.viewer.selection_enabled = step in (STEP_FIXTURE, STEP_LOAD, STEP_MESH)
        if step != STEP_RESULT:
            self.viewer.clear_result()
        if step in (STEP_FIXTURE, STEP_LOAD, STEP_MESH):
            self.viewer.set_selection([])
        self.viewer.show_mesh(self.mesh if step == STEP_MESH else None)
        if step == STEP_RESULT and self.result is not None:
            self.show_result()
        self._update_nav()

    # --- project lifecycle -----------------------------------------------------------

    def _set_project(self, project: Project, path: Path | None = None) -> None:
        self.project = project
        self.project_path = path
        self.mesh = self.result = self.summary = None
        self.case_results = self.overview = None
        self.previous = None
        self.viewer.load_project(project)
        for part in project.parts:
            if not part.visible:
                self.viewer.set_part_visible(part.id, False)
        self._annotate()
        self.panels[STEP_OPEN].refresh()
        self._update_nav()
        # a new model starts at step 1: staying on step 7 left the previous model's verdict
        # card on screen while the navigator had already disabled the result step
        if self.nav.currentRow() != STEP_OPEN:
            self._go(STEP_OPEN)
        self.stack.setCurrentIndex(STEP_OPEN)
        self._show_pane()
        self._reset_history()
        name = path.name if path else (project.source_path.name if project.source_path else "")
        self.setWindowTitle(f"KStruct — {name}" if name else tr("KStruct — 부품 강도 해석"))

    def new_project(self) -> None:
        if self.project is not None and not self._confirm(
            tr("지금 작업을 지우고 새로 시작할까요?")
        ):
            return
        self.project = None
        self.project_path = None
        self._dirty = False
        self.mesh = self.result = self.summary = None
        self.case_results = self.overview = None
        self.previous = None
        self.viewer.plotter.clear_actors()
        self.viewer.plotter.render()
        self.panels[STEP_OPEN].parts.clear()
        self._go(STEP_OPEN)
        self._update_nav()
        self._show_pane()
        self._reset_history()
        self.setWindowTitle(tr("KStruct — 부품 강도 해석"))

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("모델 열기"), prefs.last_dir(), tr(CAD_FILTER)
        )
        if path:
            self.load(Path(path))

    def open_sample(self, name: str) -> None:
        path = write_step(SAMPLES[name](), Path(tempfile.gettempdir()) / f"kstruct_{name}.step")
        self._sample_files.add(path)
        self.load(path)
        names = SAMPLE_PART_NAMES.get(name, [])
        if self.project is not None and len(names) == len(self.project.parts):
            for part, label in zip(self.project.parts, names, strict=True):
                part.name = tr(label)  # not the temporary file's name "kstruct_bracket"
            self.panels[STEP_OPEN].refresh()
            self._reset_history()

    def load(self, path: Path) -> None:
        if path.suffix.lower() == SUFFIX:
            self._load_project_file(path)
            return
        log.info("open %s", path)
        try:
            project = load_cad(path)
        except GeometryImportError as e:
            log.warning("import failed: %s", e)
            self.warn(str(e))
            return
        self._set_project(project)
        if len(project.parts) > 1:
            self.detect_connections(quiet=True)
        self._reset_history()  # the automatic connections are part of the opened state
        self._dirty = True  # a fresh import has no project file yet
        if path not in self._sample_files:  # built-in samples are not "recent files"
            prefs.remember_file(path)
        self.statusBar().showMessage(
            tr("{file}: 부품 {n}개를 불러왔습니다.", file=path.name, n=len(project.parts))
        )

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("프로젝트 열기"), prefs.last_dir(), tr(PROJECT_FILTER, suffix=SUFFIX)
        )
        if path:
            self._load_project_file(Path(path))

    def _load_project_file(self, path: Path) -> None:
        try:
            project = load_project(path)
        except ProjectFileError as e:
            self.warn(str(e))
            return
        except (KeyError, ValueError) as e:
            self.warn(tr("프로젝트 파일이 손상되었습니다: {error}", error=e))
            return
        self._set_project(project, path)
        self.mesh = load_project_mesh(path, project)
        self._dirty = False
        prefs.remember_file(path)
        extra = tr(" (저장된 메시 포함)") if self.mesh is not None else ""
        self.statusBar().showMessage(
            tr("{file} 프로젝트를 열었습니다.{extra}", file=path.name, extra=extra)
        )

    def save_project(self) -> None:
        if self.project is None:
            return
        path = self.project_path
        if path is None:
            source = self.project.source_path
            default = source.with_suffix(SUFFIX).name if source else ""
            start = str(Path(prefs.last_dir() or ".") / default) if default else prefs.last_dir()
            chosen, _ = QFileDialog.getSaveFileName(
                self, tr("프로젝트 저장"), start, tr(PROJECT_FILTER, suffix=SUFFIX)
            )
            if not chosen:
                return
            path = Path(chosen).with_suffix(SUFFIX)
        save_project(self.project, path, self.mesh)
        self.project_path = path
        self._dirty = False
        prefs.clear_autosave()
        prefs.remember_file(path)
        self.statusBar().showMessage(tr("{file}에 저장했습니다.", file=path.name))

    def detect_connections(self, quiet: bool = False) -> None:
        if self.project is None:
            return
        found = find_touching_faces(active_project(self.project))
        self.project.connections = found
        self.invalidate_mesh()
        self.panels[STEP_OPEN].refresh()
        if not quiet:
            self.statusBar().showMessage(tr("맞닿은 면 {n}쌍을 찾아 접합했습니다.", n=len(found)))

    # --- changes ---------------------------------------------------------------------

    def invalidate_mesh(self) -> None:
        self._dirty = True
        self.mesh = None
        self._invalidate_result()
        self._record()

    def _invalidate_result(self) -> None:
        if self.summary is not None:  # remember what the run being discarded came to
            self.previous = self._key_numbers() or self.previous
        self.result = self.summary = None
        self.case_results = self.overview = None
        self._update_nav()

    def _project_changed(self) -> None:
        self._dirty = True
        self._invalidate_result()
        self._annotate()
        self._record()

    # --- undo / redo -----------------------------------------------------------------

    def _document(self) -> str:
        return json.dumps(to_document(self.project), ensure_ascii=False, sort_keys=True)

    def _reset_history(self) -> None:
        """A project was opened or closed: nothing to undo yet."""
        self._history.clear()
        self._future.clear()
        self._baseline = self._document() if self.project is not None else None
        self._update_undo_actions()

    def _record(self) -> None:
        """Remember the state before the change that just happened (called after every
        edit; edits that did not change the project document are ignored)."""
        if self.project is None or self._restoring:
            return
        doc = self._document()
        if doc == self._baseline:
            return
        if self._baseline is not None:
            self._history.append(self._baseline)
            del self._history[:-MAX_UNDO]
        self._future.clear()
        self._baseline = doc
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(bool(self._history))
            self.redo_action.setEnabled(bool(self._future))

    def undo(self) -> None:
        self._step_history(self._history, self._future, tr("실행 취소했습니다."))

    def redo(self) -> None:
        self._step_history(self._future, self._history, tr("다시 실행했습니다."))

    def _step_history(self, source: list[str], target: list[str], message: str) -> None:
        if self.project is None or self._busy:
            return
        self._record()  # an edit that has not reported itself yet
        if not source:
            return
        target.append(self._baseline)
        self._restore(source.pop())
        self.statusBar().showMessage(message)

    def _restore(self, doc: str) -> None:
        """Make the project what the document says (fixtures, loads, materials, settings,
        hidden parts); the mesh survives when nothing it depends on changed."""
        old = self.project
        project = from_document(json.loads(doc), {p.id: p.shape for p in old.parts})

        def mesh_inputs(p: Project):
            return (
                p.mesh_quality,
                [(r.faces, r.size) for r in p.mesh_refinements],
                [part.id for part in p.parts if part.visible],
            )

        self._restoring = True
        try:
            self.project = project
            self.viewer.set_project(project)
            for part in project.parts:
                self.viewer.set_part_visible(part.id, part.visible)
            if mesh_inputs(old) != mesh_inputs(project):
                self.mesh = None
            self._dirty = True
            self._invalidate_result()
            self._annotate()
            self.panels[STEP_OPEN].refresh()
            row = self.nav.currentRow()
            if row == STEP_RESULT:
                self._go(STEP_SOLVE)  # the result is gone
            elif row != STEP_OPEN:
                self.panels[row].refresh()
            if self.nav.currentRow() == STEP_MESH:
                self.viewer.show_mesh(self.mesh)
            self._update_nav()
        finally:
            self._restoring = False
        self._baseline = self._document()  # a panel may normalise a value while refreshing
        self._update_undo_actions()

    def _annotate(self) -> None:
        if self.project is not None:
            self.viewer.annotate(self.project.fixtures, self.project.loads, self.project.bolts)

    def _selection_changed(self, refs: list) -> None:
        panel = self.panels[self.nav.currentRow()]
        if hasattr(panel, "update_selection"):
            panel.update_selection(refs)

    def readiness(self) -> list[tuple[bool, str]]:
        if self.project is None:
            return [(False, tr("모델을 여세요"))]
        p = active_project(self.project)
        visible = p.parts
        if not visible:
            return [(False, tr("모든 부품이 숨겨져 있습니다. 1단계에서 부품을 체크하세요"))]
        missing = [x.name for x in visible if x.material_id is None]
        checks = [
            (True, tr("모델: 부품 {n}개", n=len(visible))),
            (
                not missing,
                tr("재료: 모든 부품 지정됨")
                if not missing
                else tr("재료 미지정: {names}", names=", ".join(missing)),
            ),
        ]
        thermal_bcs = [ld for ld in p.loads if ld.kind in THERMAL_LOADS]
        if p.analysis in (AnalysisKind.STATIC, AnalysisKind.BUCKLING):
            checks.append(
                (
                    bool(p.fixtures),
                    tr("고정: {n}개", n=len(p.fixtures)) if p.fixtures else tr("고정이 없습니다"),
                )
            )
            if p.analysis == AnalysisKind.STATIC and len(p.cases) > 1:
                empty = [c.name for c in p.cases if not c.loads]
                checks.append(
                    (
                        not empty,
                        tr(
                            "하중: 케이스 {cases}개 모두 계산 ({loads}개)",
                            cases=len(p.cases),
                            loads=len(p.all_loads()),
                        )
                        if not empty
                        else tr("하중이 없는 케이스: {names}", names=", ".join(empty)),
                    )
                )
            else:
                mech = [ld for ld in p.loads if ld.kind not in THERMAL_LOADS]
                text = tr("하중: {n}개", n=len(p.loads)) if p.loads else tr("하중이 없습니다")
                if len(p.cases) > 1:
                    text += tr(" ('{case}' 케이스만 사용)", case=p.case.name)
                checks.append((bool(mech or thermal_bcs), text))
        elif p.analysis == AnalysisKind.MODAL:
            checks.append(
                (
                    True,
                    tr("고정: {n}개", n=len(p.fixtures))
                    if p.fixtures
                    else tr("고정 없음: 공중에 떠 있는 상태의 진동수를 구합니다"),
                )
            )
        elif p.thermal_duration > 0:  # over time: any heat condition will do
            checks.append(
                (
                    bool(thermal_bcs),
                    tr(
                        "열 조건: {n}개 · {duration} 동안",
                        n=len(thermal_bcs),
                        duration=format_duration(p.thermal_duration),
                    )
                    if thermal_bcs
                    else tr("4단계에서 열 입력·고정 온도·대류·복사 중 하나 이상을 지정하세요"),
                )
            )
        else:
            has_ref = any(ld.kind in TEMPERATURE_REFERENCES for ld in thermal_bcs)
            checks.append(
                (
                    has_ref,
                    tr("열 조건: {n}개", n=len(thermal_bcs))
                    if has_ref
                    else tr(
                        "4단계에서 '고정 온도', '대류 냉각', '복사 냉각' 중 하나 이상을 지정하세요"
                    ),
                )
            )
        if p.analysis == AnalysisKind.STATIC and (p.plasticity or p.large_deformation):
            reason = nonlinear_unsupported(p)
            checks.append(
                (
                    reason is None,
                    reason or tr("비선형 계산: 하중을 나눠 걸며 반복합니다(더 오래 걸림)"),
                )
            )
        excluded = excluded_by_hidden_parts(self.project)
        if excluded:
            checks.append(
                (
                    True,
                    tr(
                        "숨긴 부품에 지정된 {items}는 해석에서 제외합니다",
                        items=", ".join(excluded),
                    ),
                )
            )
        checks.append(
            (True, tr("메시: 준비됨") if self.mesh else tr("메시: 해석할 때 자동으로 만듭니다"))
        )
        return checks

    # --- meshing & solving -----------------------------------------------------------

    def build_mesh(self) -> None:
        if self.project is None or self._busy:
            return
        panel: steps.MeshStep = self.panels[STEP_MESH]
        panel.set_busy(True)
        self._busy = True
        project = active_project(self.project)

        def done(mesh):
            self._busy = False
            self._job = None
            panel.set_busy(False)
            self.mesh = mesh
            self._invalidate_result()
            panel.refresh()
            if self.nav.currentRow() == STEP_MESH:
                self.viewer.show_mesh(mesh if panel.show_mesh.isChecked() else None)

        def failed(e):
            self._busy = False
            self._job = None
            panel.set_busy(False)
            if isinstance(e, Cancelled):
                panel.info.setText(tr("메시 만들기를 취소했습니다."))
                return
            self.warn(friendly_error(e))

        self._job = run_job(
            lambda progress: generate_mesh(project, check=progress.check), done, failed
        )

    def cancel_job(self) -> None:
        """Stop the running mesh / solve job at its next checkpoint."""
        if self._job is not None and not self._job.cancelled:
            log.info("job cancel requested")
            self._job.cancel()
            solve_panel = self.panels[STEP_SOLVE]
            text = tr("취소하는 중… 지금 계산 중인 단계가 끝나면 멈춥니다.")
            if solve_panel.cancel.isVisible():
                solve_panel.status.setText(text)
            else:
                self.panels[STEP_MESH].info.setText(text)

    def solve(self) -> None:
        if self.project is None or self._busy:
            return
        panel: steps.SolveStep = self.panels[STEP_SOLVE]
        project, mesh = active_project(self.project), self.mesh
        mats = standards.project_materials(project)
        materials = {p.id: mats[p.material_id] for p in project.parts}
        kind = project.analysis
        self._busy = True
        log.info(
            "solve %s: parts=%d fixtures=%d loads=%d mesh=%s",
            kind.value,
            len(project.parts),
            len(project.fixtures),
            len(project.loads),
            project.mesh_quality,
        )
        panel.set_progress(0.0, tr("시작"))
        started = time.perf_counter()

        def show_progress(fraction: float, message: str) -> None:
            seconds = time.perf_counter() - started
            clock = tr(" · {seconds:.0f}초", seconds=seconds) if seconds >= 3 else ""
            panel.set_progress(fraction, f"{message}{clock}")

        def job(progress):
            m = mesh
            if m is None:
                progress(0.02, tr("메시 만드는 중 (잠시 걸릴 수 있습니다)"))
                m = generate_mesh(project, check=progress.check)
            if kind == AnalysisKind.MODAL:
                return m, solve_modal(project, m, materials, project.n_modes, progress)
            if kind == AnalysisKind.THERMAL:
                return m, solve_thermal(project, m, materials, progress)
            if kind == AnalysisKind.BUCKLING:
                return m, solve_buckling(project, m, materials, project.n_modes, progress)
            return m, solve_cases(project, m, materials, progress)

        def done(value):
            self._busy = False
            self._job = None
            self.previous = self._key_numbers() or self.previous  # solved again unchanged
            self._solved_quality = project.mesh_quality
            self.mesh, self.result = value
            self.materials = materials
            if isinstance(self.result, list):  # strength: one result per load case
                pack = standards.effective_pack(project)
                names = [c.name for c in project.cases]
                self.case_results = [
                    (name, r, summarize(with_case(project, i), r, pack))
                    for i, (name, r) in enumerate(zip(names, self.result, strict=True))
                ]
                summaries = [s for _, _, s in self.case_results]
                self.overview = compare_cases(names, summaries) if len(names) > 1 else None
                chosen = self.overview.worst if self.overview else 0
                self.result, self.summary = self.case_results[chosen][1:]
                self.project.active_case = chosen
            elif isinstance(self.result, ModalResult):
                self.summary = summarize_modal(project, self.result)
            elif isinstance(self.result, ThermalResult):
                self.summary = summarize_thermal(project, self.result, materials)
            elif isinstance(self.result, BucklingResult):
                pack = standards.effective_pack(project)
                self.summary = summarize_buckling(project, self.result, pack)
            panel.set_progress(None, "")
            self._configure_result_panel()
            self._update_nav()
            self._go(STEP_RESULT)
            seconds = time.perf_counter() - started
            log.info("solve finished in %.1f s", seconds)
            self.statusBar().showMessage(
                tr("해석을 마쳤습니다 ({seconds:.1f}초).", seconds=seconds)
            )

        def failed(e):
            self._busy = False
            self._job = None
            panel.set_progress(None, "")
            if isinstance(e, Cancelled):
                panel.status.setText(tr("해석을 취소했습니다."))
                return
            log.warning("solve failed: %s: %s", type(e).__name__, e)
            self.warn(friendly_error(e))

        self._job = run_job(job, done, failed, show_progress)

    # --- results ---------------------------------------------------------------------

    def _configure_result_panel(self) -> None:
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        r = self.result
        if isinstance(r, ModalResult):
            panel.set_fields([("mode", tr("진동 모양 (상대 크기)"))])
            panel.set_modes(r.frequencies)
            panel.animate.setChecked(True)
        elif isinstance(r, ThermalResult):
            panel.set_fields(_translated(THERMAL_FIELDS))
            if r.transient:  # the "mode" selector picks the moment to show
                labels = [format_duration(t) for t in r.times]
                panel.set_modes(r.times, labels, tr("시각"))
                panel.mode.setCurrentIndex(len(labels) - 1)
            else:
                panel.set_modes(None)
            panel.animate.setChecked(False)
        elif isinstance(r, BucklingResult):
            panel.set_fields([("mode", tr("좌굴 모양 (상대 크기)"))])
            panel.set_modes(
                r.factors,
                [tr("{n}차: 하중 ×{f:,.2f}", n=i + 1, f=f) for i, f in enumerate(r.factors)],
            )
            panel.animate.setChecked(True)
        else:
            fields = list(steps.FIELD_LABELS)
            if r.thermal is not None:
                fields += THERMAL_FIELDS
            if self.summary.fatigue is not None:
                fields += [("ffos", N_("피로 안전율")), ("life", N_("예상 수명 (반복 횟수)"))]
            if r.nonlinear is not None and r.nonlinear.plasticity:
                fields += [("peeq", N_("소성 변형률 (영구 변형 정도)"))]
            if r.nonlinear is not None and r.nonlinear.permanent is not None:
                fields += [("perm", N_("하중을 뺀 뒤 남는 변형"))]
            panel.set_fields(_translated(fields))
            panel.set_modes(None)
            panel.animate.setChecked(False)
        if self.case_results and len(self.case_results) > 1:
            labels = [
                f"{name} — {VERDICT_STYLE[s.verdict][0]} {s.governing_fos:.2f}"
                + (tr(" ◀ 가장 불리") if self.overview and i == self.overview.worst else "")
                for i, (name, _, s) in enumerate(self.case_results)
            ]
            panel.set_cases(labels, self.project.active_case)
        else:
            panel.set_cases(None)
        has_shape = not isinstance(r, ThermalResult)
        panel.scale.setEnabled(has_shape)
        panel.animate.setEnabled(has_shape)

    def _key_numbers(self) -> dict | None:
        """What the last strength run came to, kept for the comparison line."""
        s, p, r = self.summary, self.project, self.result
        if s is None or not isinstance(s, Summary) or r is None:
            return None
        quality = self._solved_quality or p.mesh_quality
        return {
            "label": tr(
                "{quality} 메시, 요소 {n:,}개, 케이스 '{case}'",
                quality=tr(steps.QUALITY_NAMES.get(quality, quality)),
                n=len(r.mesh.tets),
                case=p.case.name,
            ),
            "fos": s.governing_fos,
            "disp": s.max_disp,
            "stress": max(x.max_stress for x in s.parts),
        }

    def _comparison_line(self) -> str | None:
        prev, s = self.previous, self.summary
        if prev is None or not isinstance(s, Summary):
            return None

        def delta(a, b):
            return f" ({(b - a) / a * 100:+.1f}%)" if a else ""

        us = system_of(self.project)
        stress = max(x.max_stress for x in s.parts)
        text = tr(
            "<b>이전 해석과 비교</b> ({label}): 안전율 {fos0:.2f} → "
            "<b>{fos1:.2f}</b>{dfos}, 최대 응력 {stress0} → {stress1}{dstress}, "
            "최대 변형 {disp0} → {disp1}{ddisp}",
            label=prev["label"],
            fos0=prev["fos"],
            fos1=s.governing_fos,
            dfos=delta(prev["fos"], s.governing_fos),
            stress0=fmt(us, "stress", prev["stress"], ",.1f"),
            stress1=fmt(us, "stress", stress, ",.1f"),
            dstress=delta(prev["stress"], stress),
            disp0=fmt(us, "length", prev["disp"]),
            disp1=fmt(us, "length", s.max_disp),
            ddisp=delta(prev["disp"], s.max_disp),
        )
        if prev["fos"] and abs(s.governing_fos - prev["fos"]) / prev["fos"] > 0.1:
            text += tr(". 10% 넘게 달라졌으니 무엇이 바뀌었는지 확인하세요.")
        return text

    def select_case(self, index: int) -> None:
        """Result panel: show another load case of the last strength run."""
        if not self.case_results or not 0 <= index < len(self.case_results):
            return
        _, self.result, self.summary = self.case_results[index]
        self.project.active_case = index
        self._configure_result_panel()
        self.show_result()

    def _displacement(self) -> np.ndarray:
        r = self.result
        if isinstance(r, (ModalResult, BucklingResult)):
            return r.shapes[max(0, self.panels[STEP_RESULT].mode.currentIndex())]
        if isinstance(r, ThermalResult):
            return np.zeros_like(r.mesh.nodes)
        return r.displacement

    def result_field(self, key: str) -> ResultField:
        field = self._result_field_internal(key)
        kind = {
            "vm": "stress",
            "p1": "stress",
            "disp": "length",
            "ux": "length",
            "uy": "length",
            "uz": "length",
            "temp": "temperature",
            "perm": "length",
        }.get(key)
        if kind is None:
            return field
        q = quantity(system_of(self.project), kind)
        field.values = q.show(field.values)
        if field.clim is not None:
            field.clim = (q.show(field.clim[0]), q.show(field.clim[1]))
        field.title = field.title.replace(f"({field.unit})", f"({q.unit})")
        field.unit = q.unit
        return field

    def _result_field_internal(self, key: str) -> ResultField:
        r = self.result
        if key == "mode":
            mag = np.linalg.norm(self._displacement(), axis=1)
            title = (
                tr("좌굴 모양 (상대값)")
                if isinstance(r, BucklingResult)
                else tr("진동 모양 (상대값)")
            )
            return ResultField(mag / (mag.max() or 1.0), title, clim=(0, 1))
        if key in ("temp", "flux"):
            thermal = r if isinstance(r, ThermalResult) else r.thermal
            if key == "temp":
                values = thermal.temperature
                clim = None
                if isinstance(r, ThermalResult) and r.transient:
                    frame = self.panels[STEP_RESULT].mode.currentIndex()
                    values = r.history[frame if frame >= 0 else -1]
                    clim = (float(r.history.min()), float(r.history.max()))  # one scale
                return ResultField(values, tr("온도 (°C)"), "inferno", clim, "°C")
            return ResultField(
                thermal.heat_flux_magnitude, tr("열 흐름 (W/m²)"), "magma", None, "W/m²"
            )
        if key == "peeq":
            values = r.nonlinear.plastic_strain * 100
            # scale to the peak so a small yielded spot still stands out
            top = float(values.max()) if values.max() > 0 else 0.2
            return ResultField(values, tr("소성 변형률 (%)"), "YlOrRd", (0, top), "%")
        if key == "perm":
            return ResultField(
                np.linalg.norm(r.nonlinear.permanent, axis=1), tr("남는 변형 (mm)"), unit="mm"
            )
        if key in ("ffos", "life"):
            fatigue = self.summary.fatigue
            target = max(
                standards.effective_pack(self.project).target(m) for m in r.materials.values()
            )
            if key == "ffos":
                values = np.minimum(fatigue.safety_factor, 2 * target)
                return ResultField(values, tr("피로 안전율"), cmap="RdYlGn", clim=(0, 2 * target))
            # log scale: 10^3 (red) ... 10^9 = infinite (green)
            return ResultField(
                np.log10(fatigue.life),
                tr("수명 (10의 거듭제곱 회)"),
                "RdYlGn",
                (3, 9),
                tr("(log10 회)"),
            )
        if key == "vm":
            # non-negative by definition: pin the colour scale at 0 so quadratic
            # interpolation overshoot does not show as a negative minimum
            vm = r.von_mises
            return ResultField(vm, tr("응력 (MPa)"), clim=(0.0, float(vm.max())), unit="MPa")
        if key == "fos":
            pack = standards.effective_pack(self.project)
            target = max(pack.target(m) for m in r.materials.values())
            # clip before display: quadratic interpolation between a capped 100 and a small
            # neighbour would otherwise undershoot and paint false red spots
            values = np.minimum(r.safety_factor, 2 * target)
            return ResultField(values, tr("안전율"), cmap="RdYlGn", clim=(0, 2 * target))
        if key == "disp":
            d = r.displacement_magnitude
            return ResultField(d, tr("변형량 (mm)"), clim=(0.0, float(d.max())), unit="mm")
        if key in ("ux", "uy", "uz"):
            comp = r.displacement[:, "xyz".index(key[1])]
            lim = float(np.abs(comp).max()) or 1.0
            title = tr("변형량 {axis} (mm)", axis=key[1].upper())
            return ResultField(comp, title, "coolwarm", (-lim, lim), "mm")
        p1 = r.principal[:, 0]
        lim = float(np.abs(p1).max()) or 1.0
        return ResultField(p1, tr("최대 주응력 (MPa)"), "coolwarm", (-lim, lim), "MPa")

    def deformation_scale(self, key: str) -> float:
        if isinstance(self.result, ThermalResult) or key == "none":
            return 0.0
        shape_only = isinstance(self.result, (ModalResult, BucklingResult))
        if key == "true" and not shape_only:
            return 1.0  # mode shapes have no physical size
        size = float(np.ptp(self.mesh.nodes, axis=0).max())
        peak = float(np.linalg.norm(self._displacement(), axis=1).max())
        fraction = 0.1 if shape_only else 0.05
        return fraction * size / peak if peak > 0 else 0.0

    def show_result(self) -> None:
        if self.result is None or self.nav.currentRow() != STEP_RESULT:
            return
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        field = self.result_field(panel.field_key())
        scale = self.deformation_scale(panel.scale_key())
        if isinstance(self.result, (ModalResult, BucklingResult)) or scale == 0.0:
            panel.scale_note.setText("")
        elif scale == 1.0:
            panel.scale_note.setText(tr("변형을 실제 크기로 보여줍니다."))
        else:
            panel.scale_note.setText(
                tr(
                    "변형을 {scale:,.0f}배 확대해서 보여줍니다. 회색 선은 변형 전 모양입니다.",
                    scale=scale,
                )
            )
        clip = None
        section = panel.clip_fraction()
        if section is not None:
            axis, frac = section
            lo, hi = self.mesh.nodes.min(axis=0), self.mesh.nodes.max(axis=0)
            origin = (lo + hi) / 2
            origin[axis] = lo[axis] + frac * (hi[axis] - lo[axis])
            normal = np.zeros(3)
            normal[axis] = -1.0  # keep the lower side, look at the cut face
            clip = (normal, origin)
        self.viewer.show_result(
            self.mesh,
            self._displacement(),
            field,
            scale,
            symmetric=isinstance(self.result, (ModalResult, BucklingResult)),
            clip=clip,
        )
        self._fill_summary(panel)

    def _fill_summary(self, panel: steps.ResultStep) -> None:
        s = self.summary
        color, bg = self.theme.verdict(s.verdict)
        panel.card.setStyleSheet(
            f"QFrame#verdictCard {{ background: {bg}; border: 1px solid {color}; "
            f"border-left: 5px solid {color}; border-radius: 8px; }}"
        )
        panel.card_title.setText(
            f"<span style='font-size:14pt; color:{color}'><b>{s.headline}</b></span>"
        )
        panel.card_body.setText(s.explanation)
        lines = []
        r = self.result
        us = system_of(self.project)
        if isinstance(r, ModalResult):
            rows = []
            for i, f in enumerate(r.frequencies):
                px, py, pz = r.participation[i] * 100
                rows.append(
                    tr(
                        "{n}차 <b>{f:,.1f} Hz</b> ({rpm:,.0f} rpm)"
                        " · 주 방향 질량 X {px:.0f}% Y {py:.0f}% Z {pz:.0f}%",
                        n=i + 1,
                        f=f,
                        rpm=f * 60,
                        px=px,
                        py=py,
                        pz=pz,
                    )
                )
            lines.append("<br>".join(rows))
        elif isinstance(r, BucklingResult):
            applied = np.linalg.norm(r.static.applied_load)
            rows = [
                tr("{n}차 하중 배수 <b>{f:,.2f}</b>", n=i + 1, f=f)
                + (tr(" (약 {load:,.0f} N에서 좌굴)", load=f * applied) if applied > 0 else "")
                for i, f in enumerate(r.factors)
            ]
            lines.append("<br>".join(rows))
        elif isinstance(r, ThermalResult):
            for pid, m in self.materials.items():
                nodes = r.mesh.part_nodes(pid)
                limit = (
                    tr(" (권장 최고 {t:.0f} °C)", t=m.max_service_temperature)
                    if m.max_service_temperature
                    else ""
                )
                lines.append(
                    f"<b>{self.project.part(pid).name}</b>: "
                    f"{fmt(us, 'temperature', r.temperature[nodes].min(), '.1f')}"
                    f" ~ <b>{fmt(us, 'temperature', r.temperature[nodes].max(), '.1f')}</b>{limit}"
                )
            if r.heat_in:
                lines.append(
                    tr(
                        "들어온 열 {heat_in} · 빠져나간 열 {heat_out}",
                        heat_in=fmt("SI", "power", r.heat_in),
                        heat_out=fmt("SI", "power", r.heat_out),
                    )
                )
        else:
            if self.overview is not None:
                o = self.overview
                lines.append(
                    tr(
                        "<b>{headline}</b> ({worst}). 지금 보는 케이스: {case}",
                        headline=o.headline,
                        worst=o.summaries[o.worst].headline,
                        case=self.project.case.name,
                    )
                )
            comparison = self._comparison_line()
            if comparison:
                lines.append(comparison)
            for p in s.parts:
                lines.append(
                    tr(
                        "<b>{name}</b> ({material})<br>"
                        "&nbsp;최대 응력 {stress} / 강도 {capacity}<br>"
                        "&nbsp;최소 안전율 <b>{fos:.2f}</b> (권장 {target:.1f})"
                        " · 최대 변형 {disp}",
                        name=p.name,
                        material=p.material,
                        stress=fmt(us, "stress", p.max_stress, ",.1f"),
                        capacity=fmt(us, "stress", p.capacity, ",.0f"),
                        fos=p.min_fos,
                        target=p.target,
                        disp=fmt(us, "length", p.max_disp),
                    )
                )
                if p.hotspot_fos is not None:
                    lines[-1] += tr(
                        "<br>&nbsp;특이점({kind}) 제외: 응력 {stress}, 안전율 <b>{fos:.2f}</b>",
                        kind=tr_data(p.singular_kind),
                        stress=fmt(us, "stress", p.hotspot_stress, ",.1f"),
                        fos=p.hotspot_fos,
                    )
                if p.plastic_strain is not None:
                    lines[-1] += tr(
                        "<br>&nbsp;소성 변형률 <b>{strain:.2g}%</b>", strain=p.plastic_strain * 100
                    ) + (tr(" (안전율 1 미만 = 항복한 곳)") if p.plastic_strain > 0 else "")
                if p.fatigue_fos is not None:
                    life = (
                        tr("무한") if p.fatigue_life >= 1e9 else tr("{n:,.0f}회", n=p.fatigue_life)
                    )
                    lines[-1] += tr(
                        "<br>&nbsp;피로 안전율 <b>{fos:.2f}</b> · 예상 수명 {life}",
                        fos=p.fatigue_fos,
                        life=life,
                    )
            for b in r.bolts or []:
                lines.append(
                    tr(
                        "<b>{name}</b> ({size} {grade}) 축력 {axial} · 전단 {shear}"
                        " · 안전율 <b>{fos:.2f}</b>",
                        name=b.name,
                        size=b.size,
                        grade=b.grade,
                        axial=fmt(us, "force", b.axial, ",.0f"),
                        shear=fmt(us, "force", b.shear, ",.0f"),
                        fos=b.safety_factor,
                    )
                )
            if r.thermal is not None:
                lines.append(
                    tr(
                        "열응력 포함: 온도 {tmin:.1f} ~ {tmax:.1f} °C"
                        " (응력 없는 기준 온도 {tref:.0f} °C)",
                        tmin=r.thermal.temperature.min(),
                        tmax=r.thermal.temperature.max(),
                        tref=self.project.reference_temperature,
                    )
                )
        for note in s.notes:
            lines.append(f"<span style='color:{self.theme.warn}'>※ {note}</span>")
        panel.summary.setText("<br><br>".join(lines))

    def save_report(self) -> None:
        if self.result is None:
            return
        from kstruct.report.pdf import build_html, write_pdf

        stem = self.project.source_path.stem if self.project.source_path else "kstruct"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("보고서 저장"), tr("{stem}_보고서.pdf", stem=stem), "PDF (*.pdf)"
        )
        if not path:
            return
        images = self.capture_result_images()
        html_text = build_html(
            self.project, self.result, self.summary, images, self.materials, self.overview
        )
        write_pdf(html_text, path)
        self.statusBar().showMessage(tr("보고서를 저장했습니다: {path}", path=path))

    def export_result(self) -> None:
        if self.result is None:
            return
        from kstruct.report.export import export_csv, export_vtu

        stem = self.project.source_path.stem if self.project.source_path else "kstruct"
        path, chosen = QFileDialog.getSaveFileName(
            self,
            tr("결과 내보내기"),
            tr("{stem}_결과.csv", stem=stem),
            tr("CSV 표 (*.csv);;VTK 파일, ParaView용 (*.vtu)"),
        )
        if not path:
            return
        if "vtu" in chosen or path.lower().endswith(".vtu"):
            out = export_vtu(path, self.project, self.result, self.summary)
        else:
            out = export_csv(path, self.project, self.result, self.summary)
        self.statusBar().showMessage(tr("결과를 내보냈습니다: {path}", path=out))

    def capture_result_images(self) -> list[tuple[str, bytes]]:
        """Render the main result fields to PNG for the report, then restore the view."""
        r = self.result
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        animated = panel.animate.isChecked()
        self.viewer.set_animated(False)
        shots: list[tuple[str, str, int | None]]
        if isinstance(r, ModalResult):
            shots = [
                ("mode", tr("{n}차 진동 모양 ({f:,.1f} Hz)", n=i + 1, f=f), i)
                for i, f in enumerate(r.frequencies[:4])
            ]
        elif isinstance(r, BucklingResult):
            shots = [
                ("mode", tr("{n}차 좌굴 모양 (하중 ×{f:,.2f})", n=i + 1, f=f), i)
                for i, f in enumerate(r.factors[:3])
            ]
        elif isinstance(r, ThermalResult):
            shots = [("temp", tr("온도 분포"), None), ("flux", tr("열 흐름"), None)]
        else:
            shots = [
                ("vm", tr("응력 분포"), None),
                ("fos", tr("안전율 분포"), None),
                ("disp", tr("변형량"), None),
            ]
            if r.thermal is not None:
                shots.append(("temp", tr("온도 분포"), None))
            if self.summary.fatigue is not None:
                shots.append(("ffos", tr("피로 안전율 분포"), None))
        images = []
        current_mode = panel.mode.currentIndex()
        for key, caption, mode in shots:
            if mode is not None:
                panel.mode.blockSignals(True)
                panel.mode.setCurrentIndex(mode)
                panel.mode.blockSignals(False)
            field = self.result_field(key)
            self.viewer.show_result(
                self.mesh, self._displacement(), field, self.deformation_scale("auto")
            )
            images.append((caption, _png(self.viewer.plotter.screenshot(return_img=True))))
        section = panel.clip_fraction()
        if section is not None:  # the cut the user is looking at goes into the report too
            self.show_result()
            name = "XYZ"[section[0]]
            images.append(
                (
                    tr("단면 ({axis} 방향 {pos:.0f}% 위치)", axis=name, pos=section[1] * 100),
                    _png(self.viewer.plotter.screenshot(return_img=True)),
                )
            )
        panel.mode.blockSignals(True)
        panel.mode.setCurrentIndex(current_mode)
        panel.mode.blockSignals(False)
        self.show_result()
        self.viewer.set_animated(animated)
        return images

    # --- dialogs ---------------------------------------------------------------------

    def warn(self, message: str) -> None:
        QMessageBox.warning(self, tr("확인이 필요합니다"), message)

    def _confirm(self, message: str) -> bool:
        answer = QMessageBox.question(self, tr("확인"), message)
        return answer == QMessageBox.StandardButton.Yes


def _translated(fields: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(key, Korean label) pairs from N_ tables -> labels in the UI language."""
    return [(key, tr(label)) for key, label in fields]


def _png(image: np.ndarray) -> bytes:
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImage

    h, w, _ = image.shape
    rgb = np.ascontiguousarray(image[:, :, :3])
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.copy().save(buf, "PNG")
    return bytes(buf.data())
