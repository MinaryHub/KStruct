"""Main window: wizard (left), 3D viewer (center). Owns the Project, mesh and result."""

from __future__ import annotations

import logging
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from kstruct import standards
from kstruct.fem import ConstraintError, MeshingError, SolverError, generate_mesh
from kstruct.fem.buckling import BucklingResult, solve_buckling
from kstruct.fem.mesh import FEMesh
from kstruct.fem.modal import ModalResult, solve_modal
from kstruct.fem.static import StaticResult, solve_cases
from kstruct.fem.thermal import ThermalResult, solve_thermal
from kstruct.geometry import GeometryImportError, load_cad
from kstruct.geometry.contacts import find_touching_faces
from kstruct.geometry.samples import bolted_plates, l_bracket, two_plates, write_step
from kstruct.model import (
    THERMAL_LOADS,
    AnalysisKind,
    LoadKind,
    Project,
    active_project,
    excluded_by_hidden_parts,
    with_case,
)
from kstruct.project_io import SUFFIX, ProjectFileError, load_project, save_project
from kstruct.report.summary import (
    VERDICT_STYLE,
    CaseOverview,
    Summary,
    compare_cases,
    summarize,
    summarize_buckling,
    summarize_modal,
    summarize_thermal,
)
from kstruct.ui import steps
from kstruct.ui.dialogs import AboutDialog, save_bug_report
from kstruct.ui.help import HelpButton, HelpDialog, topic, with_help
from kstruct.ui.viewer import ModelViewer, ResultField
from kstruct.ui.worker import run_job
from kstruct.units import fmt, quantity, system_of

CAD_FILTER = (
    "모든 3D 파일 (*.step *.stp *.iges *.igs *.stl *.obj *.fbx *.gltf *.glb *.3mf *.kst);;"
    "CAD 파일 (*.step *.stp *.iges *.igs);;"
    "메시 파일 (*.stl *.obj *.fbx *.gltf *.glb *.3mf);;"
    "모든 파일 (*)"
)
PROJECT_FILTER = f"KStruct 프로젝트 (*{SUFFIX})"
SAMPLES = {"bracket": l_bracket, "two_plates": two_plates, "bolted": bolted_plates}

log = logging.getLogger("kstruct.ui")
STEP_OPEN, STEP_MATERIAL, STEP_FIXTURE, STEP_LOAD, STEP_MESH, STEP_SOLVE, STEP_RESULT = range(7)
STEP_NAMES = ["1. 모델 열기", "2. 재료", "3. 고정", "4. 하중", "5. 메시", "6. 해석", "7. 결과"]
THERMAL_FIELDS = [("temp", "온도"), ("flux", "열 흐름 (열유속)")]
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
            return (
                "부품이 충분히 고정되지 않아 공중에서 움직이거나 돌 수 있습니다.\n\n"
                "• 고정을 추가하세요 (예: 볼트 구멍이나 바닥면을 '완전 고정').\n"
                "• 여러 부품이면 각 부품이 고정되어 있거나, 다른 부품과 '접합'되어 있어야 합니다."
            )
        if str(e) == "bolt without contact":
            return (
                "볼트로 조인 두 부품 사이에 맞닿은 면(접촉 또는 접합)이 없습니다. 체결력은 부품끼리 "
                "서로 눌러 줄 때만 생기므로, 1단계에서 '맞닿은 면 자동으로 찾기'를 누르고 연결 종류를 "
                "'접촉'으로 지정하세요. (체결력을 0으로 하면 연결 없이도 계산할 수 있습니다.)"
            )
        if str(e) == "contact unstable":
            return (
                "접촉으로만 연결된 부품이 미끄러지거나 떨어져 나가서 붙잡혀 있지 않습니다.\n\n"
                "• 그 부품을 직접 고정하거나,\n"
                "• 실제로 볼트·용접으로 붙어 있다면 1단계에서 연결 종류를 '접합'으로 바꾸세요.\n"
                "• 접촉은 마찰이 없다고 가정하므로 옆으로 미는 힘은 버티지 못합니다."
            )
        if str(e) == "not converged":
            return "계산이 수렴하지 않았습니다. 메시를 '보통'으로 바꾸거나 고정 조건을 확인하세요."
        return str(e)
    if isinstance(e, (MeshingError, ConstraintError, GeometryImportError, ProjectFileError)):
        return str(e)
    return f"예상하지 못한 오류가 발생했습니다:\n{type(e).__name__}: {e}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KStruct — 부품 강도 해석")
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
        self.materials: dict = {}
        self._busy = False

        self.viewer = ModelViewer()
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
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1060])
        self.setCentralWidget(splitter)
        self._build_toolbar()
        self._go(STEP_OPEN)
        self.statusBar().showMessage("'파일 열기'로 STEP 파일을 불러오거나 예제를 눌러 보세요.")

    # --- layout ----------------------------------------------------------------------

    def _build_side(self) -> QWidget:
        side = QWidget()
        layout = QVBoxLayout(side)
        self.nav = QListWidget()
        for name, key in zip(STEP_NAMES, STEP_KEYS, strict=True):
            item = QListWidgetItem(name)
            item.setToolTip(topic(key).short)
            self.nav.addItem(item)
        self.nav.setMaximumHeight(7 * 22 + 8)
        self.nav.currentRowChanged.connect(self._nav_changed)
        layout.addWidget(with_help(self.nav, "help.mode"))

        self.stack = QStackedWidget()
        for panel in self.panels:
            scroll = QScrollArea()
            scroll.setWidget(panel)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self.stack.addWidget(scroll)
        layout.addWidget(self.stack, 1)

        buttons = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 이전")
        self.next_btn = QPushButton("다음 ▶")
        self.prev_btn.clicked.connect(lambda: self._go(self.nav.currentRow() - 1))
        self.next_btn.clicked.connect(lambda: self._go(self.nav.currentRow() + 1))
        buttons.addWidget(self.prev_btn)
        buttons.addWidget(self.next_btn)
        layout.addLayout(buttons)
        return side

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("기본")
        bar.setMovable(False)

        def action(text, slot, key=None, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            if key:
                a.setToolTip(topic(key).short)
            a.triggered.connect(slot)
            bar.addAction(a)
            return a

        action("새로 시작", self.new_project, "project.new")
        action("모델 열기", self.open_file, "step.open", QKeySequence.StandardKey.Open)
        action("프로젝트 열기", self.open_project, "project.open")
        action("프로젝트 저장", self.save_project, "project.save", QKeySequence.StandardKey.Save)
        bar.addWidget(HelpButton("project.save"))
        bar.addSeparator()
        action("화면 맞춤", self.viewer.fit, "view.fit", "F")
        bar.addWidget(HelpButton("view.fit"))
        bar.addSeparator()
        action("사용 안내", lambda: HelpDialog("help.mode", self).exec(), "help.mode", "F1")
        action("문제 보고", self.report_problem, "help.report")
        action("정보", lambda: AboutDialog(self).exec(), "help.about")

    def install_crash_dialog(self) -> None:
        """Unhandled exceptions in Qt slots: log (paths.install_excepthook) and tell the user
        instead of dying silently in a windowed build."""
        previous = sys.excepthook

        def hook(exc_type, exc, tb):
            previous(exc_type, exc, tb)
            detail = "".join(traceback.format_exception(exc_type, exc, tb))[-1500:]
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("예상하지 못한 오류")
            box.setText(
                "프로그램 안에서 오류가 났습니다. 작업 내용은 '프로젝트 저장'으로 보관할 수 "
                "있습니다.\n\n'문제 보고'를 누르면 로그와 프로젝트를 묶어 저장할 수 있습니다."
            )
            box.setDetailedText(detail)
            report = box.addButton("문제 보고", QMessageBox.ButtonRole.ActionRole)
            box.addButton("닫기", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is report:
                self.report_problem(detail)

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

    def _update_nav(self) -> None:
        for i in range(self.nav.count()):
            item = self.nav.item(i)
            flags = item.flags()
            if self._step_available(i):
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
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
        self.viewer.selection_enabled = step in (STEP_FIXTURE, STEP_LOAD)
        if step != STEP_RESULT:
            self.viewer.clear_result()
        if step in (STEP_FIXTURE, STEP_LOAD):
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
        self.viewer.load_project(project)
        for part in project.parts:
            if not part.visible:
                self.viewer.set_part_visible(part.id, False)
        self._annotate()
        self.panels[STEP_OPEN].refresh()
        self._update_nav()
        name = path.name if path else (project.source_path.name if project.source_path else "")
        self.setWindowTitle(f"KStruct — {name}" if name else "KStruct — 부품 강도 해석")

    def new_project(self) -> None:
        if self.project is not None and not self._confirm("지금 작업을 지우고 새로 시작할까요?"):
            return
        self.project = None
        self.mesh = self.result = self.summary = None
        self.case_results = self.overview = None
        self.viewer.plotter.clear_actors()
        self.viewer.plotter.render()
        self.panels[STEP_OPEN].parts.clear()
        self._go(STEP_OPEN)
        self._update_nav()

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "모델 열기", "", CAD_FILTER)
        if path:
            self.load(Path(path))

    def open_sample(self, name: str) -> None:
        path = write_step(SAMPLES[name](), Path(tempfile.gettempdir()) / f"kstruct_{name}.step")
        self.load(path)

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
        self.statusBar().showMessage(f"{path.name}: 부품 {len(project.parts)}개를 불러왔습니다.")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "", PROJECT_FILTER)
        if path:
            self._load_project_file(Path(path))

    def _load_project_file(self, path: Path) -> None:
        try:
            project = load_project(path)
        except ProjectFileError as e:
            self.warn(str(e))
            return
        except (KeyError, ValueError) as e:
            self.warn(f"프로젝트 파일이 손상되었습니다: {e}")
            return
        self._set_project(project, path)
        self.statusBar().showMessage(f"{path.name} 프로젝트를 열었습니다.")

    def save_project(self) -> None:
        if self.project is None:
            return
        path = self.project_path
        if path is None:
            source = self.project.source_path
            default = source.with_suffix(SUFFIX).name if source else ""
            chosen, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장", default, PROJECT_FILTER)
            if not chosen:
                return
            path = Path(chosen).with_suffix(SUFFIX)
        save_project(self.project, path)
        self.project_path = path
        self.statusBar().showMessage(f"{path.name}에 저장했습니다.")

    def detect_connections(self, quiet: bool = False) -> None:
        if self.project is None:
            return
        found = find_touching_faces(active_project(self.project))
        self.project.connections = found
        self.invalidate_mesh()
        self.panels[STEP_OPEN].refresh()
        if not quiet:
            self.statusBar().showMessage(f"맞닿은 면 {len(found)}쌍을 찾아 접합했습니다.")

    # --- changes ---------------------------------------------------------------------

    def invalidate_mesh(self) -> None:
        self.mesh = None
        self._invalidate_result()

    def _invalidate_result(self) -> None:
        self.result = self.summary = None
        self.case_results = self.overview = None
        self._update_nav()

    def _project_changed(self) -> None:
        self._invalidate_result()
        self._annotate()

    def _annotate(self) -> None:
        if self.project is not None:
            self.viewer.annotate(self.project.fixtures, self.project.loads, self.project.bolts)

    def _selection_changed(self, refs: list) -> None:
        panel = self.panels[self.nav.currentRow()]
        if isinstance(panel, steps._FaceListStep):
            panel.update_selection(refs)

    def readiness(self) -> list[tuple[bool, str]]:
        if self.project is None:
            return [(False, "모델을 여세요")]
        p = active_project(self.project)
        visible = p.parts
        if not visible:
            return [(False, "모든 부품이 숨겨져 있습니다. 1단계에서 부품을 체크하세요")]
        missing = [x.name for x in visible if x.material_id is None]
        checks = [
            (True, f"모델: 부품 {len(visible)}개"),
            (
                not missing,
                "재료: 모든 부품 지정됨" if not missing else f"재료 미지정: {', '.join(missing)}",
            ),
        ]
        thermal_bcs = [ld for ld in p.loads if ld.kind in THERMAL_LOADS]
        if p.analysis in (AnalysisKind.STATIC, AnalysisKind.BUCKLING):
            checks.append(
                (
                    bool(p.fixtures),
                    f"고정: {len(p.fixtures)}개" if p.fixtures else "고정이 없습니다",
                )
            )
            if p.analysis == AnalysisKind.STATIC and len(p.cases) > 1:
                empty = [c.name for c in p.cases if not c.loads]
                checks.append(
                    (
                        not empty,
                        f"하중: 케이스 {len(p.cases)}개 모두 계산 ({len(p.all_loads())}개)"
                        if not empty
                        else f"하중이 없는 케이스: {', '.join(empty)}",
                    )
                )
            else:
                mech = [ld for ld in p.loads if ld.kind not in THERMAL_LOADS]
                text = f"하중: {len(p.loads)}개" if p.loads else "하중이 없습니다"
                if len(p.cases) > 1:
                    text += f" ('{p.case.name}' 케이스만 사용)"
                checks.append((bool(mech or thermal_bcs), text))
        elif p.analysis == AnalysisKind.MODAL:
            checks.append(
                (
                    True,
                    f"고정: {len(p.fixtures)}개"
                    if p.fixtures
                    else "고정 없음: 공중에 떠 있는 상태의 진동수를 구합니다",
                )
            )
        else:
            has_ref = any(
                ld.kind in (LoadKind.FACE_TEMPERATURE, LoadKind.CONVECTION) for ld in thermal_bcs
            )
            checks.append(
                (
                    has_ref,
                    f"열 조건: {len(thermal_bcs)}개"
                    if has_ref
                    else "4단계에서 '고정 온도'나 '대류 냉각'을 하나 이상 지정하세요",
                )
            )
        excluded = excluded_by_hidden_parts(self.project)
        if excluded:
            checks.append((True, f"숨긴 부품에 지정된 {', '.join(excluded)}는 해석에서 제외합니다"))
        checks.append((True, "메시: 준비됨" if self.mesh else "메시: 해석할 때 자동으로 만듭니다"))
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
            panel.set_busy(False)
            self.mesh = mesh
            self._invalidate_result()
            panel.refresh()
            if self.nav.currentRow() == STEP_MESH:
                self.viewer.show_mesh(mesh if panel.show_mesh.isChecked() else None)

        def failed(e):
            self._busy = False
            panel.set_busy(False)
            self.warn(friendly_error(e))

        run_job(lambda progress: generate_mesh(project), done, failed)

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
        panel.set_progress(0.0, "시작")

        def job(progress):
            m = mesh
            if m is None:
                progress(0.02, "메시 만드는 중 (잠시 걸릴 수 있습니다)")
                m = generate_mesh(project)
            if kind == AnalysisKind.MODAL:
                return m, solve_modal(project, m, materials, project.n_modes, progress)
            if kind == AnalysisKind.THERMAL:
                return m, solve_thermal(project, m, materials, progress)
            if kind == AnalysisKind.BUCKLING:
                return m, solve_buckling(project, m, materials, project.n_modes, progress)
            return m, solve_cases(project, m, materials, progress)

        def done(value):
            self._busy = False
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

        def failed(e):
            self._busy = False
            panel.set_progress(None, "")
            log.warning("solve failed: %s: %s", type(e).__name__, e)
            self.warn(friendly_error(e))

        run_job(job, done, failed, lambda f, m: panel.set_progress(f, m))

    # --- results ---------------------------------------------------------------------

    def _configure_result_panel(self) -> None:
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        r = self.result
        if isinstance(r, ModalResult):
            panel.set_fields([("mode", "진동 모양 (상대 크기)")])
            panel.set_modes(r.frequencies)
            panel.animate.setChecked(True)
        elif isinstance(r, ThermalResult):
            panel.set_fields(THERMAL_FIELDS)
            panel.set_modes(None)
            panel.animate.setChecked(False)
        elif isinstance(r, BucklingResult):
            panel.set_fields([("mode", "좌굴 모양 (상대 크기)")])
            panel.set_modes(
                r.factors, [f"{i + 1}차: 하중 ×{f:,.2f}" for i, f in enumerate(r.factors)]
            )
            panel.animate.setChecked(True)
        else:
            fields = list(steps.FIELD_LABELS)
            if r.thermal is not None:
                fields += THERMAL_FIELDS
            if self.summary.fatigue is not None:
                fields += [("ffos", "피로 안전율"), ("life", "예상 수명 (반복 횟수)")]
            panel.set_fields(fields)
            panel.set_modes(None)
            panel.animate.setChecked(False)
        if self.case_results and len(self.case_results) > 1:
            labels = [
                f"{name} — {VERDICT_STYLE[s.verdict][0]} {s.governing_fos:.2f}"
                + (" ◀ 가장 불리" if self.overview and i == self.overview.worst else "")
                for i, (name, _, s) in enumerate(self.case_results)
            ]
            panel.set_cases(labels, self.project.active_case)
        else:
            panel.set_cases(None)
        has_shape = not isinstance(r, ThermalResult)
        panel.scale.setEnabled(has_shape)
        panel.animate.setEnabled(has_shape)

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
            title = "좌굴 모양 (상대값)" if isinstance(r, BucklingResult) else "진동 모양 (상대값)"
            return ResultField(mag / (mag.max() or 1.0), title, clim=(0, 1))
        if key in ("temp", "flux"):
            thermal = r if isinstance(r, ThermalResult) else r.thermal
            if key == "temp":
                return ResultField(thermal.temperature, "온도 (°C)", cmap="inferno", unit="°C")
            return ResultField(thermal.heat_flux_magnitude, "열 흐름 (W/m²)", "magma", None, "W/m²")
        if key in ("ffos", "life"):
            fatigue = self.summary.fatigue
            target = max(
                standards.effective_pack(self.project).target(m) for m in r.materials.values()
            )
            if key == "ffos":
                values = np.minimum(fatigue.safety_factor, 2 * target)
                return ResultField(values, "피로 안전율", cmap="RdYlGn", clim=(0, 2 * target))
            # log scale: 10^3 (red) ... 10^9 = infinite (green)
            return ResultField(
                np.log10(fatigue.life), "수명 (10의 거듭제곱 회)", "RdYlGn", (3, 9), "(log10 회)"
            )
        if key == "vm":
            # non-negative by definition: pin the colour scale at 0 so quadratic
            # interpolation overshoot does not show as a negative minimum
            vm = r.von_mises
            return ResultField(vm, "응력 (MPa)", clim=(0.0, float(vm.max())), unit="MPa")
        if key == "fos":
            pack = standards.effective_pack(self.project)
            target = max(pack.target(m) for m in r.materials.values())
            # clip before display: quadratic interpolation between a capped 100 and a small
            # neighbour would otherwise undershoot and paint false red spots
            values = np.minimum(r.safety_factor, 2 * target)
            return ResultField(values, "안전율", cmap="RdYlGn", clim=(0, 2 * target))
        if key == "disp":
            d = r.displacement_magnitude
            return ResultField(d, "변형량 (mm)", clim=(0.0, float(d.max())), unit="mm")
        if key in ("ux", "uy", "uz"):
            comp = r.displacement[:, "xyz".index(key[1])]
            lim = float(np.abs(comp).max()) or 1.0
            return ResultField(comp, f"변형량 {key[1].upper()} (mm)", "coolwarm", (-lim, lim), "mm")
        p1 = r.principal[:, 0]
        lim = float(np.abs(p1).max()) or 1.0
        return ResultField(p1, "최대 주응력 (MPa)", "coolwarm", (-lim, lim), "MPa")

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
            panel.scale_note.setText("변형을 실제 크기로 보여줍니다.")
        else:
            panel.scale_note.setText(
                f"변형을 {scale:,.0f}배 확대해서 보여줍니다. 회색 선은 변형 전 모양입니다."
            )
        self.viewer.show_result(
            self.mesh,
            self._displacement(),
            field,
            scale,
            symmetric=isinstance(self.result, (ModalResult, BucklingResult)),
        )
        self._fill_summary(panel)

    def _fill_summary(self, panel: steps.ResultStep) -> None:
        s = self.summary
        if s.verdict is None:
            color, bg = "#2b5d8a", "#eaf2fa"
        else:
            _, color, bg = VERDICT_STYLE[s.verdict]
        panel.card.setStyleSheet(
            f"QFrame#verdictCard {{ background: {bg}; border: 2px solid {color}; border-radius: 6px; }}"
        )
        panel.card_title.setText(
            f"<span style='font-size:15pt; color:{color}'><b>{s.headline}</b></span>"
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
                    f"{i + 1}차 <b>{f:,.1f} Hz</b> ({f * 60:,.0f} rpm)"
                    f" · 주 방향 질량 X {px:.0f}% Y {py:.0f}% Z {pz:.0f}%"
                )
            lines.append("<br>".join(rows))
        elif isinstance(r, BucklingResult):
            applied = np.linalg.norm(r.static.applied_load)
            rows = [
                f"{i + 1}차 하중 배수 <b>{f:,.2f}</b>"
                + (f" (약 {f * applied:,.0f} N에서 좌굴)" if applied > 0 else "")
                for i, f in enumerate(r.factors)
            ]
            lines.append("<br>".join(rows))
        elif isinstance(r, ThermalResult):
            for pid, m in self.materials.items():
                nodes = r.mesh.part_nodes(pid)
                limit = (
                    f" (권장 최고 {m.max_service_temperature:.0f} °C)"
                    if m.max_service_temperature
                    else ""
                )
                lines.append(
                    f"<b>{self.project.part(pid).name}</b>: "
                    f"{fmt(us, 'temperature', r.temperature[nodes].min(), '.1f')}"
                    f" ~ <b>{fmt(us, 'temperature', r.temperature[nodes].max(), '.1f')}</b>{limit}"
                )
            if r.heat_in:
                lines.append(f"들어온 열 {r.heat_in:.3g} W · 빠져나간 열 {r.heat_out:.3g} W")
        else:
            if self.overview is not None:
                o = self.overview
                lines.append(
                    f"<b>{o.headline}</b> ({o.summaries[o.worst].headline}). "
                    f"지금 보는 케이스: {self.project.case.name}"
                )
            for p in s.parts:
                lines.append(
                    f"<b>{p.name}</b> ({p.material})<br>"
                    f"&nbsp;최대 응력 {fmt(us, 'stress', p.max_stress, ',.1f')}"
                    f" / 강도 {fmt(us, 'stress', p.capacity, ',.0f')}<br>"
                    f"&nbsp;최소 안전율 <b>{p.min_fos:.2f}</b> (권장 {p.target:.1f})"
                    f" · 최대 변형 {fmt(us, 'length', p.max_disp)}"
                )
                if p.hotspot_fos is not None:
                    lines[-1] += (
                        f"<br>&nbsp;특이점({p.singular_kind}) 제외: 응력 "
                        f"{fmt(us, 'stress', p.hotspot_stress, ',.1f')}, 안전율 "
                        f"<b>{p.hotspot_fos:.2f}</b>"
                    )
                if p.fatigue_fos is not None:
                    life = "무한" if p.fatigue_life >= 1e9 else f"{p.fatigue_life:,.0f}회"
                    lines[-1] += (
                        f"<br>&nbsp;피로 안전율 <b>{p.fatigue_fos:.2f}</b> · 예상 수명 {life}"
                    )
            for b in r.bolts or []:
                lines.append(
                    f"<b>{b.name}</b> ({b.size} {b.grade}) 축력 {fmt(us, 'force', b.axial, ',.0f')}"
                    f" · 전단 {fmt(us, 'force', b.shear, ',.0f')}"
                    f" · 안전율 <b>{b.safety_factor:.2f}</b>"
                )
            if r.thermal is not None:
                lines.append(
                    f"열응력 포함: 온도 {r.thermal.temperature.min():.1f}"
                    f" ~ {r.thermal.temperature.max():.1f} °C"
                    f" (응력 없는 기준 온도 {self.project.reference_temperature:.0f} °C)"
                )
        for note in s.notes:
            lines.append(f"<span style='color:#8a6d3b'>※ {note}</span>")
        panel.summary.setText("<br><br>".join(lines))

    def save_report(self) -> None:
        if self.result is None:
            return
        from kstruct.report.pdf import build_html, write_pdf

        stem = self.project.source_path.stem if self.project.source_path else "kstruct"
        path, _ = QFileDialog.getSaveFileName(
            self, "보고서 저장", f"{stem}_보고서.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        images = self.capture_result_images()
        html_text = build_html(
            self.project, self.result, self.summary, images, self.materials, self.overview
        )
        write_pdf(html_text, path)
        self.statusBar().showMessage(f"보고서를 저장했습니다: {path}")

    def capture_result_images(self) -> list[tuple[str, bytes]]:
        """Render the main result fields to PNG for the report, then restore the view."""
        r = self.result
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        animated = panel.animate.isChecked()
        self.viewer.set_animated(False)
        shots: list[tuple[str, str, int | None]]
        if isinstance(r, ModalResult):
            shots = [
                ("mode", f"{i + 1}차 진동 모양 ({f:,.1f} Hz)", i)
                for i, f in enumerate(r.frequencies[:4])
            ]
        elif isinstance(r, BucklingResult):
            shots = [
                ("mode", f"{i + 1}차 좌굴 모양 (하중 ×{f:,.2f})", i)
                for i, f in enumerate(r.factors[:3])
            ]
        elif isinstance(r, ThermalResult):
            shots = [("temp", "온도 분포", None), ("flux", "열 흐름", None)]
        else:
            shots = [
                ("vm", "응력 분포", None),
                ("fos", "안전율 분포", None),
                ("disp", "변형량", None),
            ]
            if r.thermal is not None:
                shots.append(("temp", "온도 분포", None))
            if self.summary.fatigue is not None:
                shots.append(("ffos", "피로 안전율 분포", None))
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
        panel.mode.blockSignals(True)
        panel.mode.setCurrentIndex(current_mode)
        panel.mode.blockSignals(False)
        self.show_result()
        self.viewer.set_animated(animated)
        return images

    # --- dialogs ---------------------------------------------------------------------

    def warn(self, message: str) -> None:
        QMessageBox.warning(self, "확인이 필요합니다", message)

    def _confirm(self, message: str) -> bool:
        answer = QMessageBox.question(self, "확인", message)
        return answer == QMessageBox.StandardButton.Yes


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
