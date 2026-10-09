"""Main window: wizard (left), 3D viewer (center). Owns the Project, mesh and result."""

from __future__ import annotations

import tempfile
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
from kstruct.fem import ConstraintError, MeshingError, SolverError, generate_mesh, solve_static
from kstruct.fem.mesh import FEMesh
from kstruct.fem.static import StaticResult
from kstruct.geometry import GeometryImportError, load_cad
from kstruct.geometry.contacts import find_touching_faces
from kstruct.geometry.samples import l_bracket, two_plates, write_step
from kstruct.model import Project
from kstruct.project_io import SUFFIX, ProjectFileError, load_project, save_project
from kstruct.report.summary import VERDICT_STYLE, Summary, summarize
from kstruct.ui import steps
from kstruct.ui.help import HelpButton, HelpDialog, topic, with_help
from kstruct.ui.viewer import ModelViewer, ResultField
from kstruct.ui.worker import run_job

CAD_FILTER = "3D CAD 파일 (*.step *.stp *.iges *.igs);;모든 파일 (*)"
PROJECT_FILTER = f"KStruct 프로젝트 (*{SUFFIX})"
SAMPLES = {"bracket": l_bracket, "two_plates": two_plates}

STEP_OPEN, STEP_MATERIAL, STEP_FIXTURE, STEP_LOAD, STEP_MESH, STEP_SOLVE, STEP_RESULT = range(7)
STEP_NAMES = ["1. 모델 열기", "2. 재료", "3. 고정", "4. 하중", "5. 메시", "6. 해석", "7. 결과"]
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
        self.result: StaticResult | None = None
        self.summary: Summary | None = None
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
        try:
            project = load_cad(path)
        except GeometryImportError as e:
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
        found = find_touching_faces(self.project)
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
        self._update_nav()

    def _project_changed(self) -> None:
        self._invalidate_result()
        self._annotate()

    def _annotate(self) -> None:
        if self.project is not None:
            self.viewer.annotate(self.project.fixtures, self.project.loads)

    def _selection_changed(self, refs: list) -> None:
        panel = self.panels[self.nav.currentRow()]
        if isinstance(panel, steps._FaceListStep):
            panel.update_selection(refs)

    def readiness(self) -> list[tuple[bool, str]]:
        p = self.project
        if p is None:
            return [(False, "모델을 여세요")]
        visible = [x for x in p.parts if x.visible]
        missing = [x.name for x in visible if x.material_id is None]
        return [
            (True, f"모델: 부품 {len(visible)}개"),
            (
                not missing,
                "재료: 모든 부품 지정됨" if not missing else f"재료 미지정: {', '.join(missing)}",
            ),
            (bool(p.fixtures), f"고정: {len(p.fixtures)}개" if p.fixtures else "고정이 없습니다"),
            (bool(p.loads), f"하중: {len(p.loads)}개" if p.loads else "하중이 없습니다"),
            (True, "메시: 준비됨" if self.mesh else "메시: 해석할 때 자동으로 만듭니다"),
        ]

    # --- meshing & solving -----------------------------------------------------------

    def build_mesh(self) -> None:
        if self.project is None or self._busy:
            return
        panel: steps.MeshStep = self.panels[STEP_MESH]
        panel.set_busy(True)
        self._busy = True
        project = self.project

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
        project, mesh = self.project, self.mesh
        mats = standards.materials()
        materials = {p.id: mats[p.material_id] for p in project.parts if p.visible}
        self._busy = True
        panel.set_progress(0.0, "시작")

        def job(progress):
            m = mesh
            if m is None:
                progress(0.02, "메시 만드는 중 (잠시 걸릴 수 있습니다)")
                m = generate_mesh(project)
            return m, solve_static(project, m, materials, progress)

        def done(value):
            self._busy = False
            self.mesh, self.result = value
            self.summary = summarize(project, self.result, standards.pack(project.standard))
            panel.set_progress(None, "")
            self._update_nav()
            self._go(STEP_RESULT)

        def failed(e):
            self._busy = False
            panel.set_progress(None, "")
            self.warn(friendly_error(e))

        run_job(job, done, failed, lambda f, m: panel.set_progress(f, m))

    # --- results ---------------------------------------------------------------------

    def result_field(self, key: str) -> ResultField:
        r = self.result
        if key == "vm":
            return ResultField(r.von_mises, "응력 (MPa)", unit="MPa")
        if key == "fos":
            pack = standards.pack(self.project.standard)
            target = max(pack.target(m) for m in r.materials.values())
            # clip before display: quadratic interpolation between a capped 100 and a small
            # neighbour would otherwise undershoot and paint false red spots
            values = np.minimum(r.safety_factor, 2 * target)
            return ResultField(values, "안전율", cmap="RdYlGn", clim=(0, 2 * target))
        if key == "disp":
            return ResultField(r.displacement_magnitude, "변형량 (mm)", unit="mm")
        if key in ("ux", "uy", "uz"):
            comp = r.displacement[:, "xyz".index(key[1])]
            lim = float(np.abs(comp).max()) or 1.0
            return ResultField(comp, f"변형량 {key[1].upper()} (mm)", "coolwarm", (-lim, lim), "mm")
        p1 = r.principal[:, 0]
        lim = float(np.abs(p1).max()) or 1.0
        return ResultField(p1, "최대 주응력 (MPa)", "coolwarm", (-lim, lim), "MPa")

    def deformation_scale(self, key: str) -> float:
        if key == "none":
            return 0.0
        if key == "true":
            return 1.0
        size = float(np.ptp(self.mesh.nodes, axis=0).max())
        peak = float(self.result.displacement_magnitude.max())
        return 0.05 * size / peak if peak > 0 else 0.0

    def show_result(self) -> None:
        if self.result is None or self.nav.currentRow() != STEP_RESULT:
            return
        panel: steps.ResultStep = self.panels[STEP_RESULT]
        field = self.result_field(panel.field_key())
        scale = self.deformation_scale(panel.scale_key())
        if scale not in (0.0, 1.0):
            field.title += f"  [변형 ×{scale:,.0f}]"
        self.viewer.show_result(self.mesh, self.result.displacement, field, scale)
        self._fill_summary(panel)

    def _fill_summary(self, panel: steps.ResultStep) -> None:
        s = self.summary
        _, color, bg = VERDICT_STYLE[s.verdict]
        panel.card.setStyleSheet(
            f"QFrame#verdictCard {{ background: {bg}; border: 2px solid {color}; border-radius: 6px; }}"
        )
        panel.card_title.setText(
            f"<span style='font-size:15pt; color:{color}'><b>{s.headline}</b></span>"
        )
        panel.card_body.setText(s.explanation)
        lines = []
        for p in s.parts:
            lines.append(
                f"<b>{p.name}</b> ({p.material})<br>"
                f"&nbsp;최대 응력 {p.max_stress:,.1f} MPa / 강도 {p.capacity:,.0f} MPa<br>"
                f"&nbsp;최소 안전율 <b>{p.min_fos:.2f}</b> (권장 {p.target:.1f})"
                f" · 최대 변형 {p.max_disp:.4g} mm"
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
        write_pdf(build_html(self.project, self.result, self.summary, images), path)
        self.statusBar().showMessage(f"보고서를 저장했습니다: {path}")

    def capture_result_images(self) -> list[tuple[str, bytes]]:
        """Render the main result fields to PNG for the report, then restore the view."""
        images = []
        scale = self.deformation_scale("auto")
        for key, caption in (("vm", "응력 분포"), ("fos", "안전율 분포"), ("disp", "변형량")):
            field = self.result_field(key)
            self.viewer.show_result(self.mesh, self.result.displacement, field, scale)
            images.append((caption, _png(self.viewer.plotter.screenshot(return_img=True))))
        self.show_result()
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
