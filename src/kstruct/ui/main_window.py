"""Main window: wizard steps (left), 3D viewer (center), parts list."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kstruct.geometry import GeometryImportError, faces_of, load_cad
from kstruct.geometry.face_info import face_info
from kstruct.geometry.samples import l_bracket, two_plates, write_step
from kstruct.model import FaceRef, Project
from kstruct.ui.help import HelpButton, HelpDialog, topic, with_help
from kstruct.ui.viewer import ModelViewer

# (help key, label); help key doubles as the step id
WIZARD_STEPS = [
    ("step.open", "1. 모델 열기"),
    ("step.material", "2. 재료"),
    ("step.fixture", "3. 고정"),
    ("step.load", "4. 하중"),
    ("step.mesh", "5. 메시"),
    ("step.solve", "6. 해석"),
    ("step.result", "7. 결과"),
]
CAD_FILTER = "3D CAD 파일 (*.step *.stp *.iges *.igs);;모든 파일 (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KStruct — 부품 강도 해석")
        self.resize(1400, 860)
        self.project: Project | None = None

        self.viewer = ModelViewer()
        self.viewer.face_picked.connect(self._on_face_picked)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_side_panel())
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1120])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self.statusBar().showMessage("'열기'로 STEP 파일을 불러오거나 예제를 눌러 보세요.")

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(with_help(QLabel("<b>작업 단계</b>"), "help.mode"))
        self.steps = QListWidget()
        for key, label in WIZARD_STEPS:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(topic(key).short + "  (더블클릭: 자세한 설명)")
            if key != "step.open":
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.steps.addItem(item)
        self.steps.setCurrentRow(0)
        self.steps.itemDoubleClicked.connect(
            lambda item: HelpDialog(item.data(Qt.ItemDataRole.UserRole), self).exec()
        )
        self.steps.setMaximumHeight(200)
        layout.addWidget(self.steps)

        layout.addWidget(with_help(QLabel("<b>부품</b>"), "view.parts"))
        self.parts = QListWidget()
        self.parts.itemChanged.connect(self._on_part_toggled)
        layout.addWidget(self.parts, 1)

        layout.addWidget(with_help(QLabel("<b>선택한 면</b>"), "view.face_pick"))
        self.face_label = QLabel("모델의 면을 클릭하세요.")
        self.face_label.setWordWrap(True)
        layout.addWidget(self.face_label)
        return panel

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("기본")
        bar.setMovable(False)

        open_action = QAction("열기", self, shortcut=QKeySequence.StandardKey.Open)
        open_action.setToolTip(topic("step.open").short)
        open_action.triggered.connect(self.open_file)
        bar.addAction(open_action)
        bar.addWidget(HelpButton("step.open"))
        cad_help = QAction("원본 CAD 파일 여는 법", self)
        cad_help.setToolTip(topic("file.export_step").short)
        cad_help.triggered.connect(lambda: HelpDialog("file.export_step", self).exec())
        bar.addAction(cad_help)
        bar.addSeparator()

        for label, factory in (("예제: 브래킷", l_bracket), ("예제: 판 2장", two_plates)):
            action = QAction(label, self)
            action.setToolTip("연습용 예제 모델을 엽니다.")
            action.triggered.connect(lambda _=False, f=factory, n=label: self._open_sample(f, n))
            bar.addAction(action)
        bar.addSeparator()

        fit_action = QAction("화면 맞춤", self, shortcut="F")
        fit_action.setToolTip(topic("view.fit").short)
        fit_action.triggered.connect(self.viewer.fit)
        bar.addAction(fit_action)
        bar.addWidget(HelpButton("view.fit"))

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "모델 열기", "", CAD_FILTER)
        if path:
            self.load(Path(path))

    def _open_sample(self, factory, label: str) -> None:
        name = "bracket" if factory is l_bracket else "two_plates"
        path = write_step(factory(), Path(tempfile.gettempdir()) / f"kstruct_{name}.step")
        self.load(path)

    def load(self, path: Path) -> None:
        try:
            project = load_cad(path)
        except GeometryImportError as e:
            QMessageBox.warning(self, "열 수 없음", str(e))
            return
        self.project = project
        self.viewer.load_project(project)

        self.parts.blockSignals(True)
        self.parts.clear()
        for part in project.parts:
            item = QListWidgetItem(f"{part.name}  (면 {len(faces_of(part.shape))}개)")
            item.setData(Qt.ItemDataRole.UserRole, part.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.parts.addItem(item)
        self.parts.blockSignals(False)

        self.face_label.setText("모델의 면을 클릭하세요.")
        self.statusBar().showMessage(f"{path.name}: 부품 {len(project.parts)}개를 불러왔습니다.")

    def _on_part_toggled(self, item: QListWidgetItem) -> None:
        visible = item.checkState() == Qt.CheckState.Checked
        self.viewer.set_part_visible(item.data(Qt.ItemDataRole.UserRole), visible)

    def _on_face_picked(self, ref: FaceRef | None) -> None:
        if ref is None or self.project is None:
            self.face_label.setText("모델의 면을 클릭하세요.")
            return
        part = self.project.part(ref.part_id)
        info = face_info(faces_of(part.shape)[ref.face_index])
        self.face_label.setText(f"{part.name}<br>면 #{ref.face_index + 1}: {info.describe()}")
