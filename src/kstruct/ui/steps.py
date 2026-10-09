"""The seven wizard step panels. Each panel edits the shared Project through the
MainWindow (passed in as `app`) and emits `changed` when the project was modified."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from kstruct import standards
from kstruct.geometry import faces_of
from kstruct.geometry.face_info import face_info, outward_normal
from kstruct.model import GRAVITY, FaceRef, Fixture, FixtureKind, Load, LoadKind
from kstruct.ui.help import HelpButton, HelpDialog, topic, with_help

if TYPE_CHECKING:
    from kstruct.ui.main_window import MainWindow

FIXTURE_LABELS = {
    FixtureKind.FIXED: ("완전 고정", "fixture.fixed"),
    FixtureKind.ROLLER: ("미끄럼 지지", "fixture.roller"),
    FixtureKind.PIN: ("핀 지지 (구멍·축)", "fixture.pin"),
}
LOAD_LABELS = {
    LoadKind.FORCE: ("힘", "load.force"),
    LoadKind.PRESSURE: ("압력", "load.pressure"),
    LoadKind.GRAVITY: ("자중 (중력)", "load.gravity"),
    LoadKind.TEMPERATURE: ("온도 변화", "load.temperature"),
}
DIRECTIONS = [
    ("면을 수직으로 누르기", None, 1.0),
    ("면을 수직으로 당기기", None, -1.0),
    ("+X 방향", (1, 0, 0), 1.0),
    ("−X 방향", (-1, 0, 0), 1.0),
    ("+Y 방향", (0, 1, 0), 1.0),
    ("−Y 방향", (0, -1, 0), 1.0),
    ("+Z 방향", (0, 0, 1), 1.0),
    ("−Z 방향 (아래)", (0, 0, -1), 1.0),
]


def _title(text: str, key: str) -> QWidget:
    label = QLabel(f"<h3 style='margin:0'>{text}</h3>")
    return with_help(label, key)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


def _row(*widgets: QWidget) -> QWidget:
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        row.addWidget(w)
    return box


class StepPanel(QWidget):
    changed = Signal()

    def __init__(self, app: MainWindow, title: str, key: str) -> None:
        super().__init__()
        self.app = app
        self.layout_ = QVBoxLayout(self)
        self.layout_.addWidget(_title(title, key))
        self.layout_.addWidget(_hint(topic(key).short))

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.layout_.addWidget(widget, stretch)
        return widget

    def refresh(self) -> None:
        """Re-read the project into the widgets."""

    def describe_faces(self, refs: list[FaceRef]) -> str:
        project = self.app.project
        names = []
        for ref in refs:
            part = project.part(ref.part_id)
            kind = face_info(faces_of(part.shape)[ref.face_index]).kind
            prefix = f"{part.name} " if len(project.parts) > 1 else ""
            names.append(f"{prefix}면#{ref.face_index + 1}({kind})")
        return ", ".join(names)


# --- 1. open ----------------------------------------------------------------------


class OpenStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "1. 모델 열기", "step.open")
        open_btn = QPushButton("파일 열기…")
        open_btn.clicked.connect(app.open_file)
        self.add(with_help(open_btn, "step.open"))
        cad_help = QPushButton("SolidWorks·Inventor·CATIA 파일은?")
        cad_help.setFlat(True)
        cad_help.clicked.connect(lambda: HelpDialog("file.export_step", self.window()).exec())
        self.add(with_help(cad_help, "file.export_step"))
        mesh_help = QPushButton("STL·FBX 등 메시 파일은?")
        mesh_help.setFlat(True)
        mesh_help.clicked.connect(lambda: HelpDialog("file.mesh_formats", self.window()).exec())
        self.add(with_help(mesh_help, "file.mesh_formats"))

        samples = QHBoxLayout()
        for label, name in (("예제: 브래킷", "bracket"), ("예제: 판 2장", "two_plates")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, n=name: app.open_sample(n))
            samples.addWidget(b)
        box = QWidget()
        box.setLayout(samples)
        self.add(box)

        self.add(with_help(QLabel("<b>부품</b>"), "view.parts"))
        self.parts = QListWidget()
        self.parts.itemChanged.connect(self._toggled)
        self.add(self.parts, 1)

        self.conn_box = QGroupBox("부품 사이 연결")
        conn = QVBoxLayout(self.conn_box)
        conn.addWidget(_hint(topic("connection.bonded").short))
        detect = QPushButton("맞닿은 면 자동으로 찾기")
        detect.clicked.connect(app.detect_connections)
        conn.addWidget(with_help(detect, "connection.bonded"))
        self.connections = QListWidget()
        self.connections.setMaximumHeight(110)
        conn.addWidget(self.connections)
        remove = QPushButton("선택한 연결 삭제")
        remove.clicked.connect(self._remove_connection)
        conn.addWidget(remove)
        self.add(self.conn_box)

    def refresh(self) -> None:
        project = self.app.project
        self.parts.blockSignals(True)
        self.parts.clear()
        for part in project.parts if project else []:
            item = QListWidgetItem(f"{part.name}  (면 {len(faces_of(part.shape))}개)")
            item.setData(Qt.ItemDataRole.UserRole, part.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if part.visible else Qt.CheckState.Unchecked)
            self.parts.addItem(item)
        self.parts.blockSignals(False)
        self.conn_box.setVisible(bool(project and len(project.parts) > 1))
        self.connections.clear()
        for c in project.connections if project else []:
            self.connections.addItem(
                f"접합: {self.describe_faces(c.faces_a)} ↔ {self.describe_faces(c.faces_b)}"
            )

    def _toggled(self, item: QListWidgetItem) -> None:
        part = self.app.project.part(item.data(Qt.ItemDataRole.UserRole))
        part.visible = item.checkState() == Qt.CheckState.Checked
        self.app.viewer.set_part_visible(part.id, part.visible)
        self.app.invalidate_mesh()

    def _remove_connection(self) -> None:
        row = self.connections.currentRow()
        if row >= 0:
            del self.app.project.connections[row]
            self.app.invalidate_mesh()
            self.refresh()


# --- 2. material --------------------------------------------------------------------


class MaterialStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "2. 재료", "step.material")
        self.standard = QComboBox()
        for p in standards.packs().values():
            self.standard.addItem(p.label, p.id)
        self.standard.currentIndexChanged.connect(self._standard_changed)
        form = QFormLayout()
        form.addRow("적용 기준", with_help(self.standard, "material.standard"))
        box = QWidget()
        box.setLayout(form)
        self.add(box)

        self.add(QLabel("<b>부품을 고르고 재료를 지정하세요</b>"))
        self.parts = QListWidget()
        self.parts.setMaximumHeight(120)
        self.parts.currentRowChanged.connect(lambda _: self._show_part())
        self.add(self.parts)

        self.material = QComboBox()
        self.material.setMaxVisibleItems(25)
        self.material.activated.connect(self._material_chosen)
        self.add(with_help(self.material, "material.choose"))
        self.apply_all = QPushButton("모든 부품에 같은 재료 적용")
        self.apply_all.clicked.connect(self._apply_all)
        self.add(self.apply_all)

        self.props = QLabel()
        self.props.setWordWrap(True)
        self.props.setTextFormat(Qt.TextFormat.RichText)
        self.add(with_help(self.props, "material.properties"))
        self.layout_.addStretch(1)

    def _pack(self):
        return standards.pack(self.standard.currentData())

    def refresh(self) -> None:
        project = self.app.project
        self.standard.blockSignals(True)
        self.standard.setCurrentIndex(max(0, self.standard.findData(project.standard)))
        self.standard.blockSignals(False)
        self._fill_materials()
        row = max(0, self.parts.currentRow())
        self.parts.blockSignals(True)
        self.parts.clear()
        mats = standards.materials()
        for part in project.parts:
            m = mats.get(part.material_id)
            status = self._pack().material_name(m) if m else "⚠ 재료 미지정"
            self.parts.addItem(f"{part.name} — {status}")
        self.parts.blockSignals(False)
        self.parts.setCurrentRow(min(row, len(project.parts) - 1))
        self.apply_all.setVisible(len(project.parts) > 1)
        self._show_part()

    def _fill_materials(self) -> None:
        pack = self._pack()
        self.material.clear()
        self.material.addItem("— 재료를 선택하세요 —", None)
        by_cat: dict[str, list] = {}
        for m in standards.materials().values():
            by_cat.setdefault(m.category, []).append(m)
        for cat, mats in by_cat.items():
            self.material.addItem(f"[{standards.CATEGORY_NAMES[cat]}]", None)
            idx = self.material.count() - 1
            self.material.model().item(idx).setEnabled(False)
            for m in mats:
                self.material.addItem(f"   {pack.material_name(m)} — {m.name}", m.id)

    def _current_part(self):
        row = self.parts.currentRow()
        return self.app.project.parts[row] if row >= 0 else None

    def _show_part(self) -> None:
        part = self._current_part()
        if part is None:
            return
        self.material.setCurrentIndex(max(0, self.material.findData(part.material_id)))
        m = standards.materials().get(part.material_id)
        if m is None:
            self.props.setText("")
            return
        pack = self._pack()
        kind = "연성재 (휘다가 파손)" if m.ductile else "취성재 (갑자기 깨짐)"
        names = " · ".join(f"{k.upper()} {v}" for k, v in m.designations.items() if k != "iso")
        self.props.setText(
            f"<b>{pack.material_name(m)}</b> — {m.name}<br>"
            f"<small>{names}</small><br><br>"
            f"단단함(탄성계수): <b>{m.youngs_modulus / 1000:,.0f} GPa</b><br>"
            f"항복강도: <b>{m.yield_strength:,.0f} MPa</b> (이 이상이면 영구 변형)<br>"
            f"인장강도: <b>{m.tensile_strength:,.0f} MPa</b> (이 이상이면 파단)<br>"
            f"밀도: {m.density * 1e9:.2f} g/cm³<br>"
            f"성질: {kind}<br>"
            f"권장 안전율({pack.label}): {pack.target(m):.1f} 이상"
        )

    def _material_chosen(self) -> None:
        part = self._current_part()
        mid = self.material.currentData()
        if part is None or mid is None:
            return
        part.material_id = mid
        self.changed.emit()
        self.refresh()

    def _apply_all(self) -> None:
        mid = self.material.currentData()
        if mid is None:
            return
        for part in self.app.project.parts:
            part.material_id = mid
        self.changed.emit()
        self.refresh()

    def _standard_changed(self) -> None:
        self.app.project.standard = self.standard.currentData()
        self.changed.emit()
        self.refresh()


# --- 3. fixtures ----------------------------------------------------------------------


class _FaceListStep(StepPanel):
    """Shared layout for steps that attach things to selected faces."""

    def __init__(self, app: MainWindow, title: str, key: str) -> None:
        super().__init__(app, title, key)
        self.selection_label = _hint("")
        self.items = QListWidget()
        self.items.setMinimumHeight(120)
        self.remove = QPushButton("선택한 항목 삭제")
        self.remove.clicked.connect(self._remove)

    def finish_layout(self) -> None:
        self.add(self.selection_label)
        self.add(QLabel("<b>지정된 항목</b>"))
        self.add(self.items, 1)
        self.add(self.remove)

    def update_selection(self, refs: list[FaceRef]) -> None:
        if refs:
            self.selection_label.setText(f"선택한 면: {self.describe_faces(refs)}")
        else:
            self.selection_label.setText("3D 화면에서 면을 클릭하세요. Ctrl+클릭으로 여러 면 선택.")

    def entries(self) -> list:
        raise NotImplementedError

    def describe(self, entry) -> str:
        raise NotImplementedError

    def refresh(self) -> None:
        self.items.clear()
        for e in self.entries():
            self.items.addItem(self.describe(e))
        self.update_selection(self.app.viewer.selection)

    def _remove(self) -> None:
        row = self.items.currentRow()
        if row >= 0:
            del self.entries()[row]
            self.changed.emit()
            self.refresh()


class FixtureStep(_FaceListStep):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "3. 고정", "step.fixture")
        self.kinds = QButtonGroup(self)
        for i, (kind, (label, key)) in enumerate(FIXTURE_LABELS.items()):
            radio = QRadioButton(label)
            radio.setChecked(i == 0)
            self.kinds.addButton(radio, i)
            radio.setProperty("kind", kind)
            self.add(with_help(radio, key))
        self.add_btn = QPushButton("선택한 면을 고정")
        self.add_btn.clicked.connect(self._add)
        self.add(self.add_btn)
        self.finish_layout()

    def entries(self) -> list:
        return self.app.project.fixtures

    def describe(self, fx: Fixture) -> str:
        return f"{FIXTURE_LABELS[fx.kind][0]}: {self.describe_faces(fx.faces)}"

    def update_selection(self, refs: list[FaceRef]) -> None:
        super().update_selection(refs)
        self.add_btn.setEnabled(bool(refs))

    def _add(self) -> None:
        refs = self.app.viewer.selection
        kind = self.kinds.checkedButton().property("kind")
        if kind == FixtureKind.PIN:
            for ref in refs:
                if (
                    face_info(
                        faces_of(self.app.project.part(ref.part_id).shape)[ref.face_index]
                    ).radius
                    is None
                ):
                    self.app.warn(
                        "핀 지지는 원통면(구멍 안쪽이나 축 바깥면)에만 지정할 수 있습니다."
                    )
                    return
        self.app.project.fixtures.append(Fixture(kind, refs))
        self.app.viewer.set_selection([])
        self.changed.emit()
        self.refresh()


# --- 4. loads -------------------------------------------------------------------------


class LoadStep(_FaceListStep):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "4. 하중", "step.load")
        self.kind = QComboBox()
        for kind, (label, _) in LOAD_LABELS.items():
            self.kind.addItem(label, kind)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.kind_help = HelpButton("load.force")
        self.add(_row(self.kind, self.kind_help))

        form = QFormLayout()
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e9, 1e9)
        self.value.setDecimals(2)
        self.value.setValue(1000)
        self.value.valueChanged.connect(self._update_hint)
        self.value_label = QLabel("크기")
        form.addRow(self.value_label, self.value)
        self.direction = QComboBox()
        for label, _, _ in DIRECTIONS:
            self.direction.addItem(label)
        self.direction_label = QLabel("방향")
        form.addRow(self.direction_label, with_help(self.direction, "load.direction"))
        box = QWidget()
        box.setLayout(form)
        self.add(box)
        self.value_hint = _hint("")
        self.add(self.value_hint)

        self.add_btn = QPushButton("하중 추가")
        self.add_btn.clicked.connect(self._add)
        self.add(self.add_btn)
        self.finish_layout()
        self._kind_changed()

    def _current_kind(self) -> LoadKind:
        return self.kind.currentData()

    def _kind_changed(self) -> None:
        kind = self._current_kind()
        self.kind_help.key = LOAD_LABELS[kind][1]
        self.kind_help.setToolTip(topic(self.kind_help.key).short)
        needs_faces = kind in (LoadKind.FORCE, LoadKind.PRESSURE)
        self.selection_label.setVisible(needs_faces)
        self.direction.setVisible(kind in (LoadKind.FORCE, LoadKind.GRAVITY))
        self.direction_label.setVisible(self.direction.isVisible())
        units = {
            LoadKind.FORCE: ("크기 (N)", 1000.0),
            LoadKind.PRESSURE: ("압력 (MPa)", 1.0),
            LoadKind.GRAVITY: ("배수 (g)", 1.0),
            LoadKind.TEMPERATURE: ("온도 변화 (°C)", 50.0),
        }
        label, default = units[kind]
        self.value_label.setText(label)
        self.value.setValue(default)
        if kind == LoadKind.GRAVITY:
            self.direction.setCurrentIndex(len(DIRECTIONS) - 1)
        self._update_hint()
        self.update_selection(self.app.viewer.selection)

    def _update_hint(self) -> None:
        v = self.value.value()
        kind = self._current_kind()
        hints = {
            LoadKind.FORCE: f"≈ {v / 9.81:,.1f} kg의 물체를 올려놓은 무게와 같습니다.",
            LoadKind.PRESSURE: f"= {v * 10:,.2f} bar (자동차 타이어는 약 2.4 bar)",
            LoadKind.GRAVITY: "1 g = 지구 중력. 충격·가속을 고려하려면 2~3 g를 입력하세요.",
            LoadKind.TEMPERATURE: "모든 부품이 고르게 이만큼 데워진(+) 또는 식은(−) 경우입니다.",
        }
        self.value_hint.setText(hints[kind])

    def update_selection(self, refs: list[FaceRef]) -> None:
        super().update_selection(refs)
        needs_faces = self._current_kind() in (LoadKind.FORCE, LoadKind.PRESSURE)
        self.add_btn.setEnabled(bool(refs) or not needs_faces)

    def entries(self) -> list:
        return self.app.project.loads

    def describe(self, ld: Load) -> str:
        if ld.kind == LoadKind.FORCE:
            fx, fy, fz = ld.vector
            mag = (fx**2 + fy**2 + fz**2) ** 0.5
            return f"힘 {mag:,.1f} N ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}): {self.describe_faces(ld.faces)}"
        if ld.kind == LoadKind.PRESSURE:
            return f"압력 {ld.magnitude:,.3g} MPa: {self.describe_faces(ld.faces)}"
        if ld.kind == LoadKind.GRAVITY:
            g = (sum(c * c for c in ld.vector) ** 0.5) / GRAVITY
            return f"자중 {g:.2g} g"
        return f"온도 변화 {ld.magnitude:+.1f} °C"

    def _add(self) -> None:
        kind = self._current_kind()
        v = self.value.value()
        refs = self.app.viewer.selection
        project = self.app.project
        if kind == LoadKind.FORCE:
            _, axis, sign = DIRECTIONS[self.direction.currentIndex()]
            if axis is None:
                normals = [
                    outward_normal(faces_of(project.part(r.part_id).shape)[r.face_index])
                    for r in refs
                ]
                if any(n is None for n in normals) or len(set(normals)) > 1:
                    # curved or differently oriented faces: push/pull as an equivalent pressure
                    area = sum(
                        face_info(faces_of(project.part(r.part_id).shape)[r.face_index]).area
                        for r in refs
                    )
                    load = Load(LoadKind.PRESSURE, refs, magnitude=sign * v / area)
                else:
                    n = normals[0]
                    load = Load(LoadKind.FORCE, refs, vector=tuple(-sign * v * c + 0.0 for c in n))
            else:
                load = Load(LoadKind.FORCE, refs, vector=tuple(v * c + 0.0 for c in axis))
        elif kind == LoadKind.PRESSURE:
            load = Load(LoadKind.PRESSURE, refs, magnitude=v)
        elif kind == LoadKind.GRAVITY:
            _, axis, _ = DIRECTIONS[self.direction.currentIndex()]
            axis = axis or (0, 0, -1)
            project.loads[:] = [ld for ld in project.loads if ld.kind != LoadKind.GRAVITY]
            load = Load(LoadKind.GRAVITY, vector=tuple(v * GRAVITY * c for c in axis))
        else:
            project.loads[:] = [ld for ld in project.loads if ld.kind != LoadKind.TEMPERATURE]
            load = Load(LoadKind.TEMPERATURE, magnitude=v)
        project.loads.append(load)
        if kind in (LoadKind.FORCE, LoadKind.PRESSURE):
            self.app.viewer.set_selection([])
        self.changed.emit()
        self.refresh()


# --- 5. mesh --------------------------------------------------------------------------


QUALITY_LABELS = [
    ("coarse", "거침 (빠름, 대략적인 결과)"),
    ("normal", "보통 (권장)"),
    ("fine", "세밀 (정확, 느림)"),
    ("very_fine", "매우 세밀 (작은 구멍·모서리 집중 확인)"),
]


class MeshStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "5. 메시", "step.mesh")
        self.quality = QComboBox()
        for key, label in QUALITY_LABELS:
            self.quality.addItem(label, key)
        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.add(with_help(self.quality, "mesh.quality"))
        self.run = QPushButton("메시 만들기")
        self.run.clicked.connect(app.build_mesh)
        self.add(self.run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.add(self.progress)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.add(self.info)
        self.show_mesh = QCheckBox("메시 선 보기")
        self.show_mesh.setChecked(True)
        self.show_mesh.toggled.connect(lambda on: app.viewer.show_mesh(app.mesh if on else None))
        self.add(self.show_mesh)
        self.layout_.addStretch(1)

    def refresh(self) -> None:
        project = self.app.project
        self.quality.blockSignals(True)
        self.quality.setCurrentIndex(max(0, self.quality.findData(project.mesh_quality)))
        self.quality.blockSignals(False)
        mesh = self.app.mesh
        if mesh is None:
            self.info.setText("아직 메시가 없습니다. '메시 만들기'를 누르세요.")
        else:
            self.info.setText(
                f"✔ 메시 완료: 요소 {len(mesh.tets):,}개, 절점 {mesh.n_nodes:,}개 "
                f"(계산할 미지수 {mesh.n_nodes * 3:,}개)"
            )

    def _quality_changed(self) -> None:
        self.app.project.mesh_quality = self.quality.currentData()
        self.app.invalidate_mesh()
        self.refresh()

    def set_busy(self, busy: bool) -> None:
        self.run.setEnabled(not busy)
        self.progress.setVisible(busy)


# --- 6. solve -------------------------------------------------------------------------


class SolveStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "6. 해석", "step.solve")
        self.checklist = QLabel()
        self.checklist.setTextFormat(Qt.TextFormat.RichText)
        self.add(self.checklist)
        self.run = QPushButton("해석 시작")
        self.run.setMinimumHeight(40)
        self.run.setStyleSheet("font-weight: bold;")
        self.run.clicked.connect(app.solve)
        self.add(self.run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.add(self.progress)
        self.status = _hint("")
        self.add(self.status)
        self.layout_.addStretch(1)

    def refresh(self) -> None:
        problems = self.app.readiness()
        lines = []
        for ok, text in problems:
            mark = (
                "<span style='color:#2e9e5b'>✔</span>"
                if ok
                else "<span style='color:#c0392b'>✘</span>"
            )
            lines.append(f"{mark} {text}")
        self.checklist.setText("<br>".join(lines))
        self.run.setEnabled(all(ok for ok, _ in problems))

    def set_progress(self, frac: float | None, msg: str = "") -> None:
        busy = frac is not None
        self.progress.setVisible(busy)
        self.run.setEnabled(not busy)
        if busy:
            self.progress.setValue(int(frac * 100))
        self.status.setText(msg)


# --- 7. results -----------------------------------------------------------------------


FIELD_LABELS = [
    ("vm", "응력 (폰 미세스)"),
    ("fos", "안전율"),
    ("disp", "변형량 (합계)"),
    ("ux", "변형량 X"),
    ("uy", "변형량 Y"),
    ("uz", "변형량 Z"),
    ("p1", "최대 주응력 (인장)"),
]
SCALE_LABELS = [("auto", "자동 확대"), ("true", "실제 크기"), ("none", "변형 안 보기")]


class ResultStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, "7. 결과", "step.result")
        self.card = QFrame()
        self.card.setObjectName("verdictCard")
        card = QVBoxLayout(self.card)
        self.card_title = QLabel()
        self.card_title.setTextFormat(Qt.TextFormat.RichText)
        self.card_body = QLabel()
        self.card_body.setWordWrap(True)
        card.addWidget(self.card_title)
        card.addWidget(self.card_body)
        self.add(with_help(self.card, "result.verdict"))

        form = QFormLayout()
        self.field = QComboBox()
        for key, label in FIELD_LABELS:
            self.field.addItem(label, key)
        self.field.currentIndexChanged.connect(app.show_result)
        form.addRow("보기", with_help(self.field, "result.field"))
        self.scale = QComboBox()
        for key, label in SCALE_LABELS:
            self.scale.addItem(label, key)
        self.scale.currentIndexChanged.connect(app.show_result)
        form.addRow("변형", with_help(self.scale, "result.scale"))
        box = QWidget()
        box.setLayout(form)
        self.add(box)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.add(self.summary)
        self.probe = _hint("결과 위를 클릭하면 그 위치의 값을 보여줍니다.")
        self.add(self.probe)

        buttons = QHBoxLayout()
        report = QPushButton("보고서 저장 (PDF)")
        report.clicked.connect(app.save_report)
        buttons.addWidget(report)
        buttons.addWidget(HelpButton("result.report"))
        b = QWidget()
        b.setLayout(buttons)
        self.add(b)
        self.layout_.addStretch(1)
        disclaimer = _hint(topic("result.disclaimer").short)
        disclaimer.setStyleSheet("color: #8a6d3b; font-size: 11px;")
        self.add(with_help(disclaimer, "result.disclaimer"))

    def field_key(self) -> str:
        return self.field.currentData()

    def scale_key(self) -> str:
        return self.scale.currentData()
