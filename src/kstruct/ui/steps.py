"""The seven wizard step panels. Each panel edits the shared Project through the
MainWindow (passed in as `app`) and emits `changed` when the project was modified."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kstruct import standards
from kstruct.fem.bolts import grades as bolt_grades
from kstruct.fem.bolts import hole_axis, recommended_preload, suggest_size
from kstruct.fem.bolts import sizes as bolt_sizes
from kstruct.fem.constraints import ConstraintError
from kstruct.fem.fatigue import RELIABILITY_FACTORS, SURFACE_FACTORS
from kstruct.geometry import faces_of
from kstruct.geometry.face_info import common_axis, face_info, faces_centroid, outward_normal
from kstruct.i18n import N_, tr, tr_data
from kstruct.model import (
    AXIS_LOADS,
    GRAVITY,
    AnalysisKind,
    Bolt,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadCase,
    LoadKind,
    MeshRefinement,
)
from kstruct.report.describe import describe_connection, describe_fixture, describe_load
from kstruct.ui import theme
from kstruct.ui.dialogs import MaterialDialog, load_user_library, save_user_library
from kstruct.ui.help import HelpButton, HelpDialog, topic, with_help
from kstruct.units import quantity, system_of

if TYPE_CHECKING:
    from kstruct.ui.main_window import MainWindow

CONNECTION_LABELS = {
    ConnectionKind.BONDED: N_("접합 (붙어 있음 — 용접·접착)"),
    ConnectionKind.CONTACT: N_("접촉 (눌리면 버티고, 당기면 떨어짐·미끄러짐)"),
}
UNIT_LABELS = [
    (None, N_("기준에 따름")),
    ("SI", N_("SI (mm · N · MPa · °C)")),
    ("US", N_("미국 (in · lbf · psi · °F)")),
]
# which physical quantity each load's value is, for unit conversion (None: no unit change)
LOAD_QUANTITY = {
    LoadKind.FORCE: "force",
    LoadKind.PRESSURE: "stress",
    LoadKind.GRAVITY: None,
    LoadKind.TEMPERATURE: "temperature_change",
    LoadKind.FACE_TEMPERATURE: "temperature",
    LoadKind.HEAT_FLOW: None,
    LoadKind.CONVECTION: None,
    LoadKind.RADIATION: None,
    LoadKind.TORQUE: "torque",
    LoadKind.BEARING: "force",
    LoadKind.CENTRIFUGAL: "speed",
    LoadKind.REMOTE: "force",
    LoadKind.MOMENT: "torque",
}
FIXTURE_LABELS = {
    FixtureKind.FIXED: (N_("완전 고정"), "fixture.fixed"),
    FixtureKind.ROLLER: (N_("미끄럼 지지"), "fixture.roller"),
    FixtureKind.PIN: (N_("핀 지지 (구멍·축)"), "fixture.pin"),
    FixtureKind.DISPLACEMENT: (N_("강제 변위 (정해진 만큼 움직임)"), "fixture.displacement"),
    FixtureKind.ELASTIC: (N_("탄성 지지 (고무·스프링 받침)"), "fixture.elastic"),
}
LOAD_LABELS = {
    LoadKind.FORCE: (N_("힘"), "load.force"),
    LoadKind.PRESSURE: (N_("압력"), "load.pressure"),
    LoadKind.GRAVITY: (N_("자중 (중력)"), "load.gravity"),
    LoadKind.TEMPERATURE: (N_("온도 변화 (전체)"), "load.temperature"),
    LoadKind.FACE_TEMPERATURE: (N_("고정 온도 (면)"), "load.face_temperature"),
    LoadKind.HEAT_FLOW: (N_("열 입력 (면)"), "load.heat_flow"),
    LoadKind.CONVECTION: (N_("대류 냉각 (면)"), "load.convection"),
    LoadKind.RADIATION: (N_("복사 냉각 (면)"), "load.radiation"),
    LoadKind.TORQUE: (N_("토크 (비틀림)"), "load.torque"),
    LoadKind.BEARING: (N_("베어링 하중 (구멍에 끼운 축·핀)"), "load.bearing"),
    LoadKind.CENTRIFUGAL: (N_("원심력 (회전)"), "load.centrifugal"),
    LoadKind.REMOTE: (N_("원격 하중 (떨어진 곳에서 미는 힘)"), "load.remote"),
    LoadKind.MOMENT: (N_("모멘트 (굽히거나 비트는 힘)"), "load.moment"),
}
FACE_LOADS = (
    LoadKind.FORCE,
    LoadKind.PRESSURE,
    LoadKind.FACE_TEMPERATURE,
    LoadKind.HEAT_FLOW,
    LoadKind.CONVECTION,
    LoadKind.RADIATION,
    LoadKind.TORQUE,
    LoadKind.BEARING,
    LoadKind.REMOTE,
    LoadKind.MOMENT,
)
# loads whose direction must be a global axis (no "normal to the face" choices)
AXIS_DIRECTION_LOADS = (LoadKind.BEARING, LoadKind.REMOTE, LoadKind.MOMENT)
AXES = [
    (N_("선택한 원통면의 축"), None),
    (N_("X 축"), (1.0, 0.0, 0.0)),
    (N_("Y 축"), (0.0, 1.0, 0.0)),
    (N_("Z 축"), (0.0, 0.0, 1.0)),
]
DIRECTIONS = [
    (N_("면을 수직으로 누르기"), None, 1.0),
    (N_("면을 수직으로 당기기"), None, -1.0),
    (N_("+X 방향"), (1, 0, 0), 1.0),
    (N_("−X 방향"), (-1, 0, 0), 1.0),
    (N_("+Y 방향"), (0, 1, 0), 1.0),
    (N_("−Y 방향"), (0, -1, 0), 1.0),
    (N_("+Z 방향"), (0, 0, 1), 1.0),
    (N_("−Z 방향 (아래)"), (0, 0, -1), 1.0),
]


def _title(text: str, key: str) -> QWidget:
    label = QLabel(text)
    label.setObjectName("stepTitle")  # styled by ui/theme.py, like the roles below
    return with_help(label, key)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", "hint")
    return label


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "section")
    return label


def _primary(button: QPushButton) -> QPushButton:
    """The one main action of a panel (filled with the accent colour)."""
    button.setProperty("primary", "true")
    button.setMinimumHeight(34)
    return button


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
            kind = tr_data(face_info(faces_of(part.shape)[ref.face_index]).kind)
            prefix = f"{tr_data(part.name)} " if len(project.parts) > 1 else ""
            names.append(
                tr("{prefix}면#{n}({kind})", prefix=prefix, n=ref.face_index + 1, kind=kind)
            )
        return ", ".join(names)


# --- 1. open ----------------------------------------------------------------------


class OpenStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("1. 모델 열기"), "step.open")
        open_btn = _primary(QPushButton(tr("파일 열기…")))
        open_btn.clicked.connect(app.open_file)
        self.add(with_help(open_btn, "step.open"))
        cad_help = QPushButton(tr("SolidWorks·Inventor·CATIA 파일은?"))
        cad_help.setFlat(True)
        cad_help.clicked.connect(lambda: HelpDialog("file.export_step", self.window()).exec())
        self.add(with_help(cad_help, "file.export_step"))
        mesh_help = QPushButton(tr("STL·FBX 등 메시 파일은?"))
        mesh_help.setFlat(True)
        mesh_help.clicked.connect(lambda: HelpDialog("file.mesh_formats", self.window()).exec())
        self.add(with_help(mesh_help, "file.mesh_formats"))

        samples = QHBoxLayout()
        for label, name in (
            (tr("예제: 브래킷"), "bracket"),
            (tr("예제: 판 2장"), "two_plates"),
            (tr("예제: 볼트"), "bolted"),
        ):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, n=name: app.open_sample(n))
            samples.addWidget(b)
        box = QWidget()
        box.setLayout(samples)
        self.add(box)

        self.add(with_help(_section(tr("부품")), "view.parts"))
        self.parts = QListWidget()
        self.parts.itemChanged.connect(self._toggled)
        self.add(self.parts, 1)

        self.conn_box = QGroupBox(tr("부품 사이 연결"))
        conn = QVBoxLayout(self.conn_box)
        conn.addWidget(_hint(topic("connection.bonded").short))
        detect = QPushButton(tr("맞닿은 면 자동으로 찾기"))
        detect.clicked.connect(app.detect_connections)
        conn.addWidget(with_help(detect, "connection.bonded"))
        self.connections = QListWidget()
        self.connections.setMaximumHeight(110)
        conn.addWidget(self.connections)
        self.conn_kind = QComboBox()
        for kind, label in CONNECTION_LABELS.items():
            self.conn_kind.addItem(tr(label), kind)
        self.conn_kind.activated.connect(self._kind_chosen)
        self.connections.currentRowChanged.connect(self._connection_selected)
        conn.addWidget(with_help(self.conn_kind, "connection.contact"))
        self.friction = QDoubleSpinBox()
        self.friction.setRange(0.0, 2.0)
        self.friction.setSingleStep(0.05)
        self.friction.setDecimals(2)
        self.friction.setToolTip(topic("connection.friction").short)
        self.friction.valueChanged.connect(self._friction_changed)
        self.friction_row = QWidget()
        row = QHBoxLayout(self.friction_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(tr("마찰계수 μ")))
        row.addWidget(with_help(self.friction, "connection.friction"), 1)
        conn.addWidget(self.friction_row)
        remove = QPushButton(tr("선택한 연결 삭제"))
        remove.clicked.connect(self._remove_connection)
        conn.addWidget(remove)
        self.add(self.conn_box)

    def refresh(self) -> None:
        project = self.app.project
        self.parts.blockSignals(True)
        self.parts.clear()
        for part in project.parts if project else []:
            item = QListWidgetItem(
                tr("{name}  (면 {n}개)", name=tr_data(part.name), n=len(faces_of(part.shape)))
            )
            item.setData(Qt.ItemDataRole.UserRole, part.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if part.visible else Qt.CheckState.Unchecked)
            self.parts.addItem(item)
        self.parts.blockSignals(False)
        self.conn_box.setVisible(bool(project and len(project.parts) > 1))
        row = self.connections.currentRow()
        self.connections.blockSignals(True)
        self.connections.clear()
        for c in project.connections if project else []:
            self.connections.addItem(describe_connection(project, c, self.describe_faces))
        self.connections.blockSignals(False)
        if project and project.connections:
            self.connections.setCurrentRow(min(max(row, 0), len(project.connections) - 1))
        self._connection_selected()

    def _connection_selected(self) -> None:
        row = self.connections.currentRow()
        self.conn_kind.setEnabled(row >= 0)
        c = self.app.project.connections[row] if row >= 0 else None
        if c is not None:
            self.conn_kind.setCurrentIndex(self.conn_kind.findData(c.kind))
            self.friction.blockSignals(True)
            self.friction.setValue(c.friction)
            self.friction.blockSignals(False)
        self.friction_row.setVisible(c is not None and c.kind == ConnectionKind.CONTACT)

    def _kind_chosen(self) -> None:
        row = self.connections.currentRow()
        if row < 0:
            return
        self.app.project.connections[row].kind = self.conn_kind.currentData()
        self.changed.emit()
        self.refresh()

    def _friction_changed(self, value: float) -> None:
        row = self.connections.currentRow()
        if row < 0:
            return
        self.app.project.connections[row].friction = float(value)
        self.changed.emit()
        self.connections.item(row).setText(
            describe_connection(
                self.app.project, self.app.project.connections[row], self.describe_faces
            )
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
        super().__init__(app, tr("2. 재료"), "step.material")
        self.standard = QComboBox()
        for p in standards.packs().values():
            self.standard.addItem(tr_data(p.label), p.id)
        self.standard.currentIndexChanged.connect(self._standard_changed)
        form = QFormLayout()
        form.addRow(tr("적용 기준"), with_help(self.standard, "material.standard"))
        self.units = QComboBox()
        for key, label in UNIT_LABELS:
            self.units.addItem(tr(label), key)
        self.units.currentIndexChanged.connect(self._units_changed)
        form.addRow(tr("표시 단위"), with_help(self.units, "material.units"))
        box = QWidget()
        box.setLayout(form)
        self.add(box)

        self.add(_section(tr("부품을 고르고 재료를 지정하세요")))
        self.parts = QListWidget()
        self.parts.setMaximumHeight(120)
        self.parts.currentRowChanged.connect(lambda _: self._show_part())
        self.add(self.parts)

        self.material = QComboBox()
        self.material.setMaxVisibleItems(25)
        self.material.activated.connect(self._material_chosen)
        self.add(with_help(self.material, "material.choose"))
        self.apply_all = QPushButton(tr("모든 부품에 같은 재료 적용"))
        self.apply_all.clicked.connect(self._apply_all)
        self.add(self.apply_all)
        custom = QHBoxLayout()
        self.custom_add = QPushButton(tr("사용자 재료 추가…"))
        self.custom_add.clicked.connect(lambda: self._edit_custom(new=True))
        self.custom_edit = QPushButton(tr("편집…"))
        self.custom_edit.clicked.connect(lambda: self._edit_custom(new=False))
        self.custom_remove = QPushButton(tr("삭제"))
        self.custom_remove.clicked.connect(self._remove_custom)
        for b in (self.custom_add, self.custom_edit, self.custom_remove):
            custom.addWidget(b)
        box = QWidget()
        box.setLayout(custom)
        self.add(with_help(box, "material.custom"))
        self._library_loaded = False

        self.props = QLabel()
        self.props.setWordWrap(True)
        self.props.setTextFormat(Qt.TextFormat.RichText)
        self.add(with_help(self.props, "material.properties"))
        self.layout_.addStretch(1)

    def _pack(self):
        return standards.effective_pack(self.app.project)

    def _materials(self) -> dict:
        return standards.project_materials(self.app.project)

    def refresh(self) -> None:
        project = self.app.project
        if not self._library_loaded:
            # materials the user defined in earlier projects are offered here too; the ones
            # actually used travel inside the .kst file
            for mid, m in load_user_library().items():
                project.custom_materials.setdefault(mid, m)
            self._library_loaded = True
        self.standard.blockSignals(True)
        self.standard.setCurrentIndex(max(0, self.standard.findData(project.standard)))
        self.standard.blockSignals(False)
        self.units.blockSignals(True)
        self.units.setCurrentIndex(max(0, self.units.findData(project.units)))
        self.units.blockSignals(False)
        self._fill_materials()
        row = max(0, self.parts.currentRow())
        self.parts.blockSignals(True)
        self.parts.clear()
        mats = self._materials()
        for part in project.parts:
            m = mats.get(part.material_id)
            status = tr_data(self._pack().material_name(m)) if m else tr("⚠ 재료 미지정")
            self.parts.addItem(f"{tr_data(part.name)} — {status}")
        self.parts.blockSignals(False)
        self.parts.setCurrentRow(min(row, len(project.parts) - 1))
        self.apply_all.setVisible(len(project.parts) > 1)
        self._show_part()

    def _fill_materials(self) -> None:
        pack = self._pack()
        self.material.clear()
        self.material.addItem(tr("— 재료를 선택하세요 —"), None)
        by_cat: dict[str, list] = {}
        for m in standards.materials().values():
            by_cat.setdefault(m.category, []).append(m)
        custom = list(self.app.project.custom_materials.values())
        if custom:
            by_cat["custom"] = custom
        for cat, mats in by_cat.items():
            label = tr("사용자 재료") if cat == "custom" else tr_data(standards.CATEGORY_NAMES[cat])
            self.material.addItem(f"[{label}]", None)
            idx = self.material.count() - 1
            self.material.model().item(idx).setEnabled(False)
            for m in mats:
                self.material.addItem(
                    f"   {tr_data(pack.material_name(m))} — {tr_data(m.name)}", m.id
                )

    def _edit_custom(self, new: bool) -> None:
        project = self.app.project
        current = self._materials().get(self.material.currentData())
        if not new and (current is None or current.id not in project.custom_materials):
            self.app.warn(tr("편집할 사용자 재료를 목록에서 먼저 고르세요."))
            return
        dialog = MaterialDialog(current, system_of(project), self.window())
        if new and current is not None:
            dialog.name.setText(tr("{name} (수정)", name=current.name))
        if dialog.exec() != MaterialDialog.DialogCode.Accepted:
            return
        m = dialog.material()
        project.custom_materials[m.id] = m
        library = load_user_library()
        library[m.id] = m
        save_user_library(library)
        part = self._current_part()
        if part is not None:
            part.material_id = m.id
        self.changed.emit()
        self.refresh()

    def _remove_custom(self) -> None:
        project = self.app.project
        mid = self.material.currentData()
        if mid not in project.custom_materials:
            self.app.warn(tr("삭제할 사용자 재료를 목록에서 먼저 고르세요."))
            return
        del project.custom_materials[mid]
        library = load_user_library()
        library.pop(mid, None)
        save_user_library(library)
        for part in project.parts:
            if part.material_id == mid:
                part.material_id = None
        self.changed.emit()
        self.refresh()

    def _current_part(self):
        row = self.parts.currentRow()
        return self.app.project.parts[row] if row >= 0 else None

    def _show_part(self) -> None:
        part = self._current_part()
        if part is None:
            return
        self.material.setCurrentIndex(max(0, self.material.findData(part.material_id)))
        m = self._materials().get(part.material_id)
        if m is None:
            self.props.setText("")
            return
        pack = self._pack()
        kind = tr("연성재 (휘다가 파손)") if m.ductile else tr("취성재 (갑자기 깨짐)")
        names = " · ".join(
            f"{k.upper()} {tr_data(v)}" for k, v in m.designations.items() if k != "iso"
        )
        if m.id in self.app.project.custom_materials:
            names = tr("사용자 정의 재료")
        self.props.setText(
            f"<b>{tr_data(pack.material_name(m))}</b> — {tr_data(m.name)}<br>"
            f"<small>{names}</small><br><br>"
            + tr("단단함(탄성계수): <b>{e:,.0f} GPa</b>", e=m.youngs_modulus / 1000)
            + "<br>"
            + tr("항복강도: <b>{sy:,.0f} MPa</b> (이 이상이면 영구 변형)", sy=m.yield_strength)
            + "<br>"
            + tr("인장강도: <b>{su:,.0f} MPa</b> (이 이상이면 파단)", su=m.tensile_strength)
            + "<br>"
            + tr("밀도: {rho:.2f} g/cm³", rho=m.density * 1e9)
            + "<br>"
            + tr("성질: {kind}", kind=kind)
            + "<br>"
            + tr(
                "권장 안전율({pack}): {target:.1f} 이상",
                pack=tr_data(pack.label),
                target=pack.target(m),
            )
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

    def _units_changed(self) -> None:
        self.app.project.units = self.units.currentData()
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
        self.remove = QPushButton(tr("선택한 항목 삭제"))
        self.remove.clicked.connect(self._remove)

    def finish_layout(self) -> None:
        self.add(self.selection_label)
        self.add(_section(tr("지정된 항목")))
        self.add(self.items, 1)
        self.add(self.remove)

    def update_selection(self, refs: list[FaceRef]) -> None:
        if refs:
            self.selection_label.setText(tr("선택한 면: {faces}", faces=self.describe_faces(refs)))
        else:
            self.selection_label.setText(
                tr("3D 화면에서 면을 클릭하세요. Ctrl+클릭으로 여러 면 선택.")
            )

    def entries(self) -> list:
        raise NotImplementedError

    def describe(self, entry) -> str:
        raise NotImplementedError

    def refresh(self) -> None:
        self.items.clear()
        for e in self.entries():
            text = self.describe(e)
            self.items.addItem(text)
            self.items.item(self.items.count() - 1).setToolTip(text)  # long ones are cut off
        self.update_selection(self.app.viewer.selection)

    def _remove(self) -> None:
        row = self.items.currentRow()
        if row >= 0:
            del self.entries()[row]
            self.changed.emit()
            self.refresh()


class FixtureStep(_FaceListStep):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("3. 고정"), "step.fixture")
        self.kinds = QButtonGroup(self)
        for i, (kind, (label, key)) in enumerate(FIXTURE_LABELS.items()):
            radio = QRadioButton(tr(label))
            radio.setChecked(i == 0)
            self.kinds.addButton(radio, i)
            radio.setProperty("kind", kind)
            self.add(with_help(radio, key))
        self.kinds.buttonToggled.connect(lambda *_: self._kind_changed())
        self._build_value_box()
        self.add_btn = _primary(QPushButton(tr("선택한 면을 고정")))
        self.add_btn.clicked.connect(self._add)
        self.add(self.add_btn)
        self.finish_layout()
        self._build_bolt_box()
        self._kind_changed()

    def _build_value_box(self) -> None:
        """Inputs of the two fixtures that carry a value: the movement of a prescribed
        displacement (per axis, unticked = free) and the stiffness of an elastic support."""
        self.value_box = QGroupBox(tr("값"))
        form = QFormLayout(self.value_box)
        self.move_axes: list[tuple[QCheckBox, QDoubleSpinBox]] = []
        for axis in "XYZ":
            check = QCheckBox(tr("{axis} 방향", axis=axis))
            spin = QDoubleSpinBox()
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            check.toggled.connect(spin.setEnabled)
            check.setChecked(axis == "Z")
            spin.setEnabled(axis == "Z")
            self.move_axes.append((check, spin))
            form.addRow(check, spin)
        self.move_hint = _hint(
            tr(
                "체크한 방향만 그 값만큼 움직이고, 체크하지 않은 방향은 자유롭게 움직입니다. "
                "0을 넣으면 그 방향으로 고정됩니다."
            )
        )
        form.addRow(self.move_hint)
        self.stiffness = QDoubleSpinBox()
        self.stiffness.setRange(1e-6, 1e12)
        self.stiffness.setDecimals(1)
        self.stiffness.setValue(10_000.0)
        self.stiffness_label = QLabel(tr("강성"))
        form.addRow(self.stiffness_label, self.stiffness)
        self.stiffness_hint = _hint(
            tr("고른 면 전체를 1만큼 누르는 데 드는 힘입니다. 면에 수직인 방향으로만 받칩니다.")
        )
        form.addRow(self.stiffness_hint)
        self.add(self.value_box)

    def _current_kind(self) -> FixtureKind:
        return self.kinds.checkedButton().property("kind")

    def _kind_changed(self) -> None:
        kind = self._current_kind()
        moving, elastic = kind == FixtureKind.DISPLACEMENT, kind == FixtureKind.ELASTIC
        self.value_box.setVisible(moving or elastic)
        for check, spin in self.move_axes:
            check.setVisible(moving)
            spin.setVisible(moving)
        self.move_hint.setVisible(moving)
        for w in (self.stiffness, self.stiffness_label, self.stiffness_hint):
            w.setVisible(elastic)
        self._update_units()

    def _update_units(self) -> None:
        if self.app.project is None:
            return
        system = system_of(self.app.project)
        length, stiffness = quantity(system, "length"), quantity(system, "stiffness")
        for _, spin in self.move_axes:
            spin.setSuffix(f" {length.unit}")
        self.stiffness.setSuffix(f" {stiffness.unit}")

    # --- bolts ---------------------------------------------------------------------

    def _build_bolt_box(self) -> None:
        box = QGroupBox(tr("볼트 체결"))
        layout = QVBoxLayout(box)
        layout.addWidget(
            _hint(tr("두 부품의 구멍 면을 Ctrl+클릭으로 하나씩 고른 뒤 볼트를 추가하세요."))
        )
        form = QFormLayout()
        self.bolt_size = QComboBox()
        for name in bolt_sizes():
            self.bolt_size.addItem(name, name)
        self.bolt_grade = QComboBox()
        for name in bolt_grades():
            self.bolt_grade.addItem(name, name)
        self.bolt_grade.setCurrentIndex(self.bolt_grade.findData("8.8"))
        self.bolt_preload = QDoubleSpinBox()
        self.bolt_preload.setRange(0, 1e7)
        self.bolt_preload.setDecimals(0)
        self.bolt_preload.setSuffix(" N")
        self.bolt_auto = QCheckBox(tr("권장 체결력 자동"))
        self.bolt_auto.setChecked(True)
        for w in (self.bolt_size, self.bolt_grade):
            w.currentIndexChanged.connect(self._update_preload)
        self.bolt_auto.toggled.connect(self._update_preload)
        form.addRow(tr("크기"), with_help(self.bolt_size, "bolt.size"))
        form.addRow(tr("강도 등급"), with_help(self.bolt_grade, "bolt.grade"))
        form.addRow(
            tr("체결력"), with_help(_row(self.bolt_preload, self.bolt_auto), "bolt.preload")
        )
        layout.addLayout(form)
        self.bolt_add = QPushButton(tr("선택한 두 구멍을 볼트로 연결"))
        self.bolt_add.clicked.connect(self._add_bolt)
        layout.addWidget(self.bolt_add)
        self.bolt_list = QListWidget()
        self.bolt_list.setMaximumHeight(100)
        layout.addWidget(self.bolt_list)
        remove = QPushButton(tr("선택한 볼트 삭제"))
        remove.clicked.connect(self._remove_bolt)
        layout.addWidget(remove)
        self.add(with_help(box, "bolt.joint"))
        self._update_preload()

    def _update_preload(self) -> None:
        self.bolt_preload.setEnabled(not self.bolt_auto.isChecked())
        if self.bolt_auto.isChecked():
            self.bolt_preload.setValue(
                recommended_preload(self.bolt_size.currentData(), self.bolt_grade.currentData())
            )

    def _hole_pair(self, refs: list[FaceRef]):
        if len(refs) != 2 or refs[0].part_id == refs[1].part_id:
            return None
        project = self.app.project
        try:
            holes = [hole_axis(project, r) for r in refs]
        except ConstraintError:
            return None
        return holes

    def _add_bolt(self) -> None:
        refs = self.app.viewer.selection
        if self._hole_pair(refs) is None:
            self.app.warn(
                tr(
                    "서로 다른 두 부품의 구멍(원통면)을 하나씩, 모두 두 개 골라 주세요. "
                    "Ctrl을 누른 채 클릭하면 여러 면을 고를 수 있습니다."
                )
            )
            return
        bolts = self.app.project.bolts
        bolts.append(
            Bolt(
                refs[0],
                refs[1],
                self.bolt_size.currentData(),
                self.bolt_grade.currentData(),
                self.bolt_preload.value(),
                tr("볼트 {n}", n=len(bolts) + 1),
            )
        )
        self.app.viewer.set_selection([])
        self.changed.emit()
        self.refresh()

    def _remove_bolt(self) -> None:
        row = self.bolt_list.currentRow()
        if row >= 0:
            del self.app.project.bolts[row]
            self.changed.emit()
            self.refresh()

    def refresh(self) -> None:
        super().refresh()
        self._update_units()
        self.bolt_list.clear()
        for b in self.app.project.bolts:
            self.bolt_list.addItem(
                tr(
                    "{name}: {size} {grade}, 체결력 {preload:,.0f} N — {faces}",
                    name=b.name,
                    size=b.size,
                    grade=b.grade,
                    preload=b.preload,
                    faces=self.describe_faces([b.hole_a, b.hole_b]),
                )
            )

    def entries(self) -> list:
        return self.app.project.fixtures

    def describe(self, fx: Fixture) -> str:
        return describe_fixture(self.app.project, fx, self.describe_faces)

    def update_selection(self, refs: list[FaceRef]) -> None:
        super().update_selection(refs)
        self.add_btn.setEnabled(bool(refs))
        holes = self._hole_pair(refs) if hasattr(self, "bolt_add") else None
        if hasattr(self, "bolt_add"):
            self.bolt_add.setEnabled(holes is not None)
        if holes is not None:
            # suggest the bolt that fits the smaller hole
            name = suggest_size(min(d for _, _, d in holes))
            self.bolt_size.setCurrentIndex(self.bolt_size.findData(name))

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
                        tr("핀 지지는 원통면(구멍 안쪽이나 축 바깥면)에만 지정할 수 있습니다.")
                    )
                    return
        if not refs:
            self.app.warn(tr("3D 화면에서 면을 먼저 고르세요."))
            return
        system = system_of(self.app.project)
        fixture = Fixture(kind, refs)
        if kind == FixtureKind.DISPLACEMENT:
            length = quantity(system, "length")
            fixture.displacement = tuple(
                length.parse(spin.value()) if check.isChecked() else None
                for check, spin in self.move_axes
            )
            if all(d is None for d in fixture.displacement):
                self.app.warn(tr("움직일 방향을 하나 이상 체크하세요."))
                return
        elif kind == FixtureKind.ELASTIC:
            fixture.stiffness = quantity(system, "stiffness").parse(self.stiffness.value())
        self.app.project.fixtures.append(fixture)
        self.app.viewer.set_selection([])
        self.changed.emit()
        self.refresh()


# --- 4. loads -------------------------------------------------------------------------


class LoadStep(_FaceListStep):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("4. 하중"), "step.load")
        self.case = QComboBox()
        self.case.currentIndexChanged.connect(self._case_chosen)
        self.case.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        case_add = QPushButton(tr("추가"))
        case_add.clicked.connect(self._add_case)
        case_rename = QPushButton(tr("이름"))
        case_rename.clicked.connect(self._rename_case)
        self.case_remove = QPushButton(tr("삭제"))
        self.case_remove.clicked.connect(self._remove_case)
        for button in (case_add, case_rename, self.case_remove):
            button.setProperty("compact", "true")
        self.add(
            with_help(
                _row(QLabel(tr("하중 케이스")), self.case, case_add, case_rename, self.case_remove),
                "load.cases",
            )
        )
        self.kind = QComboBox()
        for kind, (label, _) in LOAD_LABELS.items():
            self.kind.addItem(tr(label), kind)
        self.kind.currentIndexChanged.connect(lambda _index: self._kind_changed())
        self.kind_help = HelpButton("load.force")
        self.add(_row(self.kind, self.kind_help))

        form = QFormLayout()
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e9, 1e9)
        self.value.setDecimals(2)
        self.value.setValue(1000)
        self.value.valueChanged.connect(self._update_hint)
        self.value_label = QLabel(tr("크기"))
        form.addRow(self.value_label, self.value)
        self.direction = QComboBox()
        for label, _, _ in DIRECTIONS:
            self.direction.addItem(tr(label))
        self.direction_label = QLabel(tr("방향"))
        form.addRow(self.direction_label, with_help(self.direction, "load.direction"))
        self.axis = QComboBox()
        for label, _ in AXES:
            self.axis.addItem(tr(label))
        self.axis_label = QLabel(tr("회전축"))
        form.addRow(self.axis_label, with_help(self.axis, "load.axis"))
        self.ambient = QDoubleSpinBox()
        self.ambient.setRange(-100, 2000)
        self.ambient.setValue(20)
        self.ambient_label = QLabel(tr("주변 공기 온도 (°C)"))
        form.addRow(self.ambient_label, self.ambient)
        self.offset = [QDoubleSpinBox() for _ in range(3)]
        for spin in self.offset:
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(2)
        self.offset[0].setValue(100.0)
        # one row per axis: three spin boxes side by side made the panel wider than the
        # side bar and cut off its buttons
        self.offset_labels = [QLabel(tr("작용점 {axis}", axis=axis)) for axis in "XYZ"]
        self.offset_rows = [
            with_help(spin, "load.remote_offset") if i == 0 else spin
            for i, spin in enumerate(self.offset)
        ]
        for label, row in zip(self.offset_labels, self.offset_rows, strict=True):
            form.addRow(label, row)
        self.offset_row = self.offset_rows[0]  # (tests and visibility)
        box = QWidget()
        box.setLayout(form)
        self.add(box)
        self.value_hint = _hint("")
        self.add(self.value_hint)

        self.add_btn = _primary(QPushButton(tr("하중 추가")))
        self.add_btn.clicked.connect(self._add)
        self.add(self.add_btn)
        self.finish_layout()
        self._shown_system = None
        self._kind_changed()

    def _current_kind(self) -> LoadKind:
        return self.kind.currentData()

    def _kind_changed(self, reset: bool = True) -> None:
        kind = self._current_kind()
        self.kind_help.key = LOAD_LABELS[kind][1]
        self.kind_help.setToolTip(topic(self.kind_help.key).short)
        self.selection_label.setVisible(kind in FACE_LOADS)
        # each control sits in a wrapper with its "?" button: hide the wrapper too, or the
        # button of a hidden row stays behind on its own
        show_direction = kind in (LoadKind.FORCE, LoadKind.GRAVITY) + AXIS_DIRECTION_LOADS
        for widget in (*self.offset_labels, *self.offset_rows, *self.offset):
            widget.setVisible(kind == LoadKind.REMOTE)
        if self.app.project is not None:
            unit = quantity(system_of(self.app.project), "length").unit
            for spin in self.offset:
                spin.setSuffix(f" {unit}")
        for widget in (self.direction, self.direction.parentWidget(), self.direction_label):
            widget.setVisible(show_direction)
        for widget in (self.axis, self.axis.parentWidget(), self.axis_label):
            widget.setVisible(kind in AXIS_LOADS)
        if kind in AXIS_DIRECTION_LOADS:
            # along a fixed global direction only: no "normal to the face" choices
            for i in range(2):
                self.direction.model().item(i).setEnabled(False)
            if self.direction.currentIndex() < 2:
                self.direction.setCurrentIndex(len(DIRECTIONS) - 1)
        else:
            for i in range(2):
                self.direction.model().item(i).setEnabled(True)
        show_ambient = kind in (LoadKind.CONVECTION, LoadKind.RADIATION)
        self.ambient.setVisible(show_ambient)
        self.ambient_label.setVisible(show_ambient)
        defaults = {  # internal units
            LoadKind.FORCE: (tr("크기"), 1000.0),
            LoadKind.PRESSURE: (tr("압력"), 1.0),
            LoadKind.GRAVITY: (tr("배수 (g)"), 1.0),
            LoadKind.TEMPERATURE: (tr("온도 변화"), 50.0),
            LoadKind.FACE_TEMPERATURE: (tr("온도"), 100.0),
            LoadKind.HEAT_FLOW: (tr("열량 (W)"), 10.0),
            LoadKind.CONVECTION: (tr("열전달계수 h (W/m²·K)"), 10.0),
            LoadKind.RADIATION: (tr("방사율 ε (0~1)"), 0.8),
            LoadKind.TORQUE: (tr("토크"), 10_000.0),  # N·mm = 10 N·m
            LoadKind.BEARING: (tr("힘"), 1000.0),
            LoadKind.CENTRIFUGAL: (tr("회전 속도"), 3000 * 2 * 3.141592653589793 / 60),  # rad/s
            LoadKind.REMOTE: (tr("힘"), 1000.0),
            LoadKind.MOMENT: (tr("모멘트"), 100_000.0),  # N·mm = 100 N·m
        }
        label, default = defaults[kind]
        q = self._quantity(kind)
        if q is not None:
            label += f" ({q.unit})"
            default = round(q.show(default), 3)
        self.value_label.setText(label)
        system = system_of(self.app.project) if self.app.project else "SI"
        if system != self._shown_system:
            reset = True  # the typed number would mean something else in the new units
        self._shown_system = system
        t = quantity(system, "temperature")
        self.ambient_label.setText(
            f"{tr('주변 온도') if kind == LoadKind.RADIATION else tr('주변 공기 온도')} ({t.unit})"
        )
        if reset:
            self.value.setValue(default)
            self.ambient.setValue(round(t.show(20.0), 1))
        if kind == LoadKind.GRAVITY and reset:
            self.direction.setCurrentIndex(len(DIRECTIONS) - 1)
        self._update_hint()
        self.update_selection(self.app.viewer.selection)

    def _quantity(self, kind: LoadKind):
        name = LOAD_QUANTITY[kind]
        if name is None or self.app.project is None:
            return None
        return quantity(system_of(self.app.project), name)

    def _internal_value(self, kind: LoadKind) -> float:
        q = self._quantity(kind)
        v = self.value.value()
        return q.parse(v) if q is not None else v

    def refresh(self) -> None:
        project = self.app.project
        self.case.blockSignals(True)
        self.case.clear()
        for c in project.cases:
            self.case.addItem(tr("{name} ({n}개)", name=c.name, n=len(c.loads)))
        self.case.setCurrentIndex(project.active_case)
        self.case.blockSignals(False)
        self.case_remove.setEnabled(len(project.cases) > 1)
        self._kind_changed(reset=False)
        super().refresh()

    def _case_chosen(self, index: int) -> None:
        if index >= 0:
            self.app.project.active_case = index
            self.changed.emit()  # the viewer shows this case's arrows, results are stale
            self.refresh()

    def _add_case(self, name: str | None = None) -> None:
        project = self.app.project
        name = name or tr("하중 케이스 {n}", n=len(project.cases) + 1)
        project.cases.append(LoadCase(name))
        project.active_case = len(project.cases) - 1
        self.changed.emit()
        self.refresh()

    def _rename_case(self) -> None:
        case = self.app.project.case
        name, ok = QInputDialog.getText(
            self.window(), tr("케이스 이름"), tr("이름"), text=case.name
        )
        if ok and name.strip():
            case.name = name.strip()
            self.changed.emit()
            self.refresh()

    def _remove_case(self) -> None:
        project = self.app.project
        if len(project.cases) < 2:
            return
        if project.case.loads and not self.app._confirm(
            tr(
                "'{name}' 케이스와 그 하중 {n}개를 지울까요?",
                name=project.case.name,
                n=len(project.case.loads),
            )
        ):
            return
        del project.cases[project.active_case]
        project.active_case = max(0, project.active_case - 1)
        self.changed.emit()
        self.refresh()

    def _update_hint(self) -> None:
        kind = self._current_kind()
        v = self._internal_value(kind)
        hints = {
            LoadKind.FORCE: tr("≈ {kg:,.1f} kg의 물체를 올려놓은 무게와 같습니다.", kg=v / 9.81),
            LoadKind.PRESSURE: tr("= {bar:,.2f} bar (자동차 타이어는 약 2.4 bar)", bar=v * 10),
            LoadKind.GRAVITY: tr("1 g = 지구 중력. 충격·가속을 고려하려면 2~3 g를 입력하세요."),
            LoadKind.TEMPERATURE: tr(
                "모든 부품이 고르게 이만큼 데워진(+) 또는 식은(−) 경우입니다."
            ),
            LoadKind.FACE_TEMPERATURE: tr(
                "선택한 면이 항상 이 온도로 유지됩니다 (예: 히터·냉각수에 닿은 면)."
            ),
            LoadKind.HEAT_FLOW: tr("선택한 면으로 들어오는 열입니다 (예: 모터·전자부품의 발열량)."),
            LoadKind.CONVECTION: tr(
                "자연 공랭 5~10, 선풍기 바람 20~50, 강제 송풍 50~200, 수랭 500 이상"
            ),
            LoadKind.RADIATION: tr(
                "검은 도장·산화 표면 0.8~0.95, 양극산화 알루미늄 0.8, 거친 강 0.5~0.7, 광택 금속 0.05~0.2"
            ),
            LoadKind.TORQUE: tr(
                "≈ 반지름 100 mm 끝에서 {kg:,.2f} kg 무게로 돌리는 힘. "
                "양수는 회전축 방향으로 오른나사가 나아가는 방향(반시계)입니다.",
                kg=v / 100 / 9.81,
            ),
            LoadKind.BEARING: tr(
                "구멍 벽의 하중 방향 절반에 코사인 분포로 눌러 줍니다(핀·축이 미는 모양)."
            ),
            LoadKind.CENTRIFUGAL: tr(
                "= {v:,.1f} rad/s. 모든 부품에 회전축에서 바깥쪽으로 ρω²r의 힘이 걸립니다.", v=v
            ),
            LoadKind.REMOTE: (
                tr(
                    "작용점까지 떨어진 거리만큼 굽히는 힘이 함께 걸립니다. 고른 면은 단단한 "
                    "연결판에 묶인 것처럼 하중을 나눠 받습니다."
                )
            ),
            LoadKind.MOMENT: (
                tr(
                    "방향은 회전축입니다(오른손 법칙). 면 중심에서 멀수록 크게, 직선으로 나뉘어 "
                    "걸립니다."
                )
            ),
        }
        self.value_hint.setText(hints[kind])

    def update_selection(self, refs: list[FaceRef]) -> None:
        super().update_selection(refs)
        needs_faces = self._current_kind() in FACE_LOADS
        self.add_btn.setEnabled(bool(refs) or not needs_faces)

    def _selected_faces(self, refs: list[FaceRef]) -> list:
        project = self.app.project
        return [faces_of(project.part(r.part_id).shape)[r.face_index] for r in refs]

    def _axis_for(self, kind: LoadKind, refs: list[FaceRef]):
        """(point, direction) for torque / centrifugal loads, or None with a message."""
        _, chosen = AXES[self.axis.currentIndex()]
        faces = self._selected_faces(refs)
        from_faces = common_axis(faces) if faces else None
        if chosen is None:
            if from_faces is None:
                return None
            point, direction = from_faces
            return tuple(float(c) for c in point), tuple(float(c) for c in direction)
        if faces:
            point = faces_centroid(faces)
        else:  # centrifugal without faces: the axis passes through the model's centre
            parts = self.app.project.visible_parts()
            point = sum(
                (np.asarray(faces_centroid(faces_of(p.shape))) for p in parts), np.zeros(3)
            ) / max(len(parts), 1)
        return tuple(float(c) for c in point), chosen

    def entries(self) -> list:
        return self.app.project.loads

    def describe(self, ld: Load) -> str:
        return describe_load(self.app.project, ld, self.describe_faces)

    def _add(self) -> None:
        kind = self._current_kind()
        v = self._internal_value(kind)
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
        elif kind == LoadKind.TORQUE:
            axis = self._axis_for(kind, refs)
            if axis is None:
                self.app.warn(
                    tr(
                        "토크의 회전축을 정할 수 없습니다. 축이나 구멍의 원통면을 고르거나, "
                        "'회전축'에서 X·Y·Z 축을 지정하세요."
                    )
                )
                return
            load = Load(LoadKind.TORQUE, refs, vector=axis[1], magnitude=v, point=axis[0])
        elif kind == LoadKind.BEARING:
            if common_axis(self._selected_faces(refs)) is None:
                self.app.warn(
                    tr(
                        "베어링 하중은 구멍(원통면)에만 줄 수 있습니다. 구멍 안쪽 면을 선택하세요. "
                        "여러 면을 고르면 같은 축의 구멍이어야 합니다."
                    )
                )
                return
            _, axis, _ = DIRECTIONS[self.direction.currentIndex()]
            load = Load(LoadKind.BEARING, refs, vector=tuple(v * c + 0.0 for c in axis))
        elif kind in (LoadKind.REMOTE, LoadKind.MOMENT):
            faces = self._selected_faces(refs)
            if not faces:
                self.app.warn(tr("하중을 전달할 면을 3D 화면에서 먼저 고르세요."))
                return
            _, axis, _ = DIRECTIONS[self.direction.currentIndex()]
            vector = tuple(v * c + 0.0 for c in axis)
            if kind == LoadKind.MOMENT:
                load = Load(LoadKind.MOMENT, refs, vector=vector)
            else:
                length = quantity(system_of(project), "length")
                center = np.asarray(faces_centroid(faces), dtype=float)
                offset = np.array([length.parse(spin.value()) for spin in self.offset])
                point = tuple(float(c) for c in center + offset)
                load = Load(LoadKind.REMOTE, refs, vector=vector, point=point)
        elif kind == LoadKind.CENTRIFUGAL:
            axis = self._axis_for(kind, refs)
            if axis is None:
                self.app.warn(
                    tr(
                        "회전축을 정할 수 없습니다. 축이나 구멍의 원통면을 고르거나 '회전축'에서 "
                        "X·Y·Z 축을 지정하세요."
                    )
                )
                return
            project.loads[:] = [ld for ld in project.loads if ld.kind != LoadKind.CENTRIFUGAL]
            load = Load(LoadKind.CENTRIFUGAL, vector=axis[1], magnitude=v, point=axis[0])
        elif kind == LoadKind.GRAVITY:
            _, axis, _ = DIRECTIONS[self.direction.currentIndex()]
            axis = axis or (0, 0, -1)
            project.loads[:] = [ld for ld in project.loads if ld.kind != LoadKind.GRAVITY]
            load = Load(LoadKind.GRAVITY, vector=tuple(v * GRAVITY * c for c in axis))
        elif kind == LoadKind.TEMPERATURE:
            project.loads[:] = [ld for ld in project.loads if ld.kind != LoadKind.TEMPERATURE]
            load = Load(LoadKind.TEMPERATURE, magnitude=v)
        else:
            t = quantity(system_of(project), "temperature")
            load = Load(kind, refs, magnitude=v, ambient=t.parse(self.ambient.value()))
        project.loads.append(load)
        if kind in FACE_LOADS:
            self.app.viewer.set_selection([])
        self.changed.emit()
        self.refresh()


# --- 5. mesh --------------------------------------------------------------------------


QUALITY_NAMES = {
    "coarse": N_("거침"),
    "normal": N_("보통"),
    "fine": N_("세밀"),
    "very_fine": N_("매우 세밀"),
}
QUALITY_LABELS = [
    ("coarse", N_("거침 (빠름, 대략적인 결과)")),
    ("normal", N_("보통 (권장)")),
    ("fine", N_("세밀 (정확, 느림)")),
    ("very_fine", N_("매우 세밀 (작은 구멍·모서리 집중 확인)")),
]


class MeshStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("5. 메시"), "step.mesh")
        self.quality = QComboBox()
        for key, label in QUALITY_LABELS:
            self.quality.addItem(tr(label), key)
        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.add(with_help(self.quality, "mesh.quality"))
        self.run = _primary(QPushButton(tr("메시 만들기")))
        self.run.clicked.connect(app.build_mesh)
        self.add(self.run)
        self.cancel = QPushButton(tr("취소"))
        self.cancel.setVisible(False)
        self.cancel.clicked.connect(app.cancel_job)
        self.add(self.cancel)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.add(self.progress)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.add(self.info)
        self.show_mesh = QCheckBox(tr("메시 선 보기"))
        self.show_mesh.setChecked(True)
        self.show_mesh.toggled.connect(lambda on: app.viewer.show_mesh(app.mesh if on else None))
        self.add(self.show_mesh)

        box = QGroupBox(tr("특정 면만 더 세밀하게"))
        layout = QVBoxLayout(box)
        layout.addWidget(_hint(topic("mesh.refine").short))
        self.selection_label = _hint("")
        layout.addWidget(self.selection_label)
        form = QFormLayout()
        self.refine_size = QDoubleSpinBox()
        self.refine_size.setRange(0.01, 1e4)
        self.refine_size.setDecimals(2)
        self.refine_size.setSuffix(" mm")
        form.addRow(tr("요소 크기"), self.refine_size)
        layout.addLayout(form)
        self.refine_add = QPushButton(tr("선택한 면 세밀하게"))
        self.refine_add.clicked.connect(self._add_refinement)
        layout.addWidget(self.refine_add)
        self.refinements = QListWidget()
        self.refinements.setMaximumHeight(90)
        layout.addWidget(self.refinements)
        remove = QPushButton(tr("선택한 항목 삭제"))
        remove.clicked.connect(self._remove_refinement)
        layout.addWidget(remove)
        self.add(with_help(box, "mesh.refine"))
        self.layout_.addStretch(1)

    def _suggested_size(self) -> float:
        from kstruct.fem.mesh import MESH_PRESETS
        from kstruct.geometry.tessellate import bounding_diagonal

        project = self.app.project
        parts = project.visible_parts() or project.parts
        diag = max(bounding_diagonal(p.shape) for p in parts) if parts else 100.0
        return diag / MESH_PRESETS[project.mesh_quality].divisions / 3

    def update_selection(self, refs: list[FaceRef]) -> None:
        if refs:
            self.selection_label.setText(tr("선택한 면: {faces}", faces=self.describe_faces(refs)))
        else:
            self.selection_label.setText(
                tr("3D 화면에서 면을 클릭하세요. Ctrl+클릭으로 여러 면 선택.")
            )
        self.refine_add.setEnabled(bool(refs))

    def _add_refinement(self) -> None:
        refs = self.app.viewer.selection
        if not refs:
            return
        self.app.project.mesh_refinements.append(MeshRefinement(refs, self.refine_size.value()))
        self.app.viewer.set_selection([])
        self.app.invalidate_mesh()
        self.refresh()

    def _remove_refinement(self) -> None:
        row = self.refinements.currentRow()
        if row >= 0:
            del self.app.project.mesh_refinements[row]
            self.app.invalidate_mesh()
            self.refresh()

    def refresh(self) -> None:
        project = self.app.project
        self.quality.blockSignals(True)
        self.quality.setCurrentIndex(max(0, self.quality.findData(project.mesh_quality)))
        self.quality.blockSignals(False)
        if not self.refine_size.value() or self.refine_size.value() == 0.01:
            self.refine_size.setValue(round(self._suggested_size(), 2))
        self.refinements.clear()
        for r in project.mesh_refinements:
            self.refinements.addItem(f"{r.size:g} mm: {self.describe_faces(r.faces)}")
        self.update_selection(self.app.viewer.selection)
        mesh = self.app.mesh
        if mesh is None:
            self.info.setText(tr("아직 메시가 없습니다. '메시 만들기'를 누르세요."))
        else:
            lines = [
                tr(
                    "✔ 메시 완료: 요소 {tets:,}개, 절점 {nodes:,}개 (계산할 미지수 {dof:,}개)",
                    tets=len(mesh.tets),
                    nodes=mesh.n_nodes,
                    dof=mesh.n_nodes * 3,
                )
            ] + [f"• {n}" for n in mesh.notes]
            self.info.setText("\n".join(lines))

    def _quality_changed(self) -> None:
        self.app.project.mesh_quality = self.quality.currentData()
        self.app.invalidate_mesh()
        self.refine_size.setValue(round(self._suggested_size(), 2))
        self.refresh()

    def set_busy(self, busy: bool) -> None:
        self.run.setEnabled(not busy)
        self.cancel.setVisible(busy)
        self.progress.setVisible(busy)


# --- 6. solve -------------------------------------------------------------------------

CYCLE_LABELS = [
    ("none", N_("아니오 — 한 번 또는 가끔 걸림")),
    ("pulsating", N_("한 방향 반복 — 0 → 최대 → 0 (예: 누름, 들어올림)")),
    ("reversed", N_("양방향 반복 — +최대 ↔ −최대 (예: 회전축의 굽힘)")),
]

ANALYSIS_LABELS = [
    (AnalysisKind.STATIC, N_("강도 — 부러지거나 휘지 않을까?"), "analysis.static"),
    (AnalysisKind.MODAL, N_("진동 — 어떤 진동수에서 흔들릴까? (공진)"), "analysis.modal"),
    (AnalysisKind.THERMAL, N_("열 — 온도가 몇 도까지 오를까?"), "analysis.thermal"),
    (AnalysisKind.BUCKLING, N_("좌굴 — 눌리면 옆으로 휘며 무너지지 않을까?"), "analysis.buckling"),
]


class SolveStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("6. 해석"), "step.solve")
        self.add(_section(tr("무엇을 알고 싶나요?")))
        self.kinds = QButtonGroup(self)
        for i, (kind, label, key) in enumerate(ANALYSIS_LABELS):
            radio = QRadioButton(tr(label))
            radio.setProperty("kind", kind)
            self.kinds.addButton(radio, i)
            self.add(with_help(radio, key))
        self.kinds.buttonClicked.connect(self._kind_changed)

        self.modal_box = QGroupBox(tr("진동 해석 설정"))
        form = QFormLayout(self.modal_box)
        self.n_modes = QSpinBox()
        self.n_modes.setRange(1, 30)
        self.n_modes.valueChanged.connect(self._modal_changed)
        form.addRow(tr("구할 진동수 개수"), with_help(self.n_modes, "modal.count"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0, 1e6)
        self.speed.setDecimals(1)
        self.speed_unit = QComboBox()
        self.speed_unit.addItems(["rpm", "Hz"])
        self.speed.valueChanged.connect(self._modal_changed)
        self.speed_unit.currentIndexChanged.connect(self._modal_changed)
        self.speed_row = with_help(_row(self.speed, self.speed_unit), "modal.operating")
        self.speed_label = QLabel(tr("운전 속도 (0=없음)"))
        form.addRow(self.speed_label, self.speed_row)
        self.add(self.modal_box)

        self.thermal_box = QGroupBox(tr("열 해석 설정"))
        hform = QFormLayout(self.thermal_box)
        self.transient = QCheckBox(tr("시간에 따른 변화 보기 (데워지고 식는 과정)"))
        self.transient.toggled.connect(self._thermal_changed)
        hform.addRow(with_help(self.transient, "analysis.transient"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1, 1e7)
        self.duration.setDecimals(0)
        self.duration_unit = QComboBox()
        self.duration_unit.addItem(tr("초"), 1.0)
        self.duration_unit.addItem(tr("분"), 60.0)
        self.duration_unit.addItem(tr("시간"), 3600.0)
        self.duration.valueChanged.connect(self._thermal_changed)
        self.duration_unit.currentIndexChanged.connect(self._thermal_changed)
        self.duration_label = QLabel(tr("계산할 시간"))
        self.duration_row = _row(self.duration, self.duration_unit)
        hform.addRow(self.duration_label, self.duration_row)
        self.initial = QDoubleSpinBox()
        self.initial.setRange(-200, 3000)
        self.initial.setDecimals(1)
        self.initial.valueChanged.connect(self._thermal_changed)
        self.initial_label = QLabel(tr("처음 온도"))
        hform.addRow(self.initial_label, self.initial)
        self.add(self.thermal_box)

        self.static_box = QGroupBox(tr("강도 해석 설정"))
        sform = QFormLayout(self.static_box)
        self.cycle = QComboBox()
        for key, label in CYCLE_LABELS:
            self.cycle.addItem(tr(label), key)
        self.cycle.currentIndexChanged.connect(self._cycle_changed)
        sform.addRow(tr("하중이 반복되나요?"), with_help(self.cycle, "fatigue.cycle"))
        self.surface = QComboBox()
        for key, (label, _a, _b) in SURFACE_FACTORS.items():
            self.surface.addItem(tr_data(label), key)
        self.surface.currentIndexChanged.connect(self._fatigue_changed)
        self.surface_label = QLabel(tr("표면 상태"))
        self.surface_box = with_help(self.surface, "fatigue.surface")
        sform.addRow(self.surface_label, self.surface_box)
        self.reliability = QComboBox()
        for key in RELIABILITY_FACTORS:
            self.reliability.addItem(f"{key} %", key)
        self.reliability.currentIndexChanged.connect(self._fatigue_changed)
        self.reliability_label = QLabel(tr("신뢰도"))
        self.reliability_box = with_help(self.reliability, "fatigue.reliability")
        sform.addRow(self.reliability_label, self.reliability_box)
        self.plasticity = QCheckBox(tr("항복한 뒤까지 계산 (영구 변형·붕괴 하중)"))
        self.plasticity.toggled.connect(self._nonlinear_changed)
        sform.addRow(with_help(self.plasticity, "analysis.plasticity"))
        self.large = QCheckBox(tr("큰 변형 고려 (얇고 많이 휘는 부품)"))
        self.large.toggled.connect(self._nonlinear_changed)
        sform.addRow(with_help(self.large, "analysis.large_deformation"))
        self.add(self.static_box)

        self.target_box = QGroupBox(tr("권장 안전율"))
        tform = QFormLayout(self.target_box)
        self.target_default = QCheckBox(tr("적용 기준의 값 사용 (권장)"))
        self.target_default.setChecked(True)
        self.target_default.toggled.connect(self._targets_changed)
        tform.addRow(with_help(self.target_default, "analysis.target"))
        self.targets: dict[str, QDoubleSpinBox] = {}
        for key, label in (
            ("ductile", tr("연성재 (강·알루미늄 등)")),
            ("brittle", tr("취성재 (주철 등)")),
            ("buckling", tr("좌굴 하중 배수")),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.valueChanged.connect(self._targets_changed)
            self.targets[key] = spin
            tform.addRow(label, spin)
        self.add(self.target_box)

        self.checklist = QLabel()
        self.checklist.setTextFormat(Qt.TextFormat.RichText)
        self.add(self.checklist)
        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setTextFormat(Qt.TextFormat.RichText)
        self.estimate.setVisible(False)
        self.add(with_help(self.estimate, "analysis.estimate")).setVisible(False)
        self.run = _primary(QPushButton(tr("해석 시작")))
        self.run.setMinimumHeight(40)
        self.run.clicked.connect(app.solve)
        self.add(self.run)
        self.cancel = QPushButton(tr("취소"))
        self.cancel.setVisible(False)
        self.cancel.clicked.connect(app.cancel_job)
        self.add(self.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.add(self.progress)
        self.status = _hint("")
        self.add(self.status)
        self.layout_.addStretch(1)

    def refresh(self) -> None:
        project = self.app.project
        for b in self.kinds.buttons():
            b.setChecked(b.property("kind") == project.analysis)
        self.modal_box.setVisible(project.analysis == AnalysisKind.MODAL)
        self.static_box.setVisible(project.analysis == AnalysisKind.STATIC)
        self._show_thermal(project)
        self.modal_box.setTitle(
            tr("좌굴 해석 설정")
            if project.analysis == AnalysisKind.BUCKLING
            else tr("진동 해석 설정")
        )
        self.modal_box.setVisible(project.analysis in (AnalysisKind.MODAL, AnalysisKind.BUCKLING))
        self.speed_row.setVisible(project.analysis == AnalysisKind.MODAL)
        self.speed_label.setVisible(project.analysis == AnalysisKind.MODAL)
        self.cycle.blockSignals(True)
        self.cycle.setCurrentIndex(max(0, self.cycle.findData(project.load_cycle)))
        self.cycle.blockSignals(False)
        for box, value in (
            (self.plasticity, project.plasticity),
            (self.large, project.large_deformation),
        ):
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)
        repeated = project.load_cycle != "none"
        for w in (
            self.surface_label,
            self.surface_box,
            self.reliability_label,
            self.reliability_box,
        ):
            w.setVisible(repeated)
        for combo, value in (
            (self.surface, project.fatigue_surface),
            (self.reliability, project.fatigue_reliability),
        ):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        for w in (self.n_modes, self.speed, self.speed_unit):
            w.blockSignals(True)
        self.n_modes.setValue(project.n_modes)
        hz = project.operating_frequency
        self.speed.setValue(hz * 60 if self.speed_unit.currentText() == "rpm" else hz)
        for w in (self.n_modes, self.speed, self.speed_unit):
            w.blockSignals(False)
        self.target_box.setVisible(project.analysis in (AnalysisKind.STATIC, AnalysisKind.BUCKLING))
        pack = standards.effective_pack(project)
        values = {
            "ductile": pack.target_ductile,
            "brittle": pack.target_brittle,
            "buckling": pack.target_buckling,
        }
        self.target_default.blockSignals(True)
        self.target_default.setChecked(not project.target_overrides)
        self.target_default.blockSignals(False)
        for key, spin in self.targets.items():
            spin.blockSignals(True)
            spin.setValue(values[key])
            spin.setEnabled(bool(project.target_overrides))
            spin.setVisible(key != "buckling" or project.analysis == AnalysisKind.BUCKLING)
            self.target_box.layout().labelForField(spin).setVisible(spin.isVisible())
            spin.blockSignals(False)
        problems = self.app.readiness()
        lines = []
        for ok, text in problems:
            mark = (
                f"<span style='color:{theme.current().ok}'>✔</span>"
                if ok
                else f"<span style='color:{theme.current().fail}'>✘</span>"
            )
            lines.append(f"{mark} {text}")
        self.checklist.setText("<br>".join(lines))
        self.run.setEnabled(all(ok for ok, _ in problems))
        # how big the run is, once there is a mesh to tell from
        mesh = self.app.mesh
        self.estimate.parentWidget().setVisible(mesh is not None and project is not None)
        if mesh is None or project is None:
            self.estimate.setVisible(False)
        else:
            from kstruct.fem.estimate import solve_estimate

            text, warning = solve_estimate(project, mesh)
            colour = theme.current().warn if warning else theme.current().muted
            self.estimate.setText(f"<span style='color:{colour}'>{text}</span>")
            self.estimate.setVisible(True)

    def _kind_changed(self) -> None:
        self.app.project.analysis = self.kinds.checkedButton().property("kind")
        self.changed.emit()
        self.refresh()

    def _nonlinear_changed(self) -> None:
        project = self.app.project
        project.plasticity = self.plasticity.isChecked()
        project.large_deformation = self.large.isChecked()
        self.changed.emit()
        self.refresh()

    def _cycle_changed(self) -> None:
        self.app.project.load_cycle = self.cycle.currentData()
        self.changed.emit()
        self.refresh()

    def _fatigue_changed(self) -> None:
        project = self.app.project
        project.fatigue_surface = self.surface.currentData()
        project.fatigue_reliability = self.reliability.currentData()
        self.changed.emit()

    def _targets_changed(self) -> None:
        project = self.app.project
        if self.target_default.isChecked():
            project.target_overrides = {}
        else:
            project.target_overrides = {k: spin.value() for k, spin in self.targets.items()}
        self.changed.emit()
        self.refresh()

    def _show_thermal(self, project) -> None:
        self.thermal_box.setVisible(project.analysis == AnalysisKind.THERMAL)
        over_time = project.thermal_duration > 0
        temperature = quantity(system_of(project), "temperature")
        widgets = (self.transient, self.duration, self.duration_unit, self.initial)
        for w in widgets:
            w.blockSignals(True)
        self.transient.setChecked(over_time)
        if over_time:
            seconds = project.thermal_duration
            unit = 2 if seconds >= 3600 and seconds % 3600 == 0 else 1 if seconds >= 60 else 0
            self.duration_unit.setCurrentIndex(unit)
            self.duration.setValue(seconds / self.duration_unit.currentData())
        elif self.duration.value() <= 1:
            self.duration_unit.setCurrentIndex(1)
            self.duration.setValue(10)  # a sensible start: 10 minutes
        self.initial.setSuffix(f" {temperature.unit}")
        self.initial.setValue(temperature.show(project.initial_temperature))
        for w in widgets:
            w.blockSignals(False)
        for w in (self.duration_label, self.duration_row, self.initial_label, self.initial):
            w.setVisible(over_time)

    def _thermal_changed(self) -> None:
        project = self.app.project
        temperature = quantity(system_of(project), "temperature")
        project.thermal_duration = (
            self.duration.value() * self.duration_unit.currentData()
            if self.transient.isChecked()
            else 0.0
        )
        project.initial_temperature = temperature.parse(self.initial.value())
        self._show_thermal(project)
        self.changed.emit()
        self.refresh()

    def _modal_changed(self) -> None:
        project = self.app.project
        project.n_modes = self.n_modes.value()
        v = self.speed.value()
        project.operating_frequency = v / 60 if self.speed_unit.currentText() == "rpm" else v
        self.changed.emit()

    def set_progress(self, frac: float | None, msg: str = "") -> None:
        busy = frac is not None
        self.progress.setVisible(busy)
        self.run.setEnabled(not busy)
        self.cancel.setVisible(busy)
        if busy:
            self.progress.setValue(int(frac * 100))
        self.status.setText(msg)


# --- 7. results -----------------------------------------------------------------------


FIELD_LABELS = [
    ("vm", N_("응력 (폰 미세스)")),
    ("fos", N_("안전율")),
    ("disp", N_("변형량 (합계)")),
    ("ux", N_("변형량 X")),
    ("uy", N_("변형량 Y")),
    ("uz", N_("변형량 Z")),
    ("p1", N_("최대 주응력 (인장)")),
]
SCALE_LABELS = [("auto", N_("자동 확대")), ("true", N_("실제 크기")), ("none", N_("변형 안 보기"))]


class ResultStep(StepPanel):
    def __init__(self, app: MainWindow) -> None:
        super().__init__(app, tr("7. 결과"), "step.result")
        self.card = QFrame()
        self.card.setObjectName("verdictCard")
        card = QVBoxLayout(self.card)
        self.card_title = QLabel()
        self.card_title.setTextFormat(Qt.TextFormat.RichText)
        self.card_title.setWordWrap(True)
        self.card_body = QLabel()
        self.card_body.setWordWrap(True)
        card.addWidget(self.card_title)
        card.addWidget(self.card_body)
        self.add(with_help(self.card, "result.verdict"))
        hotspot = QPushButton(tr("응력 특이점·'특이점 제외' 값이란?"))
        hotspot.setFlat(True)
        hotspot.clicked.connect(lambda: HelpDialog("result.hotspot", self.window()).exec())
        self.add(hotspot)

        form = QFormLayout()
        self.case = QComboBox()
        self.case.currentIndexChanged.connect(app.select_case)
        self.case_label = QLabel(tr("하중 케이스"))
        self.case_box = with_help(self.case, "result.case")
        form.addRow(self.case_label, self.case_box)
        self.field = QComboBox()
        for key, label in FIELD_LABELS:
            self.field.addItem(tr(label), key)
        self.field.currentIndexChanged.connect(app.show_result)
        form.addRow(tr("보기"), with_help(self.field, "result.field"))
        self.scale = QComboBox()
        for key, label in SCALE_LABELS:
            self.scale.addItem(tr(label), key)
        self.scale.currentIndexChanged.connect(app.show_result)
        form.addRow(tr("변형"), with_help(self.scale, "result.scale"))
        self.mode = QComboBox()
        self.mode.currentIndexChanged.connect(app.show_result)
        self.mode_label = QLabel(tr("진동 모드"))
        self.mode_box = with_help(self.mode, "result.mode")
        form.addRow(self.mode_label, self.mode_box)
        self.scale_note = _hint("")
        form.addRow("", self.scale_note)
        self.animate = QCheckBox(tr("움직임 보기 (애니메이션)"))
        self.animate.toggled.connect(app.viewer.set_animated)
        form.addRow("", self.animate)
        self.clip = QCheckBox(tr("단면 보기 (안쪽 값)"))
        self.clip.toggled.connect(self._clip_changed)
        self.clip_axis = QComboBox()
        for label in (tr("X 방향으로 자르기"), tr("Y 방향으로 자르기"), tr("Z 방향으로 자르기")):
            self.clip_axis.addItem(label)
        self.clip_axis.currentIndexChanged.connect(lambda _i: app.show_result())
        self.clip_pos = QSlider(Qt.Orientation.Horizontal)
        self.clip_pos.setRange(1, 99)
        self.clip_pos.setValue(50)
        self.clip_pos.valueChanged.connect(lambda _v: app.show_result())
        form.addRow("", with_help(self.clip, "result.clip"))
        self.clip_row = _row(self.clip_axis, self.clip_pos)
        form.addRow("", self.clip_row)
        self.clip_row.setVisible(False)
        box = QWidget()
        box.setLayout(form)
        self.add(box)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.add(self.summary)
        self.probe = _hint(tr("결과 위를 클릭하면 그 위치의 값을 보여줍니다."))
        self.add(self.probe)

        buttons = QHBoxLayout()
        report = _primary(QPushButton(tr("보고서 저장 (PDF)")))
        report.clicked.connect(app.save_report)
        buttons.addWidget(report)
        buttons.addWidget(HelpButton("result.report"))
        export = QPushButton(tr("결과 내보내기…"))
        export.clicked.connect(app.export_result)
        buttons.addWidget(export)
        buttons.addWidget(HelpButton("result.export"))
        b = QWidget()
        b.setLayout(buttons)
        self.add(b)
        self.layout_.addStretch(1)
        disclaimer = _hint(topic("result.disclaimer").short)
        disclaimer.setProperty("role", "disclaimer")
        self.add(with_help(disclaimer, "result.disclaimer"))

    def set_fields(self, fields: list[tuple[str, str]]) -> None:
        """Repopulate the field selector for the current result type, keeping the choice."""
        current = self.field.currentData()
        self.field.blockSignals(True)
        self.field.clear()
        for key, label in fields:
            self.field.addItem(tr_data(label), key)  # N_ texts, or already translated
        self.field.setCurrentIndex(max(0, self.field.findData(current)))
        self.field.blockSignals(False)

    def set_cases(self, labels: list[str] | None, current: int = 0) -> None:
        visible = bool(labels) and len(labels) > 1
        self.case_box.setVisible(visible)
        self.case_label.setVisible(visible)
        self.case.blockSignals(True)
        self.case.clear()
        for text in labels or []:
            self.case.addItem(text)
        if labels:
            self.case.setCurrentIndex(min(current, len(labels) - 1))
        self.case.blockSignals(False)

    def set_modes(self, frequencies, labels: list[str] | None = None, title=None) -> None:
        """Fill the mode selector (vibration / buckling modes, or the moments of a
        transient heat run); None hides it."""
        visible = frequencies is not None
        self.mode_box.setVisible(visible)
        self.mode_label.setVisible(visible)
        self.mode_label.setText(tr("진동 모드") if title is None else tr_data(title))
        if not visible:
            return
        current = max(0, self.mode.currentIndex())
        self.mode.blockSignals(True)
        self.mode.clear()
        for i, f in enumerate(frequencies):
            text = (
                labels[i]
                if labels
                else tr("{i}차: {f:,.1f} Hz ({rpm:,.0f} rpm)", i=i + 1, f=f, rpm=f * 60)
            )
            self.mode.addItem(text, i)
        self.mode.setCurrentIndex(min(current, len(frequencies) - 1))
        self.mode.blockSignals(False)

    def _clip_changed(self, on: bool) -> None:
        self.clip_row.setVisible(on)
        self.animate.setEnabled(not on)
        if on:
            self.animate.setChecked(False)
        self.app.show_result()

    def clip_fraction(self) -> tuple[int, float] | None:
        """(axis index, 0..1 position along it) when the section view is on."""
        if not self.clip.isChecked():
            return None
        return self.clip_axis.currentIndex(), self.clip_pos.value() / 100.0

    def field_key(self) -> str:
        return self.field.currentData()

    def scale_key(self) -> str:
        return self.scale.currentData()
