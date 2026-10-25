"""Secondary dialogs: user-defined material editor, About box, bug-report bundle."""

from __future__ import annotations

import datetime as dt
import json
import logging
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from kstruct import __version__, standards
from kstruct.model import Material
from kstruct.paths import environment_summary, log_path, notices_text, user_materials_path
from kstruct.ui.help import with_help
from kstruct.units import quantity

log = logging.getLogger(__name__)


class MaterialDialog(QDialog):
    """Create or edit a user-defined material. Values are entered in the project's display
    units for stress and temperature; everything else in everyday units (kg/m³, J/kg·K)."""

    def __init__(self, base: Material | None, system: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("사용자 재료")
        self.setMinimumWidth(460)
        self._system = system
        stress, temp = quantity(system, "stress"), quantity(system, "temperature")
        self._stress, self._temp = stress, temp
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "재료 성적서(mill sheet)나 제조사 데이터시트의 값을 넣으세요. 항복강도는 실제보다 "
                "높게 넣지 않는 것이 안전합니다."
            )
        )
        form = QFormLayout()
        self.name = QLineEdit(base.name if base else "")
        form.addRow("이름", with_help(self.name, "material.custom_name"))
        self.category = QComboBox()
        for key, label in standards.CATEGORY_NAMES.items():
            self.category.addItem(label, key)
        if base:
            self.category.setCurrentIndex(max(0, self.category.findData(base.category)))
        form.addRow("종류", self.category)

        def spin(value, lo, hi, decimals, suffix):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSuffix(f" {suffix}")
            box.setValue(value)
            return box

        b = base
        self.E = spin((b.youngs_modulus if b else 200_000) / 1000, 0.01, 2000, 1, "GPa")
        self.nu = spin(b.poisson_ratio if b else 0.3, 0.0, 0.499, 3, "")
        self.density = spin(b.density * 1e12 if b else 7850, 1, 30000, 0, "kg/m³")
        self.yield_ = spin(stress.show(b.yield_strength if b else 250), 0.1, 1e7, 1, stress.unit)
        self.uts = spin(stress.show(b.tensile_strength if b else 400), 0.1, 1e7, 1, stress.unit)
        self.alpha = spin((b.thermal_expansion if b else 12e-6) * 1e6, 0, 1000, 2, "×10⁻⁶/K")
        self.k = spin(b.thermal_conductivity if b else 50, 0, 2000, 2, "W/(m·K)")
        self.cp = spin((b.specific_heat * 1e-6) if b else 460, 0, 20000, 0, "J/(kg·K)")
        self.fatigue = spin(
            stress.show(b.fatigue_limit) if b and b.fatigue_limit else 0, 0, 1e7, 1, stress.unit
        )
        self.tmax = spin(
            temp.show(b.max_service_temperature) if b and b.max_service_temperature else 0,
            -300,
            5000,
            0,
            temp.unit,
        )
        self.ductile = QCheckBox("연성재 (휘다가 파손; 주철·다이캐스팅·세라믹은 해제)")
        self.ductile.setChecked(b.ductile if b else True)
        form.addRow("탄성계수 E", with_help(self.E, "material.custom_props"))
        form.addRow("포아송비 ν", self.nu)
        form.addRow("밀도", self.density)
        form.addRow("항복강도", self.yield_)
        form.addRow("인장강도", self.uts)
        form.addRow("열팽창계수", self.alpha)
        form.addRow("열전도율", self.k)
        form.addRow("비열", self.cp)
        form.addRow("피로 한도 (0 = 인장강도의 절반으로 추정)", self.fatigue)
        form.addRow("최고 사용온도 (0 = 판정 안 함)", self.tmax)
        form.addRow("", self.ductile)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("저장")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._id = b.id if b and b.id.startswith("user_") else standards.new_material_id()

    def _accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "확인이 필요합니다", "재료 이름을 입력하세요.")
            return
        if self.uts.value() < self.yield_.value():
            QMessageBox.warning(
                self, "확인이 필요합니다", "인장강도는 항복강도보다 작을 수 없습니다."
            )
            return
        self.accept()

    def material(self) -> Material:
        s, t = self._stress, self._temp
        return Material(
            id=self._id,
            name=self.name.text().strip(),
            category=self.category.currentData(),
            youngs_modulus=self.E.value() * 1000,
            poisson_ratio=self.nu.value(),
            density=self.density.value() * 1e-12,
            yield_strength=s.parse(self.yield_.value()),
            tensile_strength=s.parse(self.uts.value()),
            thermal_expansion=self.alpha.value() * 1e-6,
            thermal_conductivity=self.k.value(),
            specific_heat=self.cp.value() * 1e6,
            fatigue_limit=s.parse(self.fatigue.value()) if self.fatigue.value() > 0 else None,
            max_service_temperature=t.parse(self.tmax.value()) if self.tmax.value() != 0 else None,
            ductile=self.ductile.isChecked(),
            designations={},
        )


# --- user material library (persists across projects) -------------------------------


def load_user_library() -> dict[str, Material]:
    path = user_materials_path()
    try:
        records = json.loads(path.read_text("utf-8")) if path.is_file() else []
        return {m["id"]: standards.material_from_record(m) for m in records}
    except (OSError, ValueError, KeyError) as e:
        log.warning("user material library unreadable (%s): %s", path, e)
        return {}


def save_user_library(materials: dict[str, Material]) -> None:
    path = user_materials_path()
    try:
        records = [standards.material_record(m) for m in materials.values()]
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        log.warning("cannot write user material library (%s): %s", path, e)


# --- about / bug report -----------------------------------------------------------------


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KStruct 정보")
        self.resize(640, 480)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        about = QWidget()
        box = QVBoxLayout(about)
        text = QLabel(
            f"<h2>KStruct {__version__}</h2>"
            "<p>기계 부품용 3D 유한요소해석 — 비전문가를 위한 7단계 강도·진동·열·좌굴 검토</p>"
            f"<pre>{environment_summary()}</pre>"
            "<p>해석 결과는 설계 검토를 돕는 참고 자료입니다. 실제 제작·사용 여부는 자격을 갖춘 "
            "기술자가 최종 판단해야 합니다.</p>"
        )
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(text)
        box.addStretch(1)
        tabs.addTab(about, "정보")
        notices = QPlainTextEdit(notices_text())
        notices.setReadOnly(True)
        tabs.addTab(notices, "오픈소스 고지")
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("닫기")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def write_bug_report(path: Path, project_file: Path | None, extra: dict[str, str]) -> Path:
    """Zip the log, environment info, and (if given) the project file for support."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("environment.txt", environment_summary())
        for key, value in extra.items():
            z.writestr(key, value)
        lp = log_path()
        for p in sorted(lp.parent.glob(lp.name + "*")):
            if p.is_file():
                z.write(p, p.name)
        if project_file is not None and project_file.is_file():
            z.write(project_file, project_file.name)
    return path


def save_bug_report(parent: QWidget, project_file: Path | None, extra: dict[str, str]) -> None:
    default = f"kstruct_문제보고_{dt.datetime.now():%Y%m%d_%H%M}.zip"
    chosen, _ = QFileDialog.getSaveFileName(parent, "문제 보고 파일 저장", default, "ZIP (*.zip)")
    if not chosen:
        return
    write_bug_report(Path(chosen), project_file, extra)
    QMessageBox.information(
        parent,
        "문제 보고 파일",
        f"저장했습니다:\n{chosen}\n\n이 파일을 지원 담당자에게 보내 주세요. "
        "로그와 프로젝트 파일(입력 조건·형상)이 들어 있습니다.",
    )
