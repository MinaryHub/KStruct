"""Plain-language summary of a static result (shared by the result panel and the report)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kstruct.fem.fatigue import INFINITE_LIFE, LoadCycle, fatigue_check
from kstruct.fem.static import StaticResult
from kstruct.model import Project
from kstruct.standards import StandardPack, Verdict, verdict

VERDICT_STYLE = {
    Verdict.OK: ("✅ 안전", "#2e9e5b", "#e8f6ee"),
    Verdict.WARN: ("⚠ 주의", "#b9770e", "#fdf3e1"),
    Verdict.FAIL: ("❌ 위험", "#c0392b", "#fbeaea"),
}
_RANK = {Verdict.OK: 0, Verdict.WARN: 1, Verdict.FAIL: 2}


@dataclass
class PartSummary:
    part_id: int
    name: str
    material: str
    ductile: bool
    max_stress: float  # von Mises (ductile) or max principal (brittle), MPa
    max_stress_at: np.ndarray
    capacity: float  # yield or tensile strength, MPa
    min_fos: float
    target: float
    verdict: Verdict
    max_disp: float
    at_fixture: bool  # peak sits on a fixture face (likely a stress singularity)
    fatigue_fos: float | None = None
    fatigue_life: float | None = None  # cycles at the weakest node
    fatigue_verdict: Verdict | None = None


@dataclass
class Summary:
    parts: list[PartSummary]
    verdict: Verdict
    headline: str
    explanation: str
    notes: list[str]
    max_disp: float
    max_disp_at: np.ndarray
    reactions: list[np.ndarray]
    fatigue: object | None = None  # fem.fatigue.FatigueResult


def summarize(project: Project, result: StaticResult, pack: StandardPack) -> Summary:
    mesh = result.mesh
    vm = result.von_mises
    p1 = result.principal[:, 0]
    fos = result.safety_factor
    disp = result.displacement_magnitude
    fixture_nodes = mesh.face_nodes([r for fx in project.fixtures for r in fx.faces])

    parts = []
    for pid, m in result.materials.items():
        nodes = mesh.part_nodes(pid)
        demand = vm if m.ductile else p1
        peak = nodes[int(np.argmax(demand[nodes]))]
        weakest = nodes[int(np.argmin(fos[nodes]))]
        target = pack.target(m)
        parts.append(
            PartSummary(
                part_id=pid,
                name=project.part(pid).name,
                material=pack.material_name(m),
                ductile=m.ductile,
                max_stress=float(demand[peak]),
                max_stress_at=mesh.nodes[peak],
                capacity=m.yield_strength if m.ductile else m.tensile_strength,
                min_fos=float(fos[weakest]),
                target=target,
                verdict=verdict(float(fos[weakest]), target),
                max_disp=float(disp[nodes].max()),
                at_fixture=bool(np.isin(peak, fixture_nodes)),
            )
        )

    fatigue = None
    cycle = LoadCycle(project.load_cycle)
    if cycle != LoadCycle.NONE:
        fatigue = fatigue_check(result, cycle)
        for s in parts:
            nodes = mesh.part_nodes(s.part_id)
            weakest = nodes[int(np.argmin(fatigue.safety_factor[nodes]))]
            s.fatigue_fos = float(fatigue.safety_factor[weakest])
            s.fatigue_life = float(fatigue.life[nodes].min())
            s.fatigue_verdict = verdict(s.fatigue_fos, s.target)

    worst = max(parts, key=lambda s: (_RANK[s.verdict], -s.min_fos))
    failure = "영구적으로 휘어질" if worst.ductile else "깨질"
    if worst.verdict == Verdict.OK:
        explanation = (
            f"가장 약한 곳({worst.name})도 재료가 견딜 수 있는 힘의 {worst.min_fos:.1f}배 "
            f"여유가 있습니다. 권장 안전율 {worst.target:.1f} 이상을 만족합니다."
        )
    elif worst.verdict == Verdict.WARN:
        explanation = (
            f"부품은 견디지만 여유가 부족합니다(안전율 {worst.min_fos:.2f}, 권장 {worst.target:.1f} "
            f"이상). 두께를 늘리거나, 모서리를 둥글게 하거나, 더 강한 재료를 검토하세요."
        )
    else:
        explanation = (
            f"하중을 견디지 못하고 {failure} 수 있습니다(안전율 {worst.min_fos:.2f} < 1). "
            f"형상이나 재료를 바꿔야 합니다."
        )

    if fatigue is not None:
        weak = max(parts, key=lambda s: (_RANK[s.fatigue_verdict], -s.fatigue_fos))
        life = (
            "무한 수명(피로 한도 이하)"
            if weak.fatigue_life >= INFINITE_LIFE
            else f"약 {weak.fatigue_life:,.0f}회"
        )
        kind = "한 방향" if cycle == LoadCycle.PULSATING else "양방향"
        fatigue_text = (
            f" 반복 하중({kind}) 피로 안전율은 {weak.fatigue_fos:.2f}, 예상 수명은 {life}입니다."
        )
        if _RANK[weak.fatigue_verdict] > _RANK[worst.verdict]:
            worst_verdict, worst_fos = weak.fatigue_verdict, weak.fatigue_fos
            if weak.fatigue_verdict == Verdict.FAIL:
                explanation = (
                    f"한 번의 하중은 견디지만, 반복되면 피로로 균열이 생길 수 있습니다"
                    f"(예상 수명 {life})."
                )
            else:
                explanation = "한 번의 하중은 충분히 견디지만 반복 하중에 대한 여유가 부족합니다."
        else:
            worst_verdict, worst_fos = worst.verdict, worst.min_fos
        explanation += fatigue_text
    else:
        worst_verdict, worst_fos = worst.verdict, worst.min_fos

    notes = []
    if any(s.at_fixture for s in parts):
        notes.append(
            "최대 응력이 고정한 면 가장자리에 있습니다. 완전 고정의 경계에서는 응력이 실제보다 "
            "크게 계산되는 경우가 많습니다(응력 특이점). 고정면에서 조금 떨어진 곳의 값도 확인하세요."
        )
    size = float(np.ptp(mesh.nodes, axis=0).max())
    max_node = int(np.argmax(disp))
    if disp[max_node] > 0.05 * size:
        notes.append(
            "변형이 부품 크기의 5%를 넘습니다. 이 프로그램의 선형 해석은 작은 변형을 가정하므로 "
            "실제 거동과 차이가 클 수 있습니다."
        )

    title = VERDICT_STYLE[worst_verdict][0]
    label = "피로 안전율" if worst_fos != worst.min_fos else "최소 안전율"
    return Summary(
        parts=parts,
        verdict=worst_verdict,
        headline=f"{title} — {label} {worst_fos:.2f}",
        explanation=explanation,
        notes=notes,
        max_disp=float(disp[max_node]),
        max_disp_at=mesh.nodes[max_node],
        reactions=result.reactions,
        fatigue=fatigue,
    )


# --- modal ----------------------------------------------------------------------------

RESONANCE_FAIL = 0.10  # natural frequency within ±10 % of the excitation
RESONANCE_WARN = 0.20


@dataclass
class ModalSummary:
    verdict: Verdict | None  # None when no operating frequency was given
    headline: str
    explanation: str
    notes: list[str]


def summarize_modal(project: Project, result) -> ModalSummary:
    freqs = result.frequencies
    f_op = project.operating_frequency
    notes = []
    if result.rigid_modes:
        notes.append(
            "고정이 없어 부품이 공중에 떠 있는 상태로 계산했습니다(자유 진동). "
            "0 Hz에 해당하는 움직임 6개는 제외했습니다."
        )
    if f_op <= 0:
        return ModalSummary(
            None,
            f"가장 낮은 고유진동수 {freqs[0]:,.1f} Hz",
            "부품이 스스로 잘 흔들리는 진동수입니다. 모터·펌프 등 주변 진동원의 진동수가 "
            "이 값들과 가까우면 공진으로 크게 흔들릴 수 있습니다. "
            "6단계에서 '운전 진동수'를 입력하면 공진 위험을 판정해 드립니다.",
            notes,
        )
    gaps = np.abs(freqs - f_op) / f_op
    closest = int(np.argmin(gaps))
    gap = float(gaps[closest])
    if gap <= RESONANCE_FAIL:
        verdict = Verdict.FAIL
        text = (
            f"{closest + 1}차 고유진동수 {freqs[closest]:,.1f} Hz가 운전 진동수 {f_op:,.1f} Hz와 "
            f"{gap * 100:.0f}% 차이로 매우 가깝습니다. 공진으로 크게 흔들리거나 피로 파손될 수 "
            "있습니다. 두께·리브로 강성을 바꾸거나 운전 속도를 조정하세요."
        )
    elif gap <= RESONANCE_WARN:
        verdict = Verdict.WARN
        text = (
            f"{closest + 1}차 고유진동수 {freqs[closest]:,.1f} Hz가 운전 진동수 {f_op:,.1f} Hz와 "
            f"{gap * 100:.0f}% 차이입니다. 공진 여유(보통 ±20% 이상)가 부족합니다."
        )
    else:
        verdict = Verdict.OK
        text = (
            f"모든 고유진동수가 운전 진동수 {f_op:,.1f} Hz와 20% 이상 떨어져 있어 "
            f"공진 위험이 낮습니다(가장 가까운 것: {closest + 1}차 {freqs[closest]:,.1f} Hz)."
        )
    if f_op > freqs[-1] * (1 + RESONANCE_WARN):
        notes.append(
            f"운전 진동수가 계산한 {len(freqs)}개 진동수보다 높습니다. 더 높은 차수의 진동수는 "
            "확인하지 않았습니다."
        )
    title = VERDICT_STYLE[verdict][0]
    headline = f"{title} — 1차 고유진동수 {freqs[0]:,.1f} Hz (운전 {f_op:,.1f} Hz)"
    return ModalSummary(verdict, headline, text, notes)


# --- thermal --------------------------------------------------------------------------


@dataclass
class ThermalSummary:
    verdict: Verdict | None
    headline: str
    explanation: str
    notes: list[str]


def summarize_thermal(project: Project, result, materials: dict) -> ThermalSummary:
    T = result.temperature
    mesh = result.mesh
    worst, worst_ratio, limit_text = None, -np.inf, ""
    for pid, m in materials.items():
        if m.max_service_temperature is None:
            continue
        t_max = float(T[mesh.part_nodes(pid)].max())
        ratio = t_max / m.max_service_temperature
        if ratio > worst_ratio:
            worst, worst_ratio = (pid, m, t_max), ratio
    notes = []
    if result.heat_in > 0:
        balance = abs(result.heat_out - result.heat_in) / result.heat_in
        if balance > 0.02:
            notes.append(
                f"들어온 열 {result.heat_in:.3g} W와 나간 열 {result.heat_out:.3g} W의 차이가 "
                f"{balance * 100:.0f}%입니다. 메시를 더 세밀하게 하면 정확해집니다."
            )
    headline = f"최고 {T.max():,.1f} °C · 최저 {T.min():,.1f} °C"
    if worst is None:
        return ThermalSummary(None, headline, "온도 분포를 계산했습니다.", notes)
    pid, m, t_max = worst
    limit = m.max_service_temperature
    name = project.part(pid).name
    if t_max <= 0.9 * limit:
        verdict = Verdict.OK
        limit_text = (
            f"가장 뜨거운 곳도 {name} 재료의 권장 최고 사용온도({limit:.0f} °C)보다 "
            "충분히 낮습니다."
        )
    elif t_max <= limit:
        verdict = Verdict.WARN
        limit_text = (
            f"{name}의 최고 온도 {t_max:.0f} °C가 재료의 권장 최고 사용온도({limit:.0f} °C)에 "
            "가깝습니다. 냉각을 늘리거나 내열성이 높은 재료를 검토하세요."
        )
    else:
        verdict = Verdict.FAIL
        limit_text = (
            f"{name}의 최고 온도 {t_max:.0f} °C가 재료의 권장 최고 사용온도({limit:.0f} °C)를 "
            "넘습니다. 강도가 떨어지거나 변형·변색될 수 있습니다."
        )
    title = VERDICT_STYLE[verdict][0]
    return ThermalSummary(verdict, f"{title} — {headline}", limit_text, notes)


# --- buckling -------------------------------------------------------------------------


@dataclass
class BucklingSummary:
    verdict: Verdict
    headline: str
    explanation: str
    notes: list[str]


def summarize_buckling(project: Project, result, pack: StandardPack) -> BucklingSummary:
    lam = float(result.factors[0])
    target = pack.target_buckling
    v = verdict(lam, target)
    if v == Verdict.OK:
        text = (
            f"지금 하중의 {lam:,.1f}배가 되어야 좌굴(갑자기 옆으로 휘며 무너짐)이 일어납니다. "
            f"권장 여유 {target:.1f}배 이상을 만족합니다."
        )
    elif v == Verdict.WARN:
        text = (
            f"지금 하중의 {lam:,.2f}배에서 좌굴이 일어납니다. 견디기는 하지만 권장 여유 "
            f"{target:.1f}배에 못 미칩니다. 두께를 늘리거나 리브·지지를 추가하세요."
        )
    else:
        text = (
            f"지금 하중의 {lam:,.2f}배, 즉 이 하중에 도달하기 전에 좌굴로 무너집니다. "
            "단면을 키우거나 길이를 줄이거나 중간 지지를 추가해야 합니다."
        )
    notes = [
        "선형 좌굴 해석은 완벽한 형상을 가정하므로 실제 좌굴 하중은 이보다 낮을 수 있습니다"
        "(제작 오차·편심). 그래서 일반 강도보다 큰 여유(보통 3배 이상)를 권장합니다."
    ]
    static_fos = float(result.static.safety_factor.min())
    if static_fos < lam:
        notes.append(
            f"참고: 이 하중에서 강도 안전율은 {static_fos:.2f}입니다. 좌굴보다 재료 항복이 먼저 "
            "일어날 수 있으니 '강도' 해석도 확인하세요."
        )
    title = VERDICT_STYLE[v][0]
    return BucklingSummary(v, f"{title} — 좌굴 하중 배수 {lam:,.2f}", text, notes)
