"""Plain-language summary of a static result (shared by the result panel and the report)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    title = VERDICT_STYLE[worst.verdict][0]
    return Summary(
        parts=parts,
        verdict=worst.verdict,
        headline=f"{title} — 최소 안전율 {worst.min_fos:.2f}",
        explanation=explanation,
        notes=notes,
        max_disp=float(disp[max_node]),
        max_disp_at=mesh.nodes[max_node],
        reactions=result.reactions,
    )
