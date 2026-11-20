"""Plain-language summary of a static result (shared by the result panel and the report)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kstruct.fem.fatigue import (
    INFINITE_LIFE,
    SURFACE_FACTORS,
    FatigueSettings,
    LoadCycle,
    fatigue_check,
)
from kstruct.fem.singularity import CONCAVE_EDGE, KIND_NAMES, NONE, find_singularities
from kstruct.fem.static import StaticResult
from kstruct.i18n import N_, tr, tr_data
from kstruct.model import ConnectionKind, FixtureKind, Project
from kstruct.standards import StandardPack, Verdict, verdict
from kstruct.units import fmt, system_of


class _VerdictStyles(dict):
    """{verdict: (label, color, background)}; the label is translated when looked up."""

    def __getitem__(self, key):
        label, color, background = super().__getitem__(key)
        return tr(label), color, background


VERDICT_STYLE = _VerdictStyles(
    {
        Verdict.OK: (N_("✅ 안전"), "#2e9e5b", "#e8f6ee"),
        Verdict.WARN: (N_("⚠ 주의"), "#b9770e", "#fdf3e1"),
        Verdict.FAIL: (N_("❌ 위험"), "#c0392b", "#fbeaea"),
    }
)
_RANK = {Verdict.OK: 0, Verdict.WARN: 1, Verdict.FAIL: 2}
# bolts are tightened to ~70 % of proof load on purpose, so their margin is judged against
# proof strength with a small reserve rather than the 1.5 used for parts
BOLT_TARGET = 1.1


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
    at_fixture: bool  # peak sits on a fully fixed face (likely a stress singularity)
    crushing: bool = False  # brittle part governed by compression (stress = compressive)
    # when the peak sits in a stress-singularity zone: the kind of singularity and the
    # "hot-spot" value just outside the zone, which does not grow with mesh refinement
    singular_kind: str | None = None
    hotspot_stress: float | None = None
    hotspot_fos: float | None = None
    hotspot_verdict: Verdict | None = None
    fatigue_fos: float | None = None
    fatigue_life: float | None = None  # cycles at the weakest node
    fatigue_verdict: Verdict | None = None
    fatigue_excludes_singularity: bool = False  # fatigue values taken outside the zone
    plastic_strain: float | None = None  # nonlinear run with yielding: governing peeq

    @property
    def governing_fos(self) -> float:
        """Safety factor the verdict is based on: for a ductile material whose peak is a
        singularity, the hot-spot value (the peak would only yield locally and its number
        depends on the mesh); otherwise the raw minimum (brittle materials crack at sharp
        corners, so the peak is kept)."""
        if self.hotspot_fos is not None and self.ductile:
            return self.hotspot_fos
        return self.min_fos


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
    bolts: list = field(default_factory=list)  # fem.bolts.BoltResult
    governing_fos: float = 0.0  # the number in the headline


def summarize(project: Project, result: StaticResult, pack: StandardPack) -> Summary:
    mesh = result.mesh
    vm = result.von_mises
    p1 = result.principal[:, 0]
    fos = result.safety_factor
    disp = result.displacement_magnitude
    # only a fully fixed face clamps the material (rollers and pins let it breathe)
    fixture_nodes = mesh.face_nodes(
        [r for fx in project.fixtures if fx.kind == FixtureKind.FIXED for r in fx.faces]
    )
    singular = find_singularities(project, mesh)

    parts = []
    for pid, m in result.materials.items():
        nodes = mesh.part_nodes(pid)
        demand = vm if m.ductile else p1
        capacity = m.yield_strength if m.ductile else m.tensile_strength
        peak = nodes[int(np.argmax(demand[nodes]))]
        weakest = nodes[int(np.argmin(fos[nodes]))]
        target = pack.target(m)
        # a brittle part governed by compression: report the compressive stress and
        # strength the safety factor was computed from
        crushing = False
        if not m.ductile:
            crush_capacity = m.compressive_strength or m.tensile_strength
            p3 = result.principal[:, 2]
            if -p3[weakest] / crush_capacity > max(p1[weakest], 0.0) / capacity:
                crushing, demand, capacity, peak = True, -p3, crush_capacity, weakest
        part = PartSummary(
            part_id=pid,
            name=project.part(pid).name,
            material=pack.material_name(m),
            ductile=m.ductile,
            max_stress=float(demand[peak]),
            max_stress_at=mesh.nodes[peak],
            capacity=capacity,
            crushing=crushing,
            min_fos=float(fos[weakest]),
            target=target,
            verdict=verdict(float(fos[weakest]), target),
            max_disp=float(disp[nodes].max()),
            at_fixture=bool(np.isin(peak, fixture_nodes)),
        )
        zone = singular.zone(mesh, pid)
        outside = nodes[~zone[nodes]]
        if zone[peak] and len(outside):
            kind = singular.nearest_kind(mesh, pid, int(peak))
            if kind != NONE:
                hot = float(demand[outside].max())
                hot_fos = float(fos[outside].min())
                part.singular_kind = KIND_NAMES[kind]
                part.hotspot_stress = hot
                part.hotspot_fos = hot_fos
                part.hotspot_verdict = verdict(hot_fos, target)
                part.verdict = verdict(part.governing_fos, target)
        parts.append(part)

    fatigue = None
    cycle = LoadCycle(project.load_cycle)
    nonlinear = getattr(result, "nonlinear", None)
    if cycle != LoadCycle.NONE and nonlinear is not None:
        cycle = LoadCycle.NONE  # stresses beyond yield: the fatigue model does not apply
        skipped_fatigue = True
    else:
        skipped_fatigue = False
    if cycle != LoadCycle.NONE:
        settings = FatigueSettings(project.fatigue_surface, project.fatigue_reliability)
        fatigue = fatigue_check(result, cycle, settings)
        for s in parts:
            nodes = mesh.part_nodes(s.part_id)
            # Like the static check, the fatigue check of a ductile part is taken outside
            # a singular zone: the stress at a sharp corner grows with every mesh
            # refinement, and so did the fatigue verdict (1.50 / 1.36 / 1.19 on the sample
            # bracket). The note below says that the corner itself is not assessed.
            if s.ductile and s.singular_kind is not None:
                zone = singular.zone(mesh, s.part_id)
                outside = nodes[~zone[nodes]]
                if len(outside):
                    nodes, s.fatigue_excludes_singularity = outside, True
            weakest = nodes[int(np.argmin(fatigue.safety_factor[nodes]))]
            s.fatigue_fos = float(fatigue.safety_factor[weakest])
            s.fatigue_life = float(fatigue.life[nodes].min())
            s.fatigue_verdict = verdict(s.fatigue_fos, s.target)

    worst = max(parts, key=lambda s: (_RANK[s.verdict], -s.governing_fos))
    worst_fos = worst.governing_fos
    if worst.verdict == Verdict.OK:
        explanation = tr(
            "가장 약한 곳({name})도 재료가 견딜 수 있는 힘의 {fos:.1f}배 "
            "여유가 있습니다. 권장 안전율 {target:.1f} 이상을 만족합니다.",
            name=tr_data(worst.name),
            fos=worst_fos,
            target=worst.target,
        )
    elif worst.verdict == Verdict.WARN:
        explanation = tr(
            "부품은 견디지만 여유가 부족합니다(안전율 {fos:.2f}, 권장 {target:.1f} "
            "이상). 두께를 늘리거나, 모서리를 둥글게 하거나, 더 강한 재료를 검토하세요.",
            fos=worst_fos,
            target=worst.target,
        )
    elif worst.ductile:
        explanation = tr(
            "하중을 견디지 못하고 영구적으로 휘어질 수 있습니다(안전율 {fos:.2f} < 1). "
            "형상이나 재료를 바꿔야 합니다.",
            fos=worst_fos,
        )
    else:
        explanation = tr(
            "하중을 견디지 못하고 깨질 수 있습니다(안전율 {fos:.2f} < 1). "
            "형상이나 재료를 바꿔야 합니다.",
            fos=worst_fos,
        )

    if fatigue is not None:
        weak = max(parts, key=lambda s: (_RANK[s.fatigue_verdict], -s.fatigue_fos))
        life = (
            tr("무한 수명(피로 한도 이하)")
            if weak.fatigue_life >= INFINITE_LIFE
            else tr("약 {n:,.0f}회", n=weak.fatigue_life)
        )
        kind = tr("한 방향") if cycle == LoadCycle.PULSATING else tr("양방향")
        fatigue_text = tr(
            " 반복 하중({kind}) 피로 안전율은 {fos:.2f}, 예상 수명은 {life}입니다.",
            kind=kind,
            fos=weak.fatigue_fos,
            life=life,
        )
        if _RANK[weak.fatigue_verdict] > _RANK[worst.verdict]:
            worst_verdict, worst_fos = weak.fatigue_verdict, weak.fatigue_fos
            if weak.fatigue_verdict == Verdict.FAIL:
                explanation = tr(
                    "한 번의 하중은 견디지만, 반복되면 피로로 균열이 생길 수 있습니다"
                    "(예상 수명 {life}).",
                    life=life,
                )
            else:
                explanation = tr(
                    "한 번의 하중은 충분히 견디지만 반복 하중에 대한 여유가 부족합니다."
                )
        else:
            worst_verdict = worst.verdict
        explanation += fatigue_text
    else:
        worst_verdict = worst.verdict

    for b in result.bolts or []:
        bv = verdict(b.safety_factor, BOLT_TARGET)
        if _RANK[bv] > _RANK[worst_verdict]:
            worst_verdict, worst_fos = bv, b.safety_factor
            explanation = tr(
                "부품은 견디지만 {name}({size} {grade})에 걸리는 힘이 볼트 강도에 비해 큽니다"
                "(안전율 {fos:.2f}). 더 굵거나 등급이 높은 볼트, 또는 볼트 추가를 검토하세요.",
                name=tr_data(b.name),
                size=b.size,
                grade=b.grade,
                fos=b.safety_factor,
            )

    notes = []
    us = system_of(project)
    if fatigue is not None and fatigue.factors:
        surface = tr(SURFACE_FACTORS[project.fatigue_surface][0])
        parts_text = ", ".join(
            f"{tr_data(project.part(pid).name)} {f['total']:.2f}"
            for pid, f in fatigue.factors.items()
        )
        any_f = next(iter(fatigue.factors.values()))
        notes.append(
            tr(
                "피로 한도 보정: 표면 '{surface}', 크기 계수 {kb:.2f}, "
                "신뢰도 {reliability}%(계수 {kc:.2f}) → 재료 피로 한도의 "
                "{factors}배를 사용했습니다. 노치·용접부는 따로 고려하지 않았습니다.",
                surface=surface,
                kb=any_f["kb"],
                reliability=project.fatigue_reliability,
                kc=any_f["kc"],
                factors=parts_text,
            )
        )
    for s in parts:
        if s.crushing:
            assumed = (
                ""
                if result.materials[s.part_id].compressive_strength
                else tr(
                    " 이 재료에는 압축 강도 값이 없어 인장 강도와 같다고 (안전한 쪽으로) 보았습니다."
                )
            )
            notes.append(
                tr(
                    "{name}: 취성재가 눌리는 쪽에서 가장 약합니다. 표시한 응력은 압축 응력 "
                    "{stress}, 강도는 압축 강도 {strength}입니다.{assumed}",
                    name=tr_data(s.name),
                    stress=fmt(us, "stress", s.max_stress, ",.1f"),
                    strength=fmt(us, "stress", s.capacity, ",.0f"),
                    assumed=assumed,
                )
            )
    if any(s.fatigue_excludes_singularity for s in parts):
        notes.append(
            tr(
                "피로 안전율과 수명은 응력 특이점(날카로운 모서리·고정면 가장자리) 주변을 제외한 "
                "값입니다. 날카로운 모서리는 실제로 피로 균열이 시작되기 쉬운 곳이므로, 반복 하중을 "
                "받는 부품은 그 모서리에 필렛(둥글림)을 주고 다시 확인하세요."
            )
        )
    for s in parts:
        if s.singular_kind is not None:
            r = singular.radius[s.part_id]
            advice = (
                tr("모서리에 필렛(둥글림)을 주면 실제 응력이 크게 낮아집니다.")
                if s.singular_kind == KIND_NAMES[CONCAVE_EDGE]
                else tr("실제로는 고정부가 조금 탄성을 가지므로 이보다 낮은 경우가 많습니다.")
            )
            basis = (
                tr("연성재이므로 판정은 특이점을 제외한 값을 기준으로 했습니다.")
                if s.ductile
                else tr("취성재는 날카로운 모서리에서 실제로 깨질 수 있어 최대값으로 판정했습니다.")
            )
            notes.append(
                tr(
                    "{name}: 최대 응력 {stress}가 {kind}에 있습니다. "
                    "이런 곳은 메시를 세밀하게 할수록 계산값이 한없이 커지는 '응력 특이점'이라 "
                    "그 숫자를 그대로 믿기 어렵습니다. 특이점에서 {distance} 떨어진 곳 기준으로는 "
                    "응력 {hotspot}, 안전율 {fos:.2f}입니다. {basis} {advice}",
                    name=tr_data(s.name),
                    stress=fmt(us, "stress", s.max_stress, ",.1f"),
                    kind=tr(s.singular_kind),
                    distance=fmt(us, "length", r, ".3g"),
                    hotspot=fmt(us, "stress", s.hotspot_stress, ",.1f"),
                    fos=s.hotspot_fos,
                    basis=basis,
                    advice=advice,
                )
            )
        elif s.at_fixture:
            notes.append(
                tr(
                    "{name}: 최대 응력이 고정한 면 위에 있습니다. 완전 고정의 경계에서는 응력이 "
                    "실제보다 크게 계산되는 경우가 많습니다(응력 특이점). 고정면에서 조금 떨어진 곳의 "
                    "값도 확인하세요.",
                    name=tr_data(s.name),
                )
            )
    bolt_nodes = result.bolt_nodes if result.bolt_nodes is not None else np.zeros(0, int)
    if len(bolt_nodes):
        # within about one element of a head / nut bearing area
        tets = mesh.tets
        h = float(np.linalg.norm(mesh.nodes[tets[:, 1]] - mesh.nodes[tets[:, 0]], axis=1).mean())
        bearing = mesh.nodes[bolt_nodes]
        if any(np.linalg.norm(bearing - s.max_stress_at, axis=1).min() < 1.5 * h for s in parts):
            notes.append(
                tr(
                    "최대 응력이 볼트 머리·너트가 누르는 자리에 있습니다. 체결력이 좁은 면적에 몰리는 "
                    "국부 응력이므로, 와셔를 쓰거나 체결력을 낮추거나 볼트 자리 두께를 늘리는 것을 검토하세요."
                )
            )
    if not getattr(result, "contact_converged", True):
        notes.insert(
            0,
            tr(
                "접촉·마찰 상태 계산이 정해진 반복 횟수 안에 수렴하지 않았습니다. 접촉면과 볼트의 "
                "결과(특히 볼트 전단력, 마찰로 전달되는 힘)는 믿기 어렵습니다. 마찰 계수를 0으로 "
                "두고 비교하거나 메시 품질을 바꿔 결과가 비슷한지 확인하세요."
            ),
        )
    for i, (closed, total, slipping) in enumerate(result.contact or []):
        if closed == 0:
            notes.append(
                tr("접촉 {n}: 맞닿은 면이 모두 떨어졌습니다(힘을 전달하지 않음).", n=i + 1)
            )
        elif closed < total:
            notes.append(
                tr(
                    "접촉 {n}: 맞닿은 면의 약 {share:.0f}%만 눌려 있고 나머지는 살짝 벌어졌습니다.",
                    n=i + 1,
                    share=closed / total * 100,
                )
            )
        if slipping:
            notes.append(
                tr(
                    "접촉 {n}: 눌려 있는 면의 약 {share:.0f}%(누르는 힘 기준)가 마찰 한계를 넘어 "
                    "미끄러집니다. 미끄러지면 안 되는 자리라면 체결력을 높이거나 핀·키를 검토하세요.",
                    n=i + 1,
                    share=slipping / closed * 100,
                )
            )
    size = float(np.ptp(mesh.nodes, axis=0).max())
    max_node = int(np.argmax(disp))
    if skipped_fatigue:
        notes.append(
            tr(
                "반복 하중(피로) 판정은 선형 해석에서만 합니다. 항복·큰 변형 계산을 끄고 다시 "
                "해석하면 피로 안전율을 볼 수 있습니다."
            )
        )
    if disp[max_node] > 0.05 * size and nonlinear is not None:
        if not nonlinear.large_deformation:
            notes.append(
                tr(
                    "변형이 부품 크기의 5%를 넘습니다. 6단계에서 '큰 변형 고려'를 켜고 다시 "
                    "계산하면 더 정확합니다."
                )
            )
    elif disp[max_node] > 0.05 * size:
        notes.append(
            tr(
                "변형이 부품 크기의 5%를 넘습니다. 이 프로그램의 선형 해석은 작은 변형을 가정하므로 "
                "실제 거동과 차이가 클 수 있습니다."
            )
        )

    title = VERDICT_STYLE[worst_verdict][0]
    label = tr("최소 안전율")
    if fatigue is not None and worst_fos == min(s.fatigue_fos for s in parts):
        label = tr("피로 안전율")
    elif any(b.safety_factor == worst_fos for b in result.bolts or []):
        label = tr("볼트 안전율")
    elif worst.hotspot_fos is not None and worst_fos == worst.hotspot_fos:
        label = tr("최소 안전율 {fos:.2f}, 특이점 제외", fos=worst.min_fos)
    summary = Summary(
        parts=parts,
        verdict=worst_verdict,
        headline=f"{title} — {label} {worst_fos:.2f}",
        explanation=explanation,
        notes=notes,
        max_disp=float(disp[max_node]),
        max_disp_at=mesh.nodes[max_node],
        reactions=result.reactions,
        fatigue=fatigue,
        bolts=list(result.bolts or []),
        governing_fos=float(worst_fos),
    )
    if nonlinear is not None:
        _judge_nonlinear(summary, project, result, singular, us)
    return summary


# beyond this equivalent plastic strain a part is permanently bent for practical purposes;
# up to it the yielding is local and the permanent set small (the 0.2 % of a proof stress)
LOCAL_YIELD_STRAIN = 0.002


def _judge_nonlinear(summary: Summary, project: Project, result, singular, us: str) -> None:
    """Verdict of a run with yielding / large deformation: did the part carry the whole
    load, did it yield, how much deformation remains after unloading."""
    nl = result.nonlinear
    mesh = result.mesh
    worst_part, worst_strain, raw_strain = None, 0.0, 0.0
    for part in summary.parts:
        nodes = mesh.part_nodes(part.part_id)
        peeq = nl.plastic_strain[nodes]
        part_raw = float(peeq.max()) if len(peeq) else 0.0
        governing = part_raw
        if part.ductile and part.singular_kind is not None:
            # a sharp corner concentrates the plastic strain without bound as well
            zone = singular.zone(mesh, part.part_id)
            outside = nodes[~zone[nodes]]
            if len(outside):
                governing = float(nl.plastic_strain[outside].max())
        if nl.plasticity:
            part.plastic_strain = governing
            if part.ductile and governing > 0:
                # judged on how far it yielded, not on the (sub-1) elastic safety factor
                part.verdict = Verdict.WARN if governing <= LOCAL_YIELD_STRAIN else Verdict.FAIL
        if not nl.completed:
            part.verdict = Verdict.FAIL
        if governing >= worst_strain:
            worst_part, worst_strain, raw_strain = part, governing, part_raw
    applied = np.linalg.norm(result.applied_load)
    permanent = (
        float(np.linalg.norm(nl.permanent, axis=1).max()) if nl.permanent is not None else 0.0
    )
    perm_text = (
        tr(
            " 하중을 빼도 최대 {value}의 변형이 남습니다.",
            value=fmt(us, "length", permanent, ".3g"),
        )
        if permanent > 0
        else ""
    )
    kind = []
    if nl.plasticity:
        kind.append(tr("항복"))
    if nl.large_deformation:
        kind.append(tr("큰 변형"))
    summary.notes.insert(
        0,
        tr(
            "비선형 해석({kinds} 고려): 하중을 {n}단계로 나눠 걸며 계산했습니다",
            kinds="·".join(kind),
            n=nl.increments,
        )
        + (
            tr(
                ". 재료는 항복강도까지 탄성, 인장강도까지 직선으로 단단해지고 그 뒤로는 거의 "
                "늘어나기만 한다고 가정했습니다."
            )
            if nl.plasticity
            else "."
        )
        + (tr(" 하중 방향은 변형 뒤에도 처음 그대로입니다.") if nl.large_deformation else ""),
    )
    if worst_strain > 0 and raw_strain > worst_strain * 1.5:
        summary.notes.append(
            tr(
                "날카로운 모서리 바로 옆의 소성 변형률({strain:.2g}%)은 메시에 따라 "
                "계속 커지므로 판정은 그 주변을 제외한 값으로 했습니다.",
                strain=raw_strain * 100,
            )
        )
    if not nl.completed:
        reached = nl.load_factor * 100
        carried = fmt(us, "force", applied, ",.0f") if applied > 0 else ""
        summary.verdict = Verdict.FAIL
        if nl.plasticity:
            summary.headline = tr("❌ 위험 — 하중의 약 {p:.0f}%에서 무너짐", p=reached)
            summary.explanation = (
                tr(
                    "부품이 하중을 끝까지 버티지 못합니다. 약 {p:.0f}%({carried})에서 재료가 "
                    "넓게 항복해 변형이 걷잡을 수 없이 커집니다(소성 붕괴). "
                    "단면을 키우거나 더 강한 재료를 쓰세요.",
                    p=reached,
                    carried=carried,
                )
                if carried
                else tr(
                    "부품이 하중을 끝까지 버티지 못합니다. 약 {p:.0f}%에서 재료가 "
                    "넓게 항복해 변형이 걷잡을 수 없이 커집니다(소성 붕괴). "
                    "단면을 키우거나 더 강한 재료를 쓰세요.",
                    p=reached,
                )
            )
        else:
            summary.headline = tr("❌ 위험 — 하중의 약 {p:.0f}%에서 불안정", p=reached)
            summary.explanation = tr(
                "약 {p:.0f}%의 하중에서 형상이 갑자기 크게 바뀌어(좌굴·뒤집힘) 더 "
                "버티지 못합니다. 좌굴 해석으로 확인하고, 두께를 늘리거나 받침을 더하세요.",
                p=reached,
            )
        summary.governing_fos = nl.load_factor
        return
    if worst_part is None or worst_strain <= 0:
        if nl.plasticity:
            summary.notes.append(tr("어느 곳도 항복하지 않았습니다(재료가 탄성 범위 안에 있음)."))
        return
    name = tr_data(worst_part.name)
    # ductile parts are judged on their plastic strain here, not on the linear safety
    # factor (which is below 1 wherever the material yielded); brittle parts keep theirs
    brittle = [p.verdict for p in summary.parts if not p.ductile]
    floor = max(brittle, key=lambda v: _RANK[v]) if brittle else Verdict.OK
    if worst_strain <= LOCAL_YIELD_STRAIN:
        summary.verdict = max(floor, Verdict.WARN, key=lambda v: _RANK[v])
        summary.headline = tr(
            "{title} — 국부 항복 (소성 변형률 {strain:.2g}%)",
            title=VERDICT_STYLE[summary.verdict][0],
            strain=worst_strain * 100,
        )
        summary.explanation = tr(
            "하중은 모두 견디지만 {name}의 일부가 항복합니다. 항복한 범위가 작아 영구 변형은 "
            "작습니다.{permanent} 반복해서 걸리는 하중이라면 항복하지 않도록 보강하세요.",
            name=name,
            permanent=perm_text,
        )
    else:
        summary.verdict = Verdict.FAIL
        summary.headline = tr(
            "❌ 위험 — 영구 변형 (소성 변형률 {strain:.2g}%)", strain=worst_strain * 100
        )
        summary.explanation = tr(
            "하중은 견디지만 {name}이(가) 크게 항복해 영구적으로 휘어집니다.{permanent} "
            "두께를 늘리거나, 모서리를 둥글게 하거나, 더 강한 재료를 검토하세요.",
            name=name,
            permanent=perm_text,
        )


@dataclass
class CaseOverview:
    """Comparison of the load cases of one strength run (which one governs)."""

    names: list[str]
    summaries: list[Summary]
    worst: int  # index of the most unfavourable case

    @property
    def headline(self) -> str:
        return tr(
            "{n}개 케이스 중 가장 불리한 것: {name}",
            n=len(self.names),
            name=tr_data(self.names[self.worst]),
        )


def compare_cases(names: list[str], summaries: list[Summary]) -> CaseOverview:
    def fos(s: Summary) -> float:
        # the headline number: the governing safety factor of the case
        return min(
            [p.governing_fos for p in s.parts]
            + [p.fatigue_fos for p in s.parts if p.fatigue_fos is not None]
            + [b.safety_factor for b in s.bolts]
        )

    worst = max(
        range(len(summaries)), key=lambda i: (_RANK[summaries[i].verdict], -fos(summaries[i]))
    )
    return CaseOverview(list(names), list(summaries), worst)


def _contact_note(project: Project) -> list[str]:
    """Modal, thermal and buckling solves are linear: contact pairs are treated as bonded."""
    n = sum(1 for c in project.connections if c.kind == ConnectionKind.CONTACT)
    if n == 0:
        return []
    return [
        tr(
            "접촉으로 지정한 연결 {n}개는 이 해석에서 접합(붙어 있음)으로 보고 계산했습니다. "
            "접촉면이 실제로 벌어지거나 미끄러지면 결과가 달라질 수 있습니다.",
            n=n,
        )
    ]


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
    notes = _contact_note(project)
    if result.rigid_modes:
        notes.append(
            tr(
                "고정이 없어 부품이 공중에 떠 있는 상태로 계산했습니다(자유 진동). "
                "0 Hz에 해당하는 움직임 6개는 제외했습니다."
            )
        )
    if f_op <= 0:
        return ModalSummary(
            None,
            tr("가장 낮은 고유진동수 {f:,.1f} Hz", f=freqs[0]),
            tr(
                "부품이 스스로 잘 흔들리는 진동수입니다. 모터·펌프 등 주변 진동원의 진동수가 "
                "이 값들과 가까우면 공진으로 크게 흔들릴 수 있습니다. "
                "6단계에서 '운전 진동수'를 입력하면 공진 위험을 판정해 드립니다."
            ),
            notes,
        )
    gaps = np.abs(freqs - f_op) / f_op
    closest = int(np.argmin(gaps))
    gap = float(gaps[closest])
    if gap <= RESONANCE_FAIL:
        verdict = Verdict.FAIL
        text = tr(
            "{mode}차 고유진동수 {f:,.1f} Hz가 운전 진동수 {op:,.1f} Hz와 "
            "{gap:.0f}% 차이로 매우 가깝습니다. 공진으로 크게 흔들리거나 피로 파손될 수 "
            "있습니다. 두께·리브로 강성을 바꾸거나 운전 속도를 조정하세요.",
            mode=closest + 1,
            f=freqs[closest],
            op=f_op,
            gap=gap * 100,
        )
    elif gap <= RESONANCE_WARN:
        verdict = Verdict.WARN
        text = tr(
            "{mode}차 고유진동수 {f:,.1f} Hz가 운전 진동수 {op:,.1f} Hz와 "
            "{gap:.0f}% 차이입니다. 공진 여유(보통 ±20% 이상)가 부족합니다.",
            mode=closest + 1,
            f=freqs[closest],
            op=f_op,
            gap=gap * 100,
        )
    else:
        verdict = Verdict.OK
        text = tr(
            "모든 고유진동수가 운전 진동수 {op:,.1f} Hz와 20% 이상 떨어져 있어 "
            "공진 위험이 낮습니다(가장 가까운 것: {mode}차 {f:,.1f} Hz).",
            op=f_op,
            mode=closest + 1,
            f=freqs[closest],
        )
    if f_op > freqs[-1] * (1 + RESONANCE_WARN):
        notes.append(
            tr(
                "운전 진동수가 계산한 {n}개 진동수보다 높습니다. 더 높은 차수의 진동수는 "
                "확인하지 않았습니다.",
                n=len(freqs),
            )
        )
    title = VERDICT_STYLE[verdict][0]
    headline = tr(
        "{title} — 1차 고유진동수 {f:,.1f} Hz (운전 {op:,.1f} Hz)",
        title=title,
        f=freqs[0],
        op=f_op,
    )
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
    # over time the verdict follows the hottest moment, not just the end
    hottest = result.history.max(axis=0) if result.transient else T
    worst, worst_ratio, limit_text = None, -np.inf, ""
    for pid, m in materials.items():
        if m.max_service_temperature is None:
            continue
        t_max = float(hottest[mesh.part_nodes(pid)].max())
        ratio = t_max / m.max_service_temperature
        if ratio > worst_ratio:
            worst, worst_ratio = (pid, m, t_max), ratio
    notes = _contact_note(project)
    if result.heat_in > 0 and not result.transient:  # over time, heat is also stored
        balance = abs(result.heat_out - result.heat_in) / max(result.heat_in, result.heat_out)
        if balance > 0.02:
            notes.append(
                tr(
                    "들어온 열 {heat_in}와 나간 열 {heat_out}의 차이가 "
                    "{balance:.0f}%입니다. 메시를 더 세밀하게 하면 정확해집니다.",
                    heat_in=fmt("SI", "power", result.heat_in),
                    heat_out=fmt("SI", "power", result.heat_out),
                    balance=balance * 100,
                )
            )
    us = system_of(project)

    def deg(value: float, digits: str = ".0f") -> str:
        return fmt(us, "temperature", value, digits)

    headline = tr("최고 {high} · 최저 {low}", high=deg(T.max(), ",.1f"), low=deg(T.min(), ",.1f"))
    if result.transient:
        end = format_duration(float(result.times[-1]))
        headline = tr("{end} 후 {headline}", end=end, headline=headline)
        peak, when = result.peak()
        if when < result.times[-1] and peak > T.max() + 0.5:
            notes.append(
                tr(
                    "가장 뜨거웠던 때는 {when} 무렵으로, 최고 "
                    "{peak}였습니다. 판정은 이 최고 온도로 했습니다.",
                    when=format_duration(when),
                    peak=deg(peak, ",.1f"),
                )
            )
        notes.append(
            tr(
                "시작 온도 {start}에서 "
                "{end} 동안의 온도 변화를 계산했습니다. '시각'에서 중간 시점을 볼 수 있습니다.",
                start=deg(project.initial_temperature, ",.1f"),
                end=end,
            )
        )
    if worst is None:
        return ThermalSummary(None, headline, tr("온도 분포를 계산했습니다."), notes)
    pid, m, t_max = worst
    limit = m.max_service_temperature
    name = tr_data(project.part(pid).name)
    if t_max <= 0.9 * limit:
        verdict = Verdict.OK
        limit_text = tr(
            "가장 뜨거운 곳도 {name} 재료의 권장 최고 사용온도({limit})보다 충분히 낮습니다.",
            name=name,
            limit=deg(limit),
        )
    elif t_max <= limit:
        verdict = Verdict.WARN
        limit_text = tr(
            "{name}의 최고 온도 {t_max}가 재료의 권장 최고 사용온도({limit})에 "
            "가깝습니다. 냉각을 늘리거나 내열성이 높은 재료를 검토하세요.",
            name=name,
            t_max=deg(t_max),
            limit=deg(limit),
        )
    else:
        verdict = Verdict.FAIL
        limit_text = tr(
            "{name}의 최고 온도 {t_max}가 재료의 권장 최고 사용온도({limit})를 "
            "넘습니다. 강도가 떨어지거나 변형·변색될 수 있습니다.",
            name=name,
            t_max=deg(t_max),
            limit=deg(limit),
        )
    title = VERDICT_STYLE[verdict][0]
    return ThermalSummary(verdict, f"{title} — {headline}", limit_text, notes)


def format_duration(seconds: float) -> str:
    """'45초', '12분 30초', '2시간 5분' — for times on screen and in reports."""
    seconds = round(seconds)
    if seconds < 60:
        return tr("{s}초", s=seconds)
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return tr("{m}분 {s}초", m=m, s=s) if s else tr("{m}분", m=m)
    h, rest = divmod(seconds, 3600)
    m = rest // 60
    return tr("{h}시간 {m}분", h=h, m=m) if m else tr("{h}시간", h=h)


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
        text = tr(
            "지금 하중의 {lam:,.1f}배가 되어야 좌굴(갑자기 옆으로 휘며 무너짐)이 일어납니다. "
            "권장 여유 {target:.1f}배 이상을 만족합니다.",
            lam=lam,
            target=target,
        )
    elif v == Verdict.WARN:
        text = tr(
            "지금 하중의 {lam:,.2f}배에서 좌굴이 일어납니다. 견디기는 하지만 권장 여유 "
            "{target:.1f}배에 못 미칩니다. 두께를 늘리거나 리브·지지를 추가하세요.",
            lam=lam,
            target=target,
        )
    else:
        text = tr(
            "지금 하중의 {lam:,.2f}배, 즉 이 하중에 도달하기 전에 좌굴로 무너집니다. "
            "단면을 키우거나 길이를 줄이거나 중간 지지를 추가해야 합니다.",
            lam=lam,
        )
    notes = _contact_note(project) + [
        tr(
            "선형 좌굴 해석은 완벽한 형상을 가정하므로 실제 좌굴 하중은 이보다 낮을 수 있습니다"
            "(제작 오차·편심). 그래서 일반 강도보다 큰 여유(보통 3배 이상)를 권장합니다."
        )
    ]
    static_fos = float(result.static.safety_factor.min())
    if static_fos < lam:
        notes.append(
            tr(
                "참고: 이 하중에서 강도 안전율은 {fos:.2f}입니다. 좌굴보다 재료 항복이 먼저 "
                "일어날 수 있으니 '강도' 해석도 확인하세요.",
                fos=static_fos,
            )
        )
    title = VERDICT_STYLE[v][0]
    return BucklingSummary(
        v, tr("{title} — 좌굴 하중 배수 {lam:,.2f}", title=title, lam=lam), text, notes
    )
