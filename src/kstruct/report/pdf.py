"""PDF report: verdict, inputs, result pictures and the disclaimer (Qt text engine,
no external dependencies)."""

from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMarginsF, QSizeF
from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

from kstruct import __version__, standards
from kstruct.fem.mesh import MESH_PRESETS
from kstruct.fem.static import StaticResult
from kstruct.i18n import N_, tr, tr_data
from kstruct.model import THERMAL_LOADS, LoadKind, Project, active_project
from kstruct.report.describe import describe_connection, describe_fixture, describe_load
from kstruct.report.summary import VERDICT_STYLE, Summary
from kstruct.units import fmt, system_of

MESH_LABELS = {
    "coarse": N_("거침"),
    "normal": N_("보통"),
    "fine": N_("세밀"),
    "very_fine": N_("매우 세밀"),
}
FONT = "font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;"


def disclaimer(kind: str = N_("선형 정적 유한요소해석")) -> str:
    return tr(
        "이 보고서는 {kind} 결과로, 설계 검토를 돕기 위한 참고 자료입니다. "
        "실제 제작·사용 여부는 하중 조건, 재료 성적서, 제작 품질, 관련 법규를 고려하여 "
        "자격을 갖춘 기술자가 최종 판단해야 합니다.",
        kind=tr(kind),
    )


DISCLAIMER = disclaimer()


def _img(png: bytes, width: int = 640) -> str:
    data = base64.b64encode(png).decode()
    return f'<img src="data:image/png;base64,{data}" width="{width}">'


def _header(project: Project, mesh) -> str:
    """Model, date, software version, standards pack, units and mesh: what a reviewer
    needs to reproduce the run."""
    esc = html.escape
    pack = standards.effective_pack(project)
    source = esc(project.source_path.name) if project.source_path else "-"
    units = "SI (mm·N·MPa·°C)" if system_of(project) == "SI" else tr("미국 단위 (in·lbf·psi·°F)")
    quality = (
        tr(MESH_LABELS[project.mesh_quality])
        if project.mesh_quality in MESH_LABELS
        else project.mesh_quality
    )
    preset = MESH_PRESETS.get(project.mesh_quality, MESH_PRESETS["normal"])
    custom = tr(" (사용자 지정 안전율)") if project.target_overrides else ""
    return tr(
        "<p>모델: {source} &nbsp;|&nbsp; 작성일: {date} &nbsp;|&nbsp; KStruct {version}<br>"
        "적용 기준: {pack}{custom} &nbsp;|&nbsp; 단위: {units} &nbsp;|&nbsp; "
        "메시: {quality} (모델 대각선/{divisions:g}, 두께 방향 {layers}개 이상), "
        "2차 사면체 요소 {tets:,}개, 절점 {nodes:,}개</p>",
        source=source,
        date=f"{dt.date.today():%Y-%m-%d}",
        version=esc(__version__),
        pack=esc(tr_data(pack.label)),
        custom=custom,
        units=units,
        quality=quality,
        divisions=preset.divisions,
        layers=preset.layers,
        tets=len(mesh.tets),
        nodes=mesh.n_nodes,
    )


def _list(items) -> str:
    return "".join(f"<li>{html.escape(x)}</li>" for x in items)


def _connections_html(project: Project) -> str:
    if not project.connections:
        return ""
    items = _list(describe_connection(project, c) for c in project.connections)
    return f"<p><b>{tr('부품 연결')}</b></p><ul>{items}</ul>"


def _materials_list(project: Project, materials: dict) -> str:
    pack = standards.effective_pack(project)
    return _list(
        f"{tr_data(project.part(pid).name)}: {tr_data(pack.material_name(m))}"
        for pid, m in materials.items()
    )


def _pictures(images) -> str:
    return "".join(f"<h3>{html.escape(cap)}</h3><p>{_img(png)}</p>" for cap, png in images)


def _page(
    title: str,
    project: Project,
    headline: str,
    explanation: str,
    verdict,
    body: str,
    mesh,
    method: str,
) -> str:
    esc = html.escape
    if verdict is None:
        color, bg = "#2b5d8a", "#eaf2fa"
    else:
        _, color, bg = VERDICT_STYLE[verdict]
    return f"""
<html><body style="{FONT} font-size: 10pt;">
<h1>{esc(title)}</h1>
{_header(project, mesh)}
<table width="100%" cellpadding="10" style="background:{bg}; border: 2px solid {color};">
<tr><td><span style="font-size:16pt; color:{color};"><b>{esc(headline)}</b></span><br>
{esc(explanation)}</td></tr></table>
{body}
<p style="color:#8a6d3b;"><b>{tr("고지")}</b>: {disclaimer(method)}</p>
</body></html>
"""


# --- static ---------------------------------------------------------------------------


def _cases_html(project: Project, overview, us: str) -> str:
    if overview is None or len(overview.names) < 2:
        return ""
    rows = []
    for i, (name, s) in enumerate(zip(overview.names, overview.summaries, strict=True)):
        mark = tr(" ◀ 가장 불리") if i == overview.worst else ""
        rows.append(
            f"<tr><td>{html.escape(tr_data(name))}{mark}</td>"
            f"<td><b>{s.governing_fos:.2f}</b></td>"
            f"<td>{fmt(us, 'length', s.max_disp)}</td>"
            f"<td>{VERDICT_STYLE[s.verdict][0]}</td></tr>"
        )
    current = html.escape(tr_data(project.case.name))
    return (
        f"<h2>{tr('하중 케이스 비교')}</h2>"
        "<table width='100%' border='1' cellspacing='0' cellpadding='4'>"
        f"<tr style='background:#eef1f5;'><th>{tr('케이스')}</th><th>{tr('안전율')}</th>"
        f"<th>{tr('최대 변형')}</th><th>{tr('판정')}</th></tr>{''.join(rows)}</table>"
        + tr("<p>아래 상세 결과는 '{case}' 케이스의 것입니다.</p>", case=current)
    )


def _static_html(
    project: Project,
    result: StaticResult,
    summary: Summary,
    images: list[tuple[str, bytes]],
    overview=None,
) -> str:
    pack = standards.effective_pack(project)
    us = system_of(project)
    esc = html.escape
    rows = []
    for s in summary.parts:
        crit = tr("폰 미세스 응력") if s.ductile else tr("최대 주응력")
        cap = tr("항복강도") if s.ductile else tr("인장강도")
        rows.append(
            f"<tr><td>{esc(tr_data(s.name))}</td><td>{esc(tr_data(s.material))}</td>"
            f"<td>{fmt(us, 'stress', s.max_stress, ',.1f')} ({crit})</td>"
            f"<td>{fmt(us, 'stress', s.capacity, ',.0f')} ({cap})</td>"
            f"<td><b>{s.min_fos:.2f}</b> / {s.target:.1f}</td><td>{fmt(us, 'length', s.max_disp)}</td>"
            f"<td>{VERDICT_STYLE[s.verdict][0]}</td></tr>"
        )
        if s.plastic_strain is not None:
            rows.append(
                f"<tr><td colspan='2' align='right'>{tr('소성 변형률(항복 정도)')}</td>"
                f"<td colspan='5'>{s.plastic_strain * 100:.2g}%</td></tr>"
            )
        if s.hotspot_fos is not None:
            rows.append(
                "<tr><td colspan='2' align='right'>"
                + tr("특이점({kind}) 제외", kind=esc(tr_data(s.singular_kind)))
                + "</td>"
                f"<td>{fmt(us, 'stress', s.hotspot_stress, ',.1f')}</td><td></td>"
                f"<td><b>{s.hotspot_fos:.2f}</b> / {s.target:.1f}</td><td></td>"
                f"<td>{VERDICT_STYLE[s.hotspot_verdict][0]}</td></tr>"
            )
        if s.fatigue_fos is not None:
            life = tr("무한 수명") if s.fatigue_life >= 1e9 else tr("{n:,.0f}회", n=s.fatigue_life)
            rows.append(
                f"<tr><td colspan='4' align='right'>{tr('반복 하중 피로 검토')}</td>"
                f"<td><b>{s.fatigue_fos:.2f}</b> / {s.target:.1f}</td><td>{life}</td>"
                f"<td>{VERDICT_STYLE[s.fatigue_verdict][0]}</td></tr>"
            )

    fixtures = _list(describe_fixture(project, f) for f in project.fixtures)
    load_items = _list(describe_load(project, ld) for ld in project.loads)
    reactions = _list(
        f"{describe_fixture(project, f)}: "
        f"X {fmt(us, 'force', r[0], '+,.1f')}, Y {fmt(us, 'force', r[1], '+,.1f')}, "
        f"Z {fmt(us, 'force', r[2], '+,.1f')}"
        for f, r in zip(project.fixtures, summary.reactions, strict=True)
    )
    bolt_rows = "".join(
        f"<tr><td>{esc(tr_data(b.name))}</td><td>{b.size} {b.grade}</td>"
        f"<td>{fmt(us, 'force', b.preload, ',.0f')}</td>"
        f"<td>{fmt(us, 'force', b.axial, ',.0f')}</td><td>{fmt(us, 'force', b.shear, ',.0f')}</td>"
        f"<td>{fmt(us, 'stress', b.stress, ',.0f')} / {fmt(us, 'stress', b.proof, ',.0f')}</td>"
        f"<td><b>{b.safety_factor:.2f}</b></td></tr>"
        for b in result.bolts or []
    )
    bolts = (
        f"<h3>{tr('볼트')}</h3>"
        "<table width='100%' border='1' cellspacing='0' cellpadding='4'>"
        f"<tr style='background:#eef1f5;'><th>{tr('볼트')}</th><th>{tr('규격')}</th>"
        f"<th>{tr('체결력')}</th><th>{tr('축력')}</th><th>{tr('전단력')}</th>"
        f"<th>{tr('응력 / 보증응력')}</th><th>{tr('안전율')}</th></tr>"
        f"{bolt_rows}</table>"
        if bolt_rows
        else ""
    )
    thermal = ""
    if result.thermal is not None:
        t = result.thermal.temperature
        thermal = tr(
            "<p><b>열응력 포함</b>: 온도 {low} ~ {high}, 응력 없는 기준 온도 {ref}</p>",
            low=fmt(us, "temperature", t.min(), ".1f"),
            high=fmt(us, "temperature", t.max(), ".1f"),
            ref=fmt(us, "temperature", project.reference_temperature, ".0f"),
        )
    notes = _list(summary.notes)
    at = [fmt(us, "length", float(v), ".1f") for v in summary.max_disp_at]
    nl = result.nonlinear
    if nl is None:
        method = tr("선형 탄성 정적해석 (작은 변형, 재료는 항복 전까지 선형으로 가정)")
    else:
        parts = []
        if nl.plasticity:
            parts.append(
                tr(
                    "재료 항복 고려(폰 미세스 소성, 항복강도 → 인장강도까지 직선 경화, "
                    "그 뒤 거의 일정)"
                )
            )
        if nl.large_deformation:
            parts.append(tr("큰 변형 고려(전체 라그랑주, 하중 방향은 처음 그대로)"))
        method = tr(
            "비선형 정적해석: {parts}. 하중 {increments}단계, 뉴턴 반복 {iterations}회, "
            "도달 하중 {reached:.0f}%",
            parts=", ".join(parts),
            increments=nl.increments,
            iterations=nl.iterations,
            reached=nl.load_factor * 100,
        )
    body = f"""
{f"<p><b>{tr('참고')}</b></p><ul>{notes}</ul>" if notes else ""}
{_cases_html(project, overview, us)}
<h2>{tr("1. 부품별 결과")}</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>{tr("부품")}</th><th>{tr("재료")}</th><th>{tr("최대 응력")}</th><th>{tr("재료 강도")}</th>
<th>{tr("안전율 / 권장")}</th><th>{tr("최대 변형")}</th><th>{tr("판정")}</th></tr>
{"".join(rows)}
</table>
{bolts}
{thermal}
{tr("<p>최대 변형량: {disp} (위치 X {x}, Y {y}, Z {z})</p>", disp=fmt(us, "length", summary.max_disp), x=at[0], y=at[1], z=at[2])}

<h2>{tr("2. 해석 조건")}</h2>
<p><b>{tr("고정")}</b></p><ul>{fixtures}</ul>
<p><b>{tr("하중")}</b>{f" ({esc(tr_data(project.case.name))})" if len(project.cases) > 1 else ""}</p><ul>{load_items}</ul>
{_connections_html(project)}
<p>{tr("<b>반력</b> (고정면이 부품을 받치는 힘)")}</p><ul>{reactions}</ul>

<h2>{tr("3. 결과 그림")}</h2>
{_pictures(images)}

<h2>{tr("4. 해석 방법과 한계")}</h2>
<ul>
<li>{method}</li>
<li>{tr("연성재는 폰 미세스 응력과 항복강도로, 취성재는 최대 주응력과 인장강도로 안전율을 계산")}</li>
<li>{tr("권장 안전율은 하중을 잘 아는 정적 조건의 일반적 기계설계 권장값이며, 피로·충격·고온 조건은 별도 검토 필요")}</li>
<li>{tr("참고 기준: {refs}", refs=esc(", ".join(tr_data(x) for x in pack.references)))}</li>
</ul>
"""
    return _page(
        tr("구조 해석 보고서"),
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        N_("비선형 정적 유한요소해석")
        if result.nonlinear is not None
        else N_("선형 정적 유한요소해석"),
    )


# --- modal / thermal / buckling -------------------------------------------------------


def _modal_html(project, result, summary, images, materials) -> str:
    rows = "".join(
        f"<tr><td>{tr('{n}차', n=i + 1)}</td><td><b>{f:,.2f} Hz</b></td>"
        f"<td>{f * 60:,.0f} rpm</td>"
        f"<td>{p[0] * 100:.0f}% / {p[1] * 100:.0f}% / {p[2] * 100:.0f}%</td></tr>"
        for i, (f, p) in enumerate(zip(result.frequencies, result.participation, strict=True))
    )
    op = (
        tr(
            "<p>운전 진동수: {hz:,.2f} Hz ({rpm:,.0f} rpm)</p>",
            hz=project.operating_frequency,
            rpm=project.operating_frequency * 60,
        )
        if project.operating_frequency > 0
        else tr("<p>운전 진동수: 입력하지 않음</p>")
    )
    fixtures = _list(describe_fixture(project, f) for f in project.fixtures) or tr(
        "<li>없음 (자유 진동)</li>"
    )
    notes = _list(summary.notes)
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>{tr("1. 고유진동수")}</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>{tr("차수")}</th><th>{tr("진동수")}</th><th>{tr("회전수 환산")}</th>
<th>{tr("유효질량 X / Y / Z")}</th></tr>{rows}</table>
{op}
<p>{tr("유효질량은 그 진동 모양이 각 방향 흔들림에 얼마나 크게 관여하는지를 나타냅니다.")}</p>
<h2>{tr("2. 해석 조건")}</h2>
<p><b>{tr("재료")}</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>{tr("고정")}</b></p><ul>{fixtures}</ul>
{_connections_html(project)}
<h2>{tr("3. 진동 모양")}</h2>
<p>{tr("색과 변형 크기는 상대적인 값입니다(실제 진폭은 가진력의 크기에 따라 달라집니다).")}</p>
{_pictures(images)}
<h2>{tr("4. 해석 방법과 한계")}</h2>
<ul><li>{tr("선형 고유치 해석 (감쇠 무시, 작은 진동 가정)")}</li>
<li>{tr("공진 판정: 운전 진동수 ±10% 이내 위험, ±20% 이내 주의")}</li></ul>
"""
    return _page(
        tr("진동(고유진동수) 해석 보고서"),
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        N_("선형 고유진동수(모달) 유한요소해석"),
    )


def _thermal_html(project, result, summary, images, materials) -> str:
    us = system_of(project)
    loads = _list(describe_load(project, ld) for ld in project.loads if ld.kind in THERMAL_LOADS)
    rows = []
    for pid, m in materials.items():
        nodes = result.mesh.part_nodes(pid)
        limit = (
            fmt(us, "temperature", m.max_service_temperature, ".0f")
            if m.max_service_temperature
            else "-"
        )
        rows.append(
            f"<tr><td>{html.escape(tr_data(project.part(pid).name))}</td>"
            f"<td>{fmt(us, 'temperature', result.temperature[nodes].min(), '.1f')}</td>"
            f"<td><b>{fmt(us, 'temperature', result.temperature[nodes].max(), '.1f')}</b></td>"
            f"<td>{limit}</td></tr>"
        )
    notes = _list(summary.notes)
    if result.transient:
        from kstruct.report.summary import format_duration

        kind_note = tr(
            "시간에 따른 열전도 해석: 처음 온도 {start}에서 {duration} 동안 (후진 오일러 "
            "{steps}단계). 표의 온도는 마지막 시점, 판정은 그동안의 최고 온도 기준",
            start=fmt(us, "temperature", project.initial_temperature, ".1f"),
            duration=format_duration(float(result.times[-1])),
            steps=result.iterations,
        )
    else:
        kind_note = tr("정상상태 열전도 해석 (충분한 시간이 지나 온도가 더 이상 변하지 않는 상태)")
    has_radiation = any(ld.kind == LoadKind.RADIATION for ld in project.loads)
    has_table = any(m.conductivity_table for m in materials.values())
    method_note = "; ".join(
        [
            tr("복사는 '복사 냉각'을 지정한 면에서만 고려(반복 계산)")
            if has_radiation
            else tr("복사 열전달은 지정하지 않아 고려하지 않음"),
            tr("온도에 따른 열전도율 변화를 반영(재료의 열전도율 표)")
            if has_table
            else tr("열전도율은 온도와 무관한 상수로 가정"),
        ]
    )
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>{tr("1. 부품별 온도")}</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>{tr("부품")}</th><th>{tr("최저")}</th><th>{tr("최고")}</th><th>{tr("권장 최고 사용온도")}</th></tr>
{"".join(rows)}</table>
{tr("<p>들어온 열 {heat_in} · 빠져나간 열 {heat_out}</p>", heat_in=fmt(us, "power", result.heat_in), heat_out=fmt(us, "power", result.heat_out))}
<h2>{tr("2. 해석 조건")}</h2>
<p><b>{tr("재료")}</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>{tr("열 조건")}</b></p><ul>{loads}</ul>
{_connections_html(project)}
<p>{tr("지정하지 않은 면은 열이 드나들지 않는(단열) 것으로 가정합니다.")}</p>
<h2>{tr("3. 결과 그림")}</h2>
{_pictures(images)}
<h2>{tr("4. 해석 방법과 한계")}</h2>
<ul><li>{kind_note}</li>
<li>{method_note}</li>
<li>{tr("권장 최고 사용온도는 대략적인 값이며 재료 등급에 따라 다름")}</li></ul>
"""
    return _page(
        tr("열전달 해석 보고서"),
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        N_("정상상태 열전도 유한요소해석"),
    )


def _buckling_html(project, result, summary, images, materials) -> str:
    us = system_of(project)
    applied = float(np.linalg.norm(result.static.applied_load))
    rows = "".join(
        f"<tr><td>{tr('{n}차', n=i + 1)}</td><td><b>{f:,.2f}</b></td>"
        f"<td>{fmt(us, 'force', f * applied, ',.0f')}</td></tr>"
        for i, f in enumerate(result.factors)
    )
    fixtures = _list(describe_fixture(project, f) for f in project.fixtures)
    loads = _list(describe_load(project, ld) for ld in project.loads)
    notes = _list(summary.notes)
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>{tr("1. 좌굴 하중")}</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>{tr("차수")}</th><th>{tr("하중 배수")}</th><th>{tr("좌굴 시 하중 합계(추정)")}</th></tr>
{rows}</table>
<p>{tr("하중 배수 = 좌굴이 일어나는 하중 ÷ 현재 하중. 현재 하중 합계 {total}.", total=fmt(us, "force", applied, ",.1f"))}</p>
<h2>{tr("2. 해석 조건")}</h2>
<p><b>{tr("재료")}</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>{tr("고정")}</b></p><ul>{fixtures}</ul>
<p><b>{tr("하중")}</b></p><ul>{loads}</ul>
{_connections_html(project)}
<h2>{tr("3. 좌굴 모양")}</h2>
<p>{tr("변형 크기는 상대값입니다.")}</p>
{_pictures(images)}
<h2>{tr("4. 해석 방법과 한계")}</h2>
<ul><li>{tr("선형 고유치 좌굴 해석 (초기 결함·소성·대변형 미고려)")}</li>
<li>{tr("권장 하중 배수 {target:.1f} 이상", target=standards.effective_pack(project).target_buckling)}</li></ul>
"""
    return _page(
        tr("좌굴 해석 보고서"),
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        N_("선형 고유치 좌굴 유한요소해석"),
    )


def build_html(
    project, result, summary, images, materials: dict | None = None, overview=None
) -> str:
    from kstruct.fem.buckling import BucklingResult
    from kstruct.fem.modal import ModalResult
    from kstruct.fem.thermal import ThermalResult

    # the result was computed on the visible parts and the conditions that refer only to
    # them: report exactly that (a hidden part's fixture used to crash the report, its
    # loads were listed although they were not applied)
    project = active_project(project)
    if isinstance(result, BucklingResult):
        return _buckling_html(project, result, summary, images, materials or {})
    if isinstance(result, ModalResult):
        return _modal_html(project, result, summary, images, materials or {})
    if isinstance(result, ThermalResult):
        return _thermal_html(project, result, summary, images, materials or {})
    return _static_html(project, result, summary, images, overview)


def write_pdf(html_text: str, path: str | Path) -> Path:
    path = Path(path)
    writer = QPdfWriter(str(path))
    writer.setPageLayout(
        QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(15, 15, 15, 15),
        )
    )
    writer.setResolution(150)
    doc = QTextDocument()
    doc.setHtml(html_text)
    doc.setPageSize(QSizeF(writer.width(), writer.height()))
    doc.print_(writer)
    return path
