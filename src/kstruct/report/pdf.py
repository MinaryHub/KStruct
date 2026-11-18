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
from kstruct.model import THERMAL_LOADS, LoadKind, Project, active_project
from kstruct.report.describe import describe_connection, describe_fixture, describe_load
from kstruct.report.summary import VERDICT_STYLE, Summary
from kstruct.units import fmt, system_of

MESH_LABELS = {"coarse": "거침", "normal": "보통", "fine": "세밀", "very_fine": "매우 세밀"}
FONT = "font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;"


def disclaimer(kind: str = "선형 정적 유한요소해석") -> str:
    return (
        f"이 보고서는 {kind} 결과로, 설계 검토를 돕기 위한 참고 자료입니다. "
        "실제 제작·사용 여부는 하중 조건, 재료 성적서, 제작 품질, 관련 법규를 고려하여 "
        "자격을 갖춘 기술자가 최종 판단해야 합니다."
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
    units = "SI (mm·N·MPa·°C)" if system_of(project) == "SI" else "미국 단위 (in·lbf·psi·°F)"
    quality = MESH_LABELS.get(project.mesh_quality, project.mesh_quality)
    preset = MESH_PRESETS.get(project.mesh_quality, MESH_PRESETS["normal"])
    custom = " (사용자 지정 안전율)" if project.target_overrides else ""
    return (
        f"<p>모델: {source} &nbsp;|&nbsp; 작성일: {dt.date.today():%Y-%m-%d} &nbsp;|&nbsp; "
        f"KStruct {esc(__version__)}<br>"
        f"적용 기준: {esc(pack.label)}{custom} &nbsp;|&nbsp; 단위: {units} &nbsp;|&nbsp; "
        f"메시: {quality} (모델 대각선/{preset.divisions:g}, 두께 방향 {preset.layers}개 이상), "
        f"2차 사면체 요소 {len(mesh.tets):,}개, 절점 {mesh.n_nodes:,}개</p>"
    )


def _list(items) -> str:
    return "".join(f"<li>{html.escape(x)}</li>" for x in items)


def _connections_html(project: Project) -> str:
    if not project.connections:
        return ""
    items = _list(describe_connection(project, c) for c in project.connections)
    return f"<p><b>부품 연결</b></p><ul>{items}</ul>"


def _materials_list(project: Project, materials: dict) -> str:
    pack = standards.effective_pack(project)
    return _list(
        f"{project.part(pid).name}: {pack.material_name(m)}" for pid, m in materials.items()
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
<p style="color:#8a6d3b;"><b>고지</b>: {disclaimer(method)}</p>
</body></html>
"""


# --- static ---------------------------------------------------------------------------


def _cases_html(project: Project, overview, us: str) -> str:
    if overview is None or len(overview.names) < 2:
        return ""
    rows = []
    for i, (name, s) in enumerate(zip(overview.names, overview.summaries, strict=True)):
        mark = " ◀ 가장 불리" if i == overview.worst else ""
        rows.append(
            f"<tr><td>{html.escape(name)}{mark}</td><td><b>{s.governing_fos:.2f}</b></td>"
            f"<td>{fmt(us, 'length', s.max_disp)}</td><td>{VERDICT_STYLE[s.verdict][0]}</td></tr>"
        )
    current = html.escape(project.case.name)
    return (
        "<h2>하중 케이스 비교</h2>"
        "<table width='100%' border='1' cellspacing='0' cellpadding='4'>"
        "<tr style='background:#eef1f5;'><th>케이스</th><th>안전율</th><th>최대 변형</th>"
        f"<th>판정</th></tr>{''.join(rows)}</table>"
        f"<p>아래 상세 결과는 '{current}' 케이스의 것입니다.</p>"
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
        crit = "폰 미세스 응력" if s.ductile else "최대 주응력"
        cap = "항복강도" if s.ductile else "인장강도"
        rows.append(
            f"<tr><td>{esc(s.name)}</td><td>{esc(s.material)}</td>"
            f"<td>{fmt(us, 'stress', s.max_stress, ',.1f')} ({crit})</td>"
            f"<td>{fmt(us, 'stress', s.capacity, ',.0f')} ({cap})</td>"
            f"<td><b>{s.min_fos:.2f}</b> / {s.target:.1f}</td><td>{fmt(us, 'length', s.max_disp)}</td>"
            f"<td>{VERDICT_STYLE[s.verdict][0]}</td></tr>"
        )
        if s.hotspot_fos is not None:
            rows.append(
                f"<tr><td colspan='2' align='right'>특이점({esc(s.singular_kind)}) 제외</td>"
                f"<td>{fmt(us, 'stress', s.hotspot_stress, ',.1f')}</td><td></td>"
                f"<td><b>{s.hotspot_fos:.2f}</b> / {s.target:.1f}</td><td></td>"
                f"<td>{VERDICT_STYLE[s.hotspot_verdict][0]}</td></tr>"
            )
        if s.fatigue_fos is not None:
            life = "무한 수명" if s.fatigue_life >= 1e9 else f"{s.fatigue_life:,.0f}회"
            rows.append(
                f"<tr><td colspan='4' align='right'>반복 하중 피로 검토</td>"
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
        f"<tr><td>{esc(b.name)}</td><td>{b.size} {b.grade}</td>"
        f"<td>{fmt(us, 'force', b.preload, ',.0f')}</td>"
        f"<td>{fmt(us, 'force', b.axial, ',.0f')}</td><td>{fmt(us, 'force', b.shear, ',.0f')}</td>"
        f"<td>{fmt(us, 'stress', b.stress, ',.0f')} / {fmt(us, 'stress', b.proof, ',.0f')}</td>"
        f"<td><b>{b.safety_factor:.2f}</b></td></tr>"
        for b in result.bolts or []
    )
    bolts = (
        "<h3>볼트</h3><table width='100%' border='1' cellspacing='0' cellpadding='4'>"
        "<tr style='background:#eef1f5;'><th>볼트</th><th>규격</th><th>체결력</th><th>축력</th>"
        "<th>전단력</th><th>응력 / 보증응력</th><th>안전율</th></tr>"
        f"{bolt_rows}</table>"
        if bolt_rows
        else ""
    )
    thermal = ""
    if result.thermal is not None:
        t = result.thermal.temperature
        thermal = (
            f"<p><b>열응력 포함</b>: 온도 {fmt(us, 'temperature', t.min(), '.1f')} ~ "
            f"{fmt(us, 'temperature', t.max(), '.1f')}, 응력 없는 기준 온도 "
            f"{fmt(us, 'temperature', project.reference_temperature, '.0f')}</p>"
        )
    notes = _list(summary.notes)
    at = [fmt(us, "length", float(v), ".1f") for v in summary.max_disp_at]
    nl = result.nonlinear
    if nl is None:
        method = "선형 탄성 정적해석 (작은 변형, 재료는 항복 전까지 선형으로 가정)"
    else:
        parts = []
        if nl.plasticity:
            parts.append(
                "재료 항복 고려(폰 미세스 소성, 항복강도 → 인장강도까지 직선 경화, 그 뒤 거의 일정)"
            )
        if nl.large_deformation:
            parts.append("큰 변형 고려(전체 라그랑주, 하중 방향은 처음 그대로)")
        method = (
            f"비선형 정적해석: {', '.join(parts)}. 하중 {nl.increments}단계, 뉴턴 반복 "
            f"{nl.iterations}회, 도달 하중 {nl.load_factor * 100:.0f}%"
        )
    body = f"""
{f"<p><b>참고</b></p><ul>{notes}</ul>" if notes else ""}
{_cases_html(project, overview, us)}
<h2>1. 부품별 결과</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>부품</th><th>재료</th><th>최대 응력</th><th>재료 강도</th>
<th>안전율 / 권장</th><th>최대 변형</th><th>판정</th></tr>
{"".join(rows)}
</table>
{bolts}
{thermal}
<p>최대 변형량: {fmt(us, "length", summary.max_disp)} (위치 X {at[0]}, Y {at[1]}, Z {at[2]})</p>

<h2>2. 해석 조건</h2>
<p><b>고정</b></p><ul>{fixtures}</ul>
<p><b>하중</b>{f" ({esc(project.case.name)})" if len(project.cases) > 1 else ""}</p><ul>{load_items}</ul>
{_connections_html(project)}
<p><b>반력</b> (고정면이 부품을 받치는 힘)</p><ul>{reactions}</ul>

<h2>3. 결과 그림</h2>
{_pictures(images)}

<h2>4. 해석 방법과 한계</h2>
<ul>
<li>{method}</li>
<li>연성재는 폰 미세스 응력과 항복강도로, 취성재는 최대 주응력과 인장강도로 안전율을 계산</li>
<li>권장 안전율은 하중을 잘 아는 정적 조건의 일반적 기계설계 권장값이며, 피로·충격·고온 조건은 별도 검토 필요</li>
<li>참고 기준: {esc(", ".join(pack.references))}</li>
</ul>
"""
    return _page(
        "구조 해석 보고서",
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        "비선형 정적 유한요소해석" if result.nonlinear is not None else "선형 정적 유한요소해석",
    )


# --- modal / thermal / buckling -------------------------------------------------------


def _modal_html(project, result, summary, images, materials) -> str:
    rows = "".join(
        f"<tr><td>{i + 1}차</td><td><b>{f:,.2f} Hz</b></td><td>{f * 60:,.0f} rpm</td>"
        f"<td>{p[0] * 100:.0f}% / {p[1] * 100:.0f}% / {p[2] * 100:.0f}%</td></tr>"
        for i, (f, p) in enumerate(zip(result.frequencies, result.participation, strict=True))
    )
    op = (
        f"<p>운전 진동수: {project.operating_frequency:,.2f} Hz "
        f"({project.operating_frequency * 60:,.0f} rpm)</p>"
        if project.operating_frequency > 0
        else "<p>운전 진동수: 입력하지 않음</p>"
    )
    fixtures = (
        _list(describe_fixture(project, f) for f in project.fixtures) or "<li>없음 (자유 진동)</li>"
    )
    notes = _list(summary.notes)
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>1. 고유진동수</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>차수</th><th>진동수</th><th>회전수 환산</th>
<th>유효질량 X / Y / Z</th></tr>{rows}</table>
{op}
<p>유효질량은 그 진동 모양이 각 방향 흔들림에 얼마나 크게 관여하는지를 나타냅니다.</p>
<h2>2. 해석 조건</h2>
<p><b>재료</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>고정</b></p><ul>{fixtures}</ul>
{_connections_html(project)}
<h2>3. 진동 모양</h2>
<p>색과 변형 크기는 상대적인 값입니다(실제 진폭은 가진력의 크기에 따라 달라집니다).</p>
{_pictures(images)}
<h2>4. 해석 방법과 한계</h2>
<ul><li>선형 고유치 해석 (감쇠 무시, 작은 진동 가정)</li>
<li>공진 판정: 운전 진동수 ±10% 이내 위험, ±20% 이내 주의</li></ul>
"""
    return _page(
        "진동(고유진동수) 해석 보고서",
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        "선형 고유진동수(모달) 유한요소해석",
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
            f"<tr><td>{html.escape(project.part(pid).name)}</td>"
            f"<td>{fmt(us, 'temperature', result.temperature[nodes].min(), '.1f')}</td>"
            f"<td><b>{fmt(us, 'temperature', result.temperature[nodes].max(), '.1f')}</b></td>"
            f"<td>{limit}</td></tr>"
        )
    notes = _list(summary.notes)
    if result.transient:
        from kstruct.report.summary import format_duration

        kind_note = (
            f"시간에 따른 열전도 해석: 처음 온도 "
            f"{fmt(us, 'temperature', project.initial_temperature, '.1f')}에서 "
            f"{format_duration(float(result.times[-1]))} 동안 (후진 오일러 "
            f"{result.iterations}단계). 표의 온도는 마지막 시점, 판정은 그동안의 최고 온도 기준"
        )
    else:
        kind_note = "정상상태 열전도 해석 (충분한 시간이 지나 온도가 더 이상 변하지 않는 상태)"
    has_radiation = any(ld.kind == LoadKind.RADIATION for ld in project.loads)
    has_table = any(m.conductivity_table for m in materials.values())
    method_note = "; ".join(
        [
            "복사는 '복사 냉각'을 지정한 면에서만 고려(반복 계산)"
            if has_radiation
            else "복사 열전달은 지정하지 않아 고려하지 않음",
            "온도에 따른 열전도율 변화를 반영(재료의 열전도율 표)"
            if has_table
            else "열전도율은 온도와 무관한 상수로 가정",
        ]
    )
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>1. 부품별 온도</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>부품</th><th>최저</th><th>최고</th><th>권장 최고 사용온도</th></tr>
{"".join(rows)}</table>
<p>들어온 열 {fmt(us, "power", result.heat_in)} · 빠져나간 열 {fmt(us, "power", result.heat_out)}</p>
<h2>2. 해석 조건</h2>
<p><b>재료</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>열 조건</b></p><ul>{loads}</ul>
{_connections_html(project)}
<p>지정하지 않은 면은 열이 드나들지 않는(단열) 것으로 가정합니다.</p>
<h2>3. 결과 그림</h2>
{_pictures(images)}
<h2>4. 해석 방법과 한계</h2>
<ul><li>{kind_note}</li>
<li>{method_note}</li>
<li>권장 최고 사용온도는 대략적인 값이며 재료 등급에 따라 다름</li></ul>
"""
    return _page(
        "열전달 해석 보고서",
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        "정상상태 열전도 유한요소해석",
    )


def _buckling_html(project, result, summary, images, materials) -> str:
    us = system_of(project)
    applied = float(np.linalg.norm(result.static.applied_load))
    rows = "".join(
        f"<tr><td>{i + 1}차</td><td><b>{f:,.2f}</b></td>"
        f"<td>{fmt(us, 'force', f * applied, ',.0f')}</td></tr>"
        for i, f in enumerate(result.factors)
    )
    fixtures = _list(describe_fixture(project, f) for f in project.fixtures)
    loads = _list(describe_load(project, ld) for ld in project.loads)
    notes = _list(summary.notes)
    body = f"""
{f"<ul>{notes}</ul>" if notes else ""}
<h2>1. 좌굴 하중</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>차수</th><th>하중 배수</th><th>좌굴 시 하중 합계(추정)</th></tr>
{rows}</table>
<p>하중 배수 = 좌굴이 일어나는 하중 ÷ 현재 하중. 현재 하중 합계 {fmt(us, "force", applied, ",.1f")}.</p>
<h2>2. 해석 조건</h2>
<p><b>재료</b></p><ul>{_materials_list(project, materials)}</ul>
<p><b>고정</b></p><ul>{fixtures}</ul>
<p><b>하중</b></p><ul>{loads}</ul>
{_connections_html(project)}
<h2>3. 좌굴 모양</h2>
<p>변형 크기는 상대값입니다.</p>
{_pictures(images)}
<h2>4. 해석 방법과 한계</h2>
<ul><li>선형 고유치 좌굴 해석 (초기 결함·소성·대변형 미고려)</li>
<li>권장 하중 배수 {standards.effective_pack(project).target_buckling:.1f} 이상</li></ul>
"""
    return _page(
        "좌굴 해석 보고서",
        project,
        summary.headline,
        summary.explanation,
        summary.verdict,
        body,
        result.mesh,
        "선형 고유치 좌굴 유한요소해석",
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
