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

from kstruct import standards
from kstruct.fem.static import StaticResult
from kstruct.model import FixtureKind, LoadKind, Project
from kstruct.report.summary import VERDICT_STYLE, Summary

FIXTURE_NAMES = {
    FixtureKind.FIXED: "완전 고정",
    FixtureKind.ROLLER: "미끄럼 지지",
    FixtureKind.PIN: "핀 지지",
}
DISCLAIMER = (
    "이 보고서는 선형 정적 유한요소해석 결과로, 설계 검토를 돕기 위한 참고 자료입니다. "
    "실제 제작·사용 여부는 하중 조건, 재료 성적서, 제작 품질, 관련 법규를 고려하여 "
    "자격을 갖춘 기술자가 최종 판단해야 합니다."
)


def _faces(project: Project, refs) -> str:
    multi = len(project.parts) > 1
    return ", ".join(
        (f"{project.part(r.part_id).name} " if multi else "") + f"면#{r.face_index + 1}"
        for r in refs
    )


def _img(png: bytes, width: int = 640) -> str:
    data = base64.b64encode(png).decode()
    return f'<img src="data:image/png;base64,{data}" width="{width}">'


def build_html(
    project: Project, result: StaticResult, summary: Summary, images: list[tuple[str, bytes]]
) -> str:
    pack = standards.pack(project.standard)
    title, color, bg = VERDICT_STYLE[summary.verdict]
    esc = html.escape
    rows = []
    for s in summary.parts:
        crit = "폰 미세스 응력" if s.ductile else "최대 주응력"
        cap = "항복강도" if s.ductile else "인장강도"
        rows.append(
            f"<tr><td>{esc(s.name)}</td><td>{esc(s.material)}</td>"
            f"<td>{s.max_stress:,.1f} MPa ({crit})</td><td>{s.capacity:,.0f} MPa ({cap})</td>"
            f"<td><b>{s.min_fos:.2f}</b> / {s.target:.1f}</td><td>{s.max_disp:.4g} mm</td>"
            f"<td>{VERDICT_STYLE[s.verdict][0]}</td></tr>"
        )

    fixtures = "".join(
        f"<li>{FIXTURE_NAMES[f.kind]}: {esc(_faces(project, f.faces))}</li>"
        for f in project.fixtures
    )
    loads = []
    for ld in project.loads:
        if ld.kind == LoadKind.FORCE:
            loads.append(
                f"힘 {np.linalg.norm(ld.vector):,.1f} N, 방향 ({', '.join(f'{c:+.0f}' for c in ld.vector)})"
                f": {esc(_faces(project, ld.faces))}"
            )
        elif ld.kind == LoadKind.PRESSURE:
            loads.append(f"압력 {ld.magnitude:,.4g} MPa: {esc(_faces(project, ld.faces))}")
        elif ld.kind == LoadKind.GRAVITY:
            loads.append(f"자중 {np.linalg.norm(ld.vector) / 9810:.2g} g")
        else:
            loads.append(f"온도 변화 {ld.magnitude:+.1f} °C")
    load_items = "".join(f"<li>{x}</li>" for x in loads)
    connections = "".join(
        f"<li>접합: {esc(_faces(project, c.faces_a))} ↔ {esc(_faces(project, c.faces_b))}</li>"
        for c in project.connections
    )
    reactions = "".join(
        f"<li>{FIXTURE_NAMES[f.kind]} ({esc(_faces(project, f.faces))}): "
        f"X {r[0]:+,.1f} N, Y {r[1]:+,.1f} N, Z {r[2]:+,.1f} N</li>"
        for f, r in zip(project.fixtures, summary.reactions, strict=True)
    )
    notes = "".join(f"<li>{esc(n)}</li>" for n in summary.notes)
    pictures = "".join(f"<h3>{esc(cap)}</h3><p>{_img(png)}</p>" for cap, png in images)
    source = esc(project.source_path.name) if project.source_path else "-"
    mesh = result.mesh

    return f"""
<html><body style="font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif; font-size: 10pt;">
<h1>구조 해석 보고서</h1>
<p>모델: {source} &nbsp;|&nbsp; 작성일: {dt.date.today():%Y-%m-%d} &nbsp;|&nbsp; 적용 기준: {esc(pack.label)}</p>
<table width="100%" cellpadding="10" style="background:{bg}; border: 2px solid {color};">
<tr><td><span style="font-size:16pt; color:{color};"><b>{esc(summary.headline)}</b></span><br>
{esc(summary.explanation)}</td></tr></table>
{f"<p><b>참고</b></p><ul>{notes}</ul>" if notes else ""}

<h2>1. 부품별 결과</h2>
<table width="100%" border="1" cellspacing="0" cellpadding="4">
<tr style="background:#eef1f5;"><th>부품</th><th>재료</th><th>최대 응력</th><th>재료 강도</th>
<th>안전율 / 권장</th><th>최대 변형</th><th>판정</th></tr>
{"".join(rows)}
</table>
<p>최대 변형량: {summary.max_disp:.4g} mm (위치 X {summary.max_disp_at[0]:.1f}, Y {summary.max_disp_at[1]:.1f}, Z {summary.max_disp_at[2]:.1f} mm)</p>

<h2>2. 해석 조건</h2>
<p><b>고정</b></p><ul>{fixtures}</ul>
<p><b>하중</b></p><ul>{load_items}</ul>
{f"<p><b>부품 연결</b></p><ul>{connections}</ul>" if connections else ""}
<p><b>메시</b>: 2차 사면체 요소 {len(mesh.tets):,}개, 절점 {mesh.n_nodes:,}개</p>
<p><b>반력</b> (고정면이 부품을 받치는 힘)</p><ul>{reactions}</ul>

<h2>3. 결과 그림</h2>
{pictures}

<h2>4. 해석 방법과 한계</h2>
<ul>
<li>선형 탄성 정적해석 (작은 변형, 재료는 항복 전까지 선형으로 가정)</li>
<li>연성재는 폰 미세스 응력과 항복강도로, 취성재는 최대 주응력과 인장강도로 안전율을 계산</li>
<li>권장 안전율은 하중을 잘 아는 정적 조건의 일반적 기계설계 권장값이며, 피로·충격·고온 조건은 별도 검토 필요</li>
<li>참고 기준: {esc(", ".join(pack.references))}</li>
</ul>
<p style="color:#8a6d3b;"><b>고지</b>: {DISCLAIMER}</p>
</body></html>
"""


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
