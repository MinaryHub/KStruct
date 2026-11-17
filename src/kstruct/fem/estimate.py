"""Rough size and duration of a solve, shown before the user starts it.

The times are orders of magnitude measured on a four-core desktop (see the table in
fem/multigrid.py); they are worded as ranges on purpose.
"""

from __future__ import annotations

from kstruct.fem.mesh import FEMesh
from kstruct.model import AnalysisKind, ConnectionKind, Project

# degrees of freedom up to which a run takes "seconds", "tens of seconds", "minutes"
_PLAIN_STATIC = (150_000, 600_000, 2_000_000)  # p-multigrid: time grows linearly
_FACTORIZED = (40_000, 120_000, 300_000)  # contact, modal, buckling: factorizations
_LABELS = ("몇 초", "수십 초", "몇 분")


def solve_estimate(project: Project, mesh: FEMesh) -> tuple[str, bool]:
    """(sentence for the solve panel, whether it is a warning)."""
    dof = mesh.n_nodes * 3
    size = f"요소 {len(mesh.tets):,}개 · 미지수 {dof:,}개"
    contact = project.analysis == AnalysisKind.STATIC and any(
        c.kind == ConnectionKind.CONTACT for c in project.connections
    )
    heavy = contact or project.analysis in (AnalysisKind.MODAL, AnalysisKind.BUCKLING)
    if project.analysis == AnalysisKind.THERMAL:
        dof //= 3  # one unknown per node
    limits = _FACTORIZED if heavy else _PLAIN_STATIC
    cases = len(project.cases) if project.analysis == AnalysisKind.STATIC else 1
    for limit, label in zip(limits, _LABELS, strict=True):
        if dof <= limit:
            extra = ""
            if contact:
                extra = " 접촉 계산은 반복이 필요해 더 걸릴 수 있습니다."
            elif cases > 1 and label != _LABELS[0]:
                extra = f" 하중 케이스 {cases}개는 한 번의 준비 계산을 함께 씁니다."
            return f"{size} — 보통 {label} 안에 끝납니다.{extra}", False
    reason = "접촉·진동·좌굴 해석은 큰 모델에서 오래 걸립니다. " if heavy else ""
    return (
        f"{size} — 모델이 커서 오래 걸리고 메모리가 부족할 수 있습니다. {reason}"
        "5단계에서 메시 품질을 한 단계 낮추는 것을 권장합니다. (해석 중 '취소'로 멈출 수 있습니다.)",
        True,
    )
