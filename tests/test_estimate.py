"""The size / duration sentence shown before a solve."""

import numpy as np

from kstruct.fem.estimate import solve_estimate
from kstruct.fem.mesh import FEMesh
from kstruct.model import AnalysisKind, Connection, ConnectionKind, FaceRef, Project


def _mesh(nodes: int, tets: int) -> FEMesh:
    empty = np.zeros((0, 6), dtype=int)
    return FEMesh(
        np.zeros((nodes, 3)), np.zeros((tets, 10), dtype=int), np.zeros(tets, dtype=int),
        empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int),
    )  # fmt: skip


def test_estimate_follows_size_and_analysis_kind():
    project = Project()
    text, warning = solve_estimate(project, _mesh(12_500, 6_500))
    assert "요소 6,500개" in text and "미지수 37,500개" in text
    assert "몇 초" in text and not warning
    # the same model with contact or as a vibration run is a heavier job
    assert "몇 초" in solve_estimate(project, _mesh(30_000, 18_000))[0]
    project.analysis = AnalysisKind.MODAL
    assert "수십 초" in solve_estimate(project, _mesh(30_000, 18_000))[0]
    project.analysis = AnalysisKind.STATIC
    project.connections = [Connection(ConnectionKind.CONTACT, [FaceRef(0, 0)], [FaceRef(1, 0)])]
    text, warning = solve_estimate(project, _mesh(30_000, 18_000))
    assert "접촉" in text and not warning
    # far too large for a contact run: a warning that says what to do
    text, warning = solve_estimate(project, _mesh(200_000, 120_000))
    assert warning and "메시 품질" in text and "취소" in text
    # a plain strength run of that size is still fine
    project.connections = []
    assert not solve_estimate(project, _mesh(200_000, 120_000))[1]
