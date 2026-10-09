"""Buckling verification against Euler's column formula."""

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh
from kstruct.fem.buckling import solve_buckling
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Material, Part, Project

STEEL = Material("s", "s", "steel", 210_000.0, 0.3, 7.85e-9, 250.0, 400.0)


def _column(L=200.0, b=10.0, force=-1000.0):
    shape = _box(0, 0, 0, b, b, L)
    project = Project(parts=[Part(0, "col", shape)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (5, 5, 0)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (5, 5, L)))], vector=(0, 0, force))
    ]
    return project


def test_cantilever_column_matches_euler():
    L, b, P = 200.0, 10.0, 1000.0
    project = _column(L, b, -P)
    r = solve_buckling(project, generate_mesh(project, "normal"), {0: STEEL}, n_modes=3)
    p_cr = np.pi**2 * 210_000 * (b**4 / 12) / (4 * L**2)  # fixed-free (K = 2)
    # square section: two equal sway modes
    assert r.factors[0] == pytest.approx(p_cr / P, rel=0.03)
    assert r.factors[1] == pytest.approx(p_cr / P, rel=0.03)
    assert r.factors[2] > 5 * r.factors[0]
    # the first mode is sideways sway of the free end
    top = np.isclose(r.mesh.nodes[:, 2], L)
    # (equal modes may mix x and y, so compare the in-plane magnitude)
    assert np.linalg.norm(r.shapes[0][top][:, :2], axis=1).max() == pytest.approx(1.0, rel=0.05)


def test_tension_does_not_buckle():
    project = _column(force=+1000.0)
    with pytest.raises(SolverError):
        solve_buckling(project, generate_mesh(project, "coarse"), {0: STEEL}, n_modes=2)
