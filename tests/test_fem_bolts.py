"""Bolted joint bookkeeping: equilibrium of bolt forces and force-controlled preload."""

import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.gp import gp_Dir

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.bolts import recommended_preload, suggest_size
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box, _hole
from kstruct.model import (
    Bolt,
    Connection,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Material,
    Part,
    Project,
)

STEEL = Material("s", "s", "steel", 210_000.0, 0.3, 7.85e-9, 250.0, 400.0)
HOLES = [(10, 10), (70, 10), (40, 50)]  # not in a line: three bolts fix all six motions


def _plate(z0: float):
    shape = _box(0, 0, z0, 80, 60, 5)
    for x, y in HOLES:
        shape = BRepAlgoAPI_Cut(shape, _hole(x, y, z0 - 1, gp_Dir(0, 0, 1), 5.5, 7)).Shape()
    return shape


def _joint(force=(0.0, 0.0, 0.0), preload=0.0, interface=None) -> Project:
    lower, upper = _plate(0), _plate(5)
    project = Project(parts=[Part(0, "lower", lower), Part(1, "upper", upper)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (40, 30, 0)))])]
    project.bolts = [
        Bolt(
            FaceRef(0, face_at(lower, (x + 5.5, y, 2.5))),
            FaceRef(1, face_at(upper, (x + 5.5, y, 7.5))),
            "M10",
            "8.8",
            preload,
        )
        for x, y in HOLES
    ]
    if interface is not None:
        project.connections = [
            Connection(
                interface,
                [FaceRef(0, face_at(lower, (40, 30, 5)))],
                [FaceRef(1, face_at(upper, (40, 30, 5)))],
            )
        ]
    if any(force):
        project.loads = [
            Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (40, 30, 10)))], vector=force)
        ]
    return project


def _solve(project):
    mesh = generate_mesh(project, "coarse")
    return solve_static(project, mesh, {0: STEEL, 1: STEEL})


def test_pull_off_load_goes_through_the_bolts():
    r = _solve(_joint(force=(0, 0, 3000.0)))
    assert sum(b.axial for b in r.bolts) == pytest.approx(3000.0, rel=1e-6)
    assert all(b.axial > 0 for b in r.bolts)


def test_shear_load_goes_through_the_bolts():
    r = _solve(_joint(force=(1500.0, 0, 0)))
    assert sum(b.shear for b in r.bolts) == pytest.approx(1500.0, rel=0.05)


def test_preload_alone_is_carried_exactly():
    P = recommended_preload("M10", "8.8")
    assert P == pytest.approx(0.7 * 640 * 58.0)
    r = _solve(_joint(preload=P, interface=ConnectionKind.BONDED))
    assert [b.axial for b in r.bolts] == pytest.approx([P] * 3, rel=1e-9)
    assert r.bolts[0].stress == pytest.approx(P / 58.0, rel=1e-3)  # + tiny shear
    assert r.bolts[0].safety_factor == pytest.approx(1 / 0.7, rel=1e-3)


def test_size_suggestion():
    assert suggest_size(11.0) == "M10"  # M10 clearance hole
    assert suggest_size(8.5) == "M8"  # M10 tapped hole
    assert suggest_size(2.0) == "M3"


def test_preload_needs_the_parts_to_touch():
    with pytest.raises(SolverError, match="bolt without contact"):
        _solve(_joint(preload=10_000.0))


def test_preloaded_contact_joint():
    """Small pull: the clamped interface takes most of it. Pull beyond the total clamp
    force: the plates separate and the bolts carry the whole load."""
    P = 10_000.0
    small = _solve(_joint(force=(0, 0, 3000.0), preload=P, interface=ConnectionKind.CONTACT))
    change = sum(b.axial for b in small.bolts) - 3 * P
    assert abs(change) < 0.5 * 3000.0

    big = _solve(_joint(force=(0, 0, 60_000.0), preload=P, interface=ConnectionKind.CONTACT))
    closed, total = big.contact[0]
    assert closed < 0.2 * total
    # once separated the bolts carry the whole pull; prying at the plate edges (the few
    # pairs still pressed) can only add to it
    assert sum(b.axial for b in big.bolts) >= 0.99 * 60_000.0
