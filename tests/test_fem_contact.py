"""Frictionless contact: transmits compression, never tension."""

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import (
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


def _stack(kind: ConnectionKind, force_z: float) -> Project:
    """Two 10x10x50 blocks stacked along z, lower one standing on rollers."""
    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = Project(parts=[Part(0, "lower", lower), Part(1, "upper", upper)])

    def face(pid, pt):
        return FaceRef(pid, face_at(project.part(pid).shape, pt))

    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [face(0, (0, 5, 25)), face(1, (0, 5, 75))]),
        Fixture(FixtureKind.ROLLER, [face(0, (5, 0, 25)), face(1, (5, 0, 75))]),
        Fixture(FixtureKind.ROLLER, [face(0, (5, 5, 0))]),
    ]
    project.connections = [Connection(kind, [face(0, (5, 5, 50))], [face(1, (5, 5, 50))])]
    project.loads = [Load(LoadKind.FORCE, [face(1, (5, 5, 100))], vector=(0, 0, force_z))]
    return project


def _solve(project):
    mesh = generate_mesh(project, "coarse")
    return mesh, solve_static(project, mesh, {0: STEEL, 1: STEEL})


def test_compression_is_transmitted_like_bonded():
    _, bonded = _solve(_stack(ConnectionKind.BONDED, -1000.0))
    mesh, contact = _solve(_stack(ConnectionKind.CONTACT, -1000.0))
    top = np.isclose(mesh.nodes[:, 2], 100)
    expected = -1000 * 100 / (100 * 210_000)
    assert contact.displacement[top, 2] == pytest.approx(expected, rel=1e-3)
    assert contact.stress[:, 2] == pytest.approx(-10.0, rel=1e-3)
    # nearly all pairs closed: quadratic faces carry ~no load at corner nodes, so a few
    # corner pairs may sit at zero force and open without affecting the result
    closed, total = contact.contact[0]
    assert closed >= 0.8 * total
    assert bonded.displacement[top, 2] == pytest.approx(expected, rel=1e-3)


def test_pulling_apart_is_reported():
    with pytest.raises(SolverError, match="contact unstable"):
        _solve(_stack(ConnectionKind.CONTACT, +1000.0))


def test_lap_joint_opens_on_one_side():
    """A cantilever plate resting on a fixed block, pushed down at its free end beyond the
    block: it pivots on the block's edge, so the contact closes only near that edge."""
    base = _box(0, 0, 0, 40, 20, 10)
    plate = _box(0, 0, 10, 100, 20, 4)
    project = Project(parts=[Part(0, "base", base), Part(1, "plate", plate)])

    def face(pid, pt):
        return FaceRef(pid, face_at(project.part(pid).shape, pt))

    project.fixtures = [
        Fixture(FixtureKind.FIXED, [face(0, (20, 10, 0))]),
        # the plate's left end is clamped down so it can only lever about the block edge
        Fixture(FixtureKind.FIXED, [face(1, (0, 10, 12))]),
    ]
    project.connections = [
        Connection(ConnectionKind.CONTACT, [face(0, (20, 10, 10))], [face(1, (20, 10, 10))])
    ]
    project.loads = [Load(LoadKind.FORCE, [face(1, (100, 10, 12))], vector=(0, 0, -200.0))]
    mesh, r = _solve(project)
    closed, total = r.contact[0]
    assert 0 < closed < total
