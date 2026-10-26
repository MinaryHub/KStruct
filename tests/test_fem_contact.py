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


def test_contact_solver_reuses_the_factorization():
    """The active-set loop must not refactorize on every iteration: a lap joint that opens
    on one side settles in a few iterations with at most two factorizations."""
    from kstruct.fem.assembly import make_solver, rigid_body_modes, stiffness_matrix
    from kstruct.fem.constraints import build_constraints
    from kstruct.fem.contact import ContactSolver, build_contact
    from kstruct.fem.static import _load_vector
    from kstruct.model import ConnectionKind

    project = _stack(ConnectionKind.CONTACT, -500.0)
    # pull one corner so part of the interface opens (several active-set changes)
    project.loads[0].vector = (0, 0, 500.0)
    mesh = generate_mesh(project, "coarse")
    materials = {0: STEEL, 1: STEEL}
    c = build_constraints(project, mesh, contacts_as_bonded=False)
    contacts = [build_contact(mesh, x) for x in project.connections]
    K = stiffness_matrix(mesh, materials)
    scale = K.diagonal().max()
    Kp = K + c.penalty_matrix(mesh.n_nodes * 3, scale)
    free = np.flatnonzero(~c.fixed)
    near_null = rigid_body_modes(mesh.nodes)[free]
    f = _load_vector(project, mesh, materials)
    solver = ContactSolver(Kp, free, contacts, lambda A: make_solver(A.tocsr(), near_null), scale)
    u, K_c = solver.solve(f)
    assert contacts[0].active.sum() < len(contacts[0].active)  # the interface opened
    rhs = f[free] + solver._penalty()[1]
    direct = make_solver(K_c.tocsr(), near_null)(rhs)
    assert np.abs(u[free] - direct).max() < 1e-6 * np.abs(direct).max()
    # small systems refactorize because that is cheaper; pretend factorizing is expensive
    # (as for a large model) and change the load: the loop must then reuse the old
    # factorization through a low-rank update and still return an exact solution
    before = solver.factorizations
    solver._t_factorize = 1e9
    project.loads[0].vector = (0, 0, -500.0)
    u2, K_c2 = solver.solve(_load_vector(project, mesh, materials))
    assert solver.factorizations == before and solver.updates >= 1
    assert contacts[0].active.all()  # pressed together again
    # as accurate as a fresh factorization (the raw residual is not a usable measure:
    # the penalty rows amplify round-off far beyond what the displacements need)
    rhs = _load_vector(project, mesh, materials)[free] + solver._penalty()[1]
    direct = make_solver(K_c2.tocsr(), near_null)(rhs)
    assert np.abs(u2[free] - direct).max() < 1e-6 * np.abs(direct).max()
