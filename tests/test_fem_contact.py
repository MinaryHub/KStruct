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
    return mesh, solve_static(project, mesh, {p.id: STEEL for p in project.parts})


def test_compression_is_transmitted_like_bonded():
    _, bonded = _solve(_stack(ConnectionKind.BONDED, -1000.0))
    mesh, contact = _solve(_stack(ConnectionKind.CONTACT, -1000.0))
    top = np.isclose(mesh.nodes[:, 2], 100)
    expected = -1000 * 100 / (100 * 210_000)
    assert contact.displacement[top, 2] == pytest.approx(expected, rel=1e-3)
    assert contact.stress[:, 2] == pytest.approx(-10.0, rel=1e-3)
    # nearly all pairs closed: quadratic faces carry ~no load at corner nodes, so a few
    # corner pairs may sit at zero force and open without affecting the result
    closed, total, _ = contact.contact[0]
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
    closed, total, _ = r.contact[0]
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


# --- Coulomb friction ----------------------------------------------------------------


def _sheared_stack(friction: float, shear: float, kind=ConnectionKind.CONTACT) -> Project:
    """Lower block fixed at its base; a flat upper block is pressed down by 1000 N and
    pushed sideways (x) by `shear` N on its top face. A roller on one side face keeps it
    from drifting or turning sideways; along x only contact (and friction) holds it. The
    block is thin so the overturning moment barely changes the pressure (2.5 MPa ± 0.45)."""
    lower, upper = _box(0, 0, 0, 20, 20, 10), _box(0, 0, 10, 20, 20, 3)
    project = Project(parts=[Part(0, "lower", lower), Part(1, "upper", upper)])

    def face(pid, pt):
        return FaceRef(pid, face_at(project.part(pid).shape, pt))

    project.fixtures = [
        Fixture(FixtureKind.FIXED, [face(0, (10, 10, 0))]),
        Fixture(FixtureKind.ROLLER, [face(1, (10, 0, 11.5))]),  # no sideways drift / yaw
    ]
    project.connections = [
        Connection(kind, [face(0, (10, 10, 10))], [face(1, (10, 10, 10))], friction=friction)
    ]
    project.loads = [
        Load(LoadKind.FORCE, [face(1, (10, 10, 20))], vector=(shear, 0, -1000.0)),
    ]
    return project


def _top_shift(mesh, result) -> float:
    top = np.isclose(mesh.nodes[:, 2], 13.0)
    return float(result.displacement[top, 0].mean())


def test_friction_holds_a_pushed_block_like_a_bond():
    """τ = 0.5 MPa < μ p = 0.4 × (2.05…2.95) MPa everywhere: every pair sticks and the
    joint behaves as if bonded."""
    bonded = _sheared_stack(0.0, 200.0, ConnectionKind.BONDED)
    mesh, ref = _solve(bonded)
    mesh2, r = _solve(_sheared_stack(0.4, 200.0))
    closed, total, slipping = r.contact[0]
    # the rim lifts a little and its last row of pairs, nearly unloaded, slips: the core holds
    assert closed > 0.9 * total and slipping < 0.15 * closed
    assert _top_shift(mesh2, r) == pytest.approx(_top_shift(mesh, ref), rel=0.05)
    # the base carries the full shear and the full normal load (the sideways reaction is
    # the Poisson expansion of the pressed blocks held back by the sticking interface)
    assert r.reactions[0][[0, 2]] == pytest.approx([-200.0, 1000.0], abs=1.0)


def test_block_pushed_beyond_the_friction_limit_slides_away():
    """F = 500 N > μ N = 400 N: the upper block has nothing else holding it."""
    with pytest.raises(SolverError, match="contact unstable"):
        _solve(_sheared_stack(0.4, 500.0))


def test_frictionless_contact_cannot_hold_shear():
    with pytest.raises(SolverError, match="contact unstable"):
        _solve(_sheared_stack(0.0, 50.0))


def _sprung_stack(friction: float, shear: float) -> Project:
    """The sheared stack of `_sheared_stack` plus a slender rod (2x2x40 mm, axial
    stiffness EA/L = 21 kN/mm) bonded to the side of the upper block and fixed at its far
    end: whatever the friction cannot hold goes through the rod."""
    project = _sheared_stack(friction, shear)
    rod = _box(20, 9, 10.5, 40, 2, 2)
    project.parts.append(Part(2, "rod", rod))
    upper = project.part(1).shape
    project.fixtures.append(Fixture(FixtureKind.FIXED, [FaceRef(2, face_at(rod, (60, 10, 11.5)))]))
    project.connections.append(
        Connection(
            ConnectionKind.BONDED,
            [FaceRef(1, face_at(upper, (20, 10, 11.5)))],
            [FaceRef(2, face_at(rod, (20, 10, 11.5)))],
        )
    )
    return project


def test_sliding_contact_transmits_exactly_mu_times_the_normal_force():
    """Coulomb: with F = 700 N > μ N = 400 N the block slides and the rod carries the
    excess F − μ N; with F = 200 N the friction holds and the rod carries nothing."""
    _, held = _solve(_sprung_stack(0.4, 200.0))
    closed, _, slipping = held.contact[0]
    assert slipping < 0.2 * closed  # only the lightly pressed rim
    assert abs(held.reactions[2][0]) < 0.03 * 200.0  # the rod
    _, slid = _solve(_sprung_stack(0.4, 700.0))
    closed, _, slipping = slid.contact[0]
    assert slipping == closed
    assert slid.reactions[2][0] == pytest.approx(-(700.0 - 0.4 * 1000.0), rel=0.05)
    assert slid.reactions[0][0] == pytest.approx(-0.4 * 1000.0, rel=0.05)


def _bolted_lap(friction: float, shear: float, preload: float) -> Project:
    """Two 40x30x5 plates joined by one M10 bolt (preload P) with friction between them;
    the lower plate is fixed at its base and the upper one pushed sideways by F."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir

    from kstruct.geometry.samples import _hole
    from kstruct.model import Bolt

    def plate(z0):
        shape = _box(0, 0, z0, 40, 30, 5)
        return BRepAlgoAPI_Cut(shape, _hole(20, 15, z0 - 1, gp_Dir(0, 0, 1), 5.5, 7)).Shape()

    lower, upper = plate(0), plate(5)
    project = Project(parts=[Part(0, "lower", lower), Part(1, "upper", upper)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (20, 15, 0)))])]
    project.connections = [
        Connection(
            ConnectionKind.CONTACT,
            [FaceRef(0, face_at(lower, (5, 5, 5)))],
            [FaceRef(1, face_at(upper, (5, 5, 5)))],
            friction=friction,
        )
    ]
    project.bolts = [
        Bolt(
            FaceRef(0, face_at(lower, (25.5, 15, 2.5))),
            FaceRef(1, face_at(upper, (25.5, 15, 7.5))),
            "M10",
            "8.8",
            preload,
        )
    ]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (0, 15, 7.5)))], vector=(shear, 0, 0))
    ]
    return project


def test_friction_relieves_the_bolt_of_part_of_the_shear():
    """Without friction the bolt carries the whole 1000 N; with μ = 0.2 the preloaded
    interface takes a good part of it. (The plates have no head bearing faces here, so the
    bolt pulls on the hole walls and the plates dish: the friction grip is well below μ·P.)"""
    _, plain = _solve(_bolted_lap(0.0, 1000.0, 10_000.0))
    _, gripped = _solve(_bolted_lap(0.2, 1000.0, 10_000.0))
    assert plain.bolts[0].shear == pytest.approx(1000.0, rel=0.05)
    assert gripped.bolts[0].shear < 0.6 * plain.bolts[0].shear
    assert gripped.bolts[0].axial == pytest.approx(10_000.0, rel=0.05)
