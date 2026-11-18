"""Prescribed displacement, elastic support, remote force and moment: closed-form checks."""

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.constraints import ConstraintError, _tri6_mass
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Material, Part, Project

E, NU = 210_000.0, 0.3
STEEL = Material("s", "s", "steel", E, NU, 7.85e-9, 250.0, 400.0)


def _project(shape) -> Project:
    return Project(parts=[Part(0, "p", shape, "s")])


def _face(project, point) -> FaceRef:
    return FaceRef(0, face_at(project.parts[0].shape, point))


def _bar(length=100.0) -> Project:
    """10x10xL bar along z on rollers at its three coordinate planes (uniaxial stress)."""
    project = _project(_box(0, 0, 0, 10, 10, length))
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0))]),
    ]
    return project


def test_tri6_surface_mass_integrates_the_shape_functions():
    m = _tri6_mass(np.array([2.0]))[0]
    assert m.sum() == pytest.approx(2.0)  # ∫ (Σ N_i)(Σ N_j) dA = A
    assert m[:3].sum(axis=1) == pytest.approx(0.0)  # corner nodes carry no area
    assert m[3:].sum(axis=1) == pytest.approx(2.0 / 3)


@pytest.mark.parametrize("quality", ["coarse"])
def test_prescribed_displacement_stretches_a_bar(quality):
    """End face pulled by δ along the bar only (sideways free): σ = E δ / L everywhere,
    the support carries σ A."""
    L, delta = 100.0, 0.05
    project = _bar(L)
    end = _face(project, (5, 5, L))
    project.fixtures.append(Fixture(FixtureKind.DISPLACEMENT, [end], (None, None, delta)))
    mesh = generate_mesh(project, quality)
    r = solve_static(project, mesh, {0: STEEL})
    sigma = E * delta / L
    assert r.stress[:, 2] == pytest.approx(sigma, rel=1e-6)
    assert r.displacement[mesh.face_nodes([end]), 2] == pytest.approx(delta, abs=1e-12)
    assert r.reactions[3][2] == pytest.approx(sigma * 100.0, rel=1e-6)
    assert sum(r.reactions)[2] == pytest.approx(0.0, abs=1e-6)  # nothing else pushes


def test_prescribed_displacement_needs_a_direction():
    project = _bar()
    project.fixtures.append(
        Fixture(FixtureKind.DISPLACEMENT, [_face(project, (5, 5, 100))], (None, None, None))
    )
    project.loads = [Load(LoadKind.GRAVITY, vector=(0, 0, -9810.0))]
    with pytest.raises(ConstraintError):
        solve_static(project, generate_mesh(project, "coarse"), {0: STEEL})


def test_elastic_support_adds_its_own_compliance():
    """Bar on a spring foundation of total stiffness k under its base, pushed down with F
    on top: the top moves F/k (spring) + F L/(E A) (bar)."""
    L, F, k = 100.0, 1000.0, 5_000.0
    project = _project(_box(0, 0, 0, 10, 10, L))
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, L / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, L / 2))]),
        Fixture(FixtureKind.ELASTIC, [_face(project, (5, 5, 0))], stiffness=k),
    ]
    top = _face(project, (5, 5, L))
    project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, -F))]
    mesh = generate_mesh(project, "coarse")
    r = solve_static(project, mesh, {0: STEEL})
    expected = F / k + F * L / (E * 100.0)
    assert -r.displacement[mesh.face_nodes([top]), 2].mean() == pytest.approx(expected, rel=1e-6)
    assert r.reactions[2] == pytest.approx([0, 0, F], abs=1e-6)  # the springs carry it all
    # the springs act along the normal only: without the rollers nothing holds a sideways
    # push, which is reported instead of solved
    project.fixtures = project.fixtures[2:]
    project.loads.append(Load(LoadKind.FORCE, [top], vector=(10.0, 0, 0)))
    with pytest.raises(SolverError):
        solve_static(project, mesh, {0: STEEL})


def _cantilever():
    L, b = 200.0, 20.0
    project = _project(_box(0, 0, 0, L, b, b))
    root, tip = _face(project, (0, 10, 10)), _face(project, (L, 10, 10))
    project.fixtures = [Fixture(FixtureKind.FIXED, [root])]
    return project, root, tip, L, b


def _tip(mesh, r, tip):
    return r.displacement[mesh.face_nodes([tip]), 2].mean()


def test_end_moment_bends_a_cantilever():
    """Pure moment M about y at the free end: w = M L² / (2 E I), σ = M c / I along the
    whole span, root reaction moment -M and no net force."""
    project, root, tip, L, b = _cantilever()
    M = 100_000.0  # N·mm, tip goes down (moment about -y lifts... sign below)
    project.loads = [Load(LoadKind.MOMENT, [tip], vector=(0.0, M, 0.0))]
    mesh = generate_mesh(project, "normal")
    r = solve_static(project, mesh, {0: STEEL})
    inertia = b**4 / 12
    # +M about y rotates +x towards -z: the tip goes down
    assert -_tip(mesh, r, tip) == pytest.approx(M * L**2 / (2 * E * inertia), rel=0.02)
    x, z = mesh.nodes[:, 0], mesh.nodes[:, 2]
    mid_top = np.flatnonzero((np.abs(x - L / 2) < 0.15 * L) & np.isclose(z, b))
    assert r.stress[mid_top, 0].mean() == pytest.approx(M * (b / 2) / inertia, rel=0.03)
    assert r.applied_load == pytest.approx([0, 0, 0], abs=1e-9)
    assert r.reactions[0] == pytest.approx([0, 0, 0], abs=1e-6)


def test_remote_force_is_force_plus_moment():
    """F at a point d beyond the tip = F at the tip + moment F d:
    w = F L³/(3EI) + F d L²/(2EI) (+ shear deflection of the force part)."""
    project, root, tip, L, b = _cantilever()
    F, d = 1000.0, 100.0
    project.loads = [Load(LoadKind.REMOTE, [tip], vector=(0, 0, -F), point=(L + d, b / 2, b / 2))]
    mesh = generate_mesh(project, "normal")
    r = solve_static(project, mesh, {0: STEEL})
    inertia = b**4 / 12
    shear = F * L / (5 / 6 * E / (2 * (1 + NU)) * b * b)
    expected = F * L**3 / (3 * E * inertia) + shear + F * d * L**2 / (2 * E * inertia)
    assert -_tip(mesh, r, tip) == pytest.approx(expected, rel=0.02)
    assert r.reactions[0] == pytest.approx([0, 0, F], abs=1e-6)
    # at the face centroid the remote force is just the force
    plain = Load(LoadKind.REMOTE, [tip], vector=(0, 0, -F), point=(L, b / 2, b / 2))
    project.loads = [plain]
    r2 = solve_static(project, mesh, {0: STEEL})
    project.loads = [Load(LoadKind.FORCE, [tip], vector=(0, 0, -F))]
    r3 = solve_static(project, mesh, {0: STEEL})
    assert _tip(mesh, r2, tip) == pytest.approx(_tip(mesh, r3, tip), rel=1e-6)


def test_new_conditions_survive_save_and_load(tmp_path):
    from kstruct.project_io import load_project, save_project

    project = _bar()
    end = _face(project, (5, 5, 100))
    project.fixtures += [
        Fixture(FixtureKind.DISPLACEMENT, [end], (0.1, None, -0.2)),
        Fixture(FixtureKind.ELASTIC, [end], stiffness=1234.5),
    ]
    project.loads = [
        Load(LoadKind.REMOTE, [end], vector=(1, 2, 3), point=(4, 5, 6)),
        Load(LoadKind.MOMENT, [end], vector=(7, 8, 9)),
    ]
    back = load_project(save_project(project, tmp_path / "p.kst"))
    assert back.fixtures[3].displacement == (0.1, None, -0.2)
    assert back.fixtures[4].stiffness == 1234.5
    assert back.loads[0].point == (4, 5, 6) and back.loads[1].vector == (7, 8, 9)


def test_prescribed_displacement_on_the_iterative_and_contact_paths(monkeypatch):
    """The same stretch through p-multigrid (constraints eliminated) and through the
    contact solver (two blocks pressed together by moving the top face down)."""
    import kstruct.fem.static as static
    from kstruct.model import Connection, ConnectionKind

    L, delta = 100.0, 0.05
    project = _bar(L)
    end = _face(project, (5, 5, L))
    project.fixtures.append(Fixture(FixtureKind.DISPLACEMENT, [end], (None, None, delta)))
    mesh = generate_mesh(project, "coarse")
    monkeypatch.setattr(static, "ITERATIVE_MIN_DOF", 0)
    shared = static._stiffness(project, mesh, {0: STEEL})
    r = solve_static(project, mesh, {0: STEEL}, shared=shared)
    assert shared.exact
    assert r.stress[:, 2] == pytest.approx(E * delta / L, rel=1e-6)

    # two 10x10x50 blocks in contact, squeezed by moving the top face down by δ
    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = Project(parts=[Part(0, "lower", lower, "s"), Part(1, "upper", upper, "s")])

    def face(pid, point):
        return FaceRef(pid, face_at(project.parts[pid].shape, point))

    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [face(0, (0, 5, 25)), face(1, (0, 5, 75))]),
        Fixture(FixtureKind.ROLLER, [face(0, (5, 0, 25)), face(1, (5, 0, 75))]),
        Fixture(FixtureKind.ROLLER, [face(0, (5, 5, 0))]),
        Fixture(FixtureKind.DISPLACEMENT, [face(1, (5, 5, 100))], (None, None, -delta)),
    ]
    project.connections = [
        Connection(ConnectionKind.CONTACT, [face(0, (5, 5, 50))], [face(1, (5, 5, 50))])
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_static(project, mesh, {0: STEEL, 1: STEEL})
    closed, total, _ = r.contact[0]
    assert closed == total  # pressed together everywhere
    # contact penalty adds a little compliance (~1e-3 of the displacements)
    assert r.stress[:, 2].mean() == pytest.approx(-E * delta / 100.0, rel=0.01)
    assert r.reactions[3][2] == pytest.approx(-E * delta, rel=0.01)  # σ A, A = 100 mm²
