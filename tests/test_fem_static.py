"""Static solver verification against closed-form solutions."""

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.constraints import ConstraintError
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import (
    GRAVITY,
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

STEEL = Material(
    id="steel",
    name="steel",
    category="steel",
    youngs_modulus=210_000.0,
    poisson_ratio=0.3,
    density=7.85e-9,
    yield_strength=250.0,
    tensile_strength=400.0,
    thermal_expansion=12e-6,
)


def _project(*shapes) -> Project:
    return Project(parts=[Part(id=i, name=f"p{i}", shape=s) for i, s in enumerate(shapes)])


def _face(project: Project, point, part_id: int = 0) -> FaceRef:
    return FaceRef(part_id, face_at(project.part(part_id).shape, point))


def _symmetric_bar(length=100.0):
    """10x10xL bar along z, held by rollers on its three coordinate planes: free to
    contract laterally, so the stress state is exactly uniaxial."""
    project = _project(_box(0, 0, 0, 10, 10, length))
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0))]),
    ]
    return project


def _solve(project, quality="coarse"):
    mesh = generate_mesh(project, quality)
    return mesh, solve_static(project, mesh, {p.id: STEEL for p in project.parts})


def test_uniaxial_tension_is_exact():
    project = _symmetric_bar()
    top = _face(project, (5, 5, 100))
    project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, 1000.0))]
    mesh, r = _solve(project)

    top_nodes = mesh.face_nodes([top])
    assert r.displacement[top_nodes, 2] == pytest.approx(1000 * 100 / (100 * 210_000), rel=1e-6)
    assert r.stress[:, 2] == pytest.approx(10.0, rel=1e-6)
    assert r.von_mises.max() == pytest.approx(10.0, rel=1e-6)
    lateral = np.isclose(mesh.nodes[:, 0], 10.0)
    assert r.displacement[lateral, 0] == pytest.approx(-0.3 * 10 / 210_000 * 10, rel=1e-6)
    # equilibrium: reactions balance the applied load
    assert sum(r.reactions) + r.applied_load == pytest.approx(np.zeros(3), abs=1e-6)


def test_cantilever_tip_deflection():
    L, b, P = 200.0, 20.0, 1000.0
    project = _project(_box(0, 0, 0, L, b, b))
    root, tip = _face(project, (0, 10, 10)), _face(project, (L, 10, 10))
    project.fixtures = [Fixture(FixtureKind.FIXED, [root])]
    project.loads = [Load(LoadKind.FORCE, [tip], vector=(0, 0, -P))]
    mesh, r = _solve(project, "normal")

    E, nu = 210_000.0, 0.3
    inertia = b**4 / 12
    shear = P * L / (5 / 6 * E / (2 * (1 + nu)) * b * b)
    expected = P * L**3 / (3 * E * inertia) + shear  # Timoshenko beam
    tip_uz = r.displacement[mesh.face_nodes([tip]), 2].mean()
    assert -tip_uz == pytest.approx(expected, rel=0.02)

    # bending stress on the top fibre around mid-span: M(x) c / I at each node's own x
    x, z = mesh.nodes[:, 0], mesh.nodes[:, 2]
    mid_top = np.flatnonzero((np.abs(x - L / 2) < 0.1 * L) & np.isclose(z, b))
    assert len(mid_top) > 0
    expected = P * (L - x[mid_top]) * (b / 2) / inertia
    assert r.stress[mid_top, 0].mean() == pytest.approx(expected.mean(), rel=0.03)
    assert r.reactions[0] == pytest.approx([0, 0, P], abs=1e-6)


def test_gravity_reaction_equals_weight():
    project = _project(_box(0, 0, 0, 50, 50, 20))
    project.fixtures = [Fixture(FixtureKind.FIXED, [_face(project, (25, 25, 0))])]
    project.loads = [Load(LoadKind.GRAVITY, vector=(0, 0, -GRAVITY))]
    _, r = _solve(project)
    weight = 50 * 50 * 20 * STEEL.density * GRAVITY
    assert r.reactions[0][2] == pytest.approx(weight, rel=1e-9)


def test_free_thermal_expansion_has_no_stress():
    project = _symmetric_bar()
    project.loads = [Load(LoadKind.TEMPERATURE, magnitude=100.0)]
    mesh, r = _solve(project)
    top = mesh.face_nodes([_face(project, (5, 5, 100))])
    assert r.displacement[top, 2] == pytest.approx(12e-6 * 100 * 100, rel=1e-6)
    assert np.abs(r.von_mises).max() < 1e-6


def test_bonded_parts_act_as_one():
    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = _project(lower, upper)
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, 25), 0), _face(project, (0, 5, 75), 1)]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, 25), 0), _face(project, (5, 0, 75), 1)]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0), 0)]),
    ]
    project.connections = [
        Connection(
            ConnectionKind.BONDED,
            faces_a=[_face(project, (5, 5, 50), 0)],
            faces_b=[_face(project, (5, 5, 50), 1)],
        )
    ]
    top = _face(project, (5, 5, 100), 1)
    project.loads = [Load(LoadKind.FORCE, [top], vector=(0, 0, 1000.0))]
    mesh, r = _solve(project)
    assert r.displacement[mesh.face_nodes([top]), 2] == pytest.approx(
        1000 * 100 / (100 * 210_000), rel=1e-3
    )


def test_unheld_part_is_reported():
    project = _project(_box(0, 0, 0, 10, 10, 10))
    project.fixtures = [Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0))])]
    project.loads = [Load(LoadKind.FORCE, [_face(project, (5, 5, 10))], vector=(1, 0, 0))]
    with pytest.raises(SolverError):
        _solve(project)


def test_no_fixture_is_reported():
    project = _project(_box(0, 0, 0, 10, 10, 10))
    with pytest.raises(ConstraintError):
        _solve(project)


def test_iterative_solver_matches_direct(monkeypatch):
    from kstruct.fem import assembly
    from kstruct.geometry.samples import l_bracket

    project = _project(l_bracket())
    project.fixtures = [Fixture(FixtureKind.FIXED, [_face(project, (0, 30, 40))])]
    project.loads = [Load(LoadKind.FORCE, [_face(project, (90, 30, 8))], vector=(0, 0, -500.0))]
    mesh, direct = _solve(project, "normal")
    monkeypatch.setattr(assembly, "DIRECT_SOLVER_MAX_DOF", 0)
    iterative = solve_static(project, mesh, {0: STEEL})
    tip = np.abs(direct.displacement).max()
    assert np.abs(iterative.displacement - direct.displacement).max() < 1e-4 * tip


def test_pin_allows_rotation_only():
    """Plate on a pin through a hole, pulled sideways at the far end: a pin alone cannot
    stop rotation, so it must be reported as unstable; adding a roller stabilises it."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir

    from kstruct.geometry.samples import _hole

    plate = BRepAlgoAPI_Cut(_box(0, 0, 0, 100, 20, 5), _hole(10, 10, -1, gp_Dir(0, 0, 1), 4, 7))
    project = _project(plate.Shape())
    hole = _face(project, (14, 10, 2.5))
    end = _face(project, (100, 10, 2.5))
    project.fixtures = [Fixture(FixtureKind.PIN, [hole])]
    project.loads = [Load(LoadKind.FORCE, [end], vector=(0, 100.0, 0))]
    with pytest.raises(SolverError):
        _solve(project)

    project.loads = [Load(LoadKind.FORCE, [end], vector=(500.0, 0, 0))]  # pure tension
    project.fixtures.append(Fixture(FixtureKind.ROLLER, [_face(project, (50, 0, 2.5))]))
    project.fixtures.append(Fixture(FixtureKind.ROLLER, [_face(project, (50, 10, 0))]))
    mesh, r = _solve(project)
    total = sum(r.reactions)
    assert total + r.applied_load == pytest.approx(np.zeros(3), abs=1e-3)
    # the pin carries the whole axial load
    assert r.reactions[0][0] == pytest.approx(-500.0, rel=1e-3)


def test_meshing_timeout_is_reported(monkeypatch):
    from kstruct.fem import MeshingError
    from kstruct.fem import mesh as mesh_module

    monkeypatch.setattr(mesh_module, "MESH_TIMEOUT", 0.01)  # nothing can mesh this fast
    with pytest.raises(MeshingError, match="끝나지 않았습니다"):
        generate_mesh(_project(_box(0, 0, 0, 10, 10, 10)), "coarse")


def _pinned_plate() -> Project:
    """Plate on a pin through a hole, pulled along its length, with rollers against
    rotation and drift."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir

    from kstruct.geometry.samples import _hole

    plate = BRepAlgoAPI_Cut(_box(0, 0, 0, 100, 20, 5), _hole(10, 10, -1, gp_Dir(0, 0, 1), 4, 7))
    project = _project(plate.Shape())
    project.fixtures = [
        Fixture(FixtureKind.PIN, [_face(project, (14, 10, 2.5))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (50, 0, 2.5))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (50, 10, 0))]),
    ]
    project.loads = [Load(LoadKind.FORCE, [_face(project, (100, 10, 2.5))], vector=(500.0, 0, 0))]
    return project


def test_multigrid_solver_matches_the_factorization(monkeypatch):
    """Large models are solved by p-multigrid CG with the constraints eliminated instead
    of penalised. Forced onto small models here: bonded parts (ties), a pin (two coupled
    equations per node) and an unheld part (falls back and is diagnosed)."""
    import kstruct.fem.static as static

    def both(project, materials):
        mesh = generate_mesh(project, "coarse")
        monkeypatch.setattr(static, "ITERATIVE_MIN_DOF", 10**9)
        direct = static.solve_static(project, mesh, materials)
        monkeypatch.setattr(static, "ITERATIVE_MIN_DOF", 0)
        shared = static._stiffness(project, mesh, materials)
        iterative = static.solve_static(project, mesh, materials, shared=shared)
        assert shared.exact and 0 < shared.iterations < 60
        return direct, iterative

    # two bonded blocks pulled apart: the exact tie against the penalty tie
    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = _project(lower, upper)
    project.fixtures = [Fixture(FixtureKind.FIXED, [_face(project, (5, 5, 0), 0)])]
    project.connections = [
        Connection(
            ConnectionKind.BONDED,
            faces_a=[_face(project, (5, 5, 50), 0)],
            faces_b=[_face(project, (5, 5, 50), 1)],
        )
    ]
    project.loads = [
        Load(LoadKind.FORCE, [_face(project, (5, 5, 100), 1)], vector=(50.0, 0, 1000.0))
    ]
    direct, iterative = both(project, {0: STEEL, 1: STEEL})
    assert iterative.displacement == pytest.approx(direct.displacement, rel=1e-3, abs=1e-7)
    assert iterative.von_mises.max() == pytest.approx(direct.von_mises.max(), rel=1e-3)
    assert iterative.reactions[0] == pytest.approx([-50.0, 0.0, -1000.0], abs=1e-4)

    # a pinned hole: the reaction comes from constraint multipliers, not spring forces
    project = _pinned_plate()
    direct, iterative = both(project, {0: STEEL})
    assert iterative.displacement == pytest.approx(direct.displacement, rel=1e-3, abs=1e-7)
    total = sum(iterative.reactions)
    assert total == pytest.approx(-iterative.applied_load, abs=1e-3)
    for a, b in zip(iterative.reactions, direct.reactions, strict=True):
        assert a == pytest.approx(b, rel=1e-3, abs=1e-2)

    # nothing holds the upper block: the iteration cannot solve it, the factorization
    # that takes over reports it
    project = _project(lower, upper)
    project.fixtures = [Fixture(FixtureKind.FIXED, [_face(project, (5, 5, 0), 0)])]
    project.loads = [Load(LoadKind.FORCE, [_face(project, (5, 5, 100), 1)], vector=(0, 0, 1.0))]
    monkeypatch.setattr(static, "ITERATIVE_MIN_DOF", 0)
    with pytest.raises(SolverError, match="unstable"):
        _solve(project)
