"""Yielding and large deformation against closed-form solutions."""

from dataclasses import replace

import numpy as np
import pytest

from kstruct.fem import generate_mesh, solve_static
from kstruct.fem.nonlinear import solve_nonlinear
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Material, Part, Project

E, NU, SY, RM = 210_000.0, 0.3, 250.0, 400.0
ET = 2_000.0  # tangent modulus beyond yield
STEEL = Material("s", "s", "steel", E, NU, 7.85e-9, SY, RM, tangent_modulus=ET)
AREA = 100.0  # 10 x 10 bar


def _project(shape, **flags) -> Project:
    return Project(parts=[Part(0, "p", shape, "s")], **flags)


def _face(project, point) -> FaceRef:
    return FaceRef(0, face_at(project.parts[0].shape, point))


def _bar(length=100.0, **flags) -> Project:
    """10x10xL bar along z on rollers at its three coordinate planes (uniaxial stress)."""
    project = _project(_box(0, 0, 0, 10, 10, length), **flags)
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, length / 2))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0))]),
    ]
    return project


def test_below_yield_the_nonlinear_run_equals_the_linear_one():
    project = _bar(plasticity=True)
    end = _face(project, (5, 5, 100))
    project.loads = [Load(LoadKind.FORCE, [end], vector=(0, 0, 0.5 * SY * AREA))]
    mesh = generate_mesh(project, "coarse")
    linear = solve_static(project, mesh, {0: STEEL})
    r = solve_nonlinear(project, mesh, {0: STEEL})
    assert r.nonlinear.completed and not r.nonlinear.yielded
    assert r.displacement == pytest.approx(linear.displacement, rel=1e-6, abs=1e-9)
    assert r.stress[:, 2] == pytest.approx(0.5 * SY, rel=1e-6)
    assert r.nonlinear.permanent is None
    assert r.reactions[2][2] == pytest.approx(-0.5 * SY * AREA, rel=1e-6)


def test_bar_pulled_past_yield_follows_the_bilinear_curve():
    """Imposed stretch: σ = σy + E_t (ε − εy) everywhere, the support carries σ A."""
    L, strain = 100.0, 0.01
    project = _bar(L, plasticity=True)
    end = _face(project, (5, 5, L))
    project.fixtures.append(Fixture(FixtureKind.DISPLACEMENT, [end], (None, None, strain * L)))
    mesh = generate_mesh(project, "coarse")
    r = solve_nonlinear(project, mesh, {0: STEEL})
    sigma = SY + ET * (strain - SY / E)
    assert r.nonlinear.completed
    assert r.stress[:, 2] == pytest.approx(sigma, rel=1e-4)
    # equivalent plastic strain = total minus elastic part
    assert r.nonlinear.plastic_strain == pytest.approx(strain - sigma / E, rel=1e-3)
    assert r.reactions[3][2] == pytest.approx(sigma * AREA, rel=1e-4)


def test_overload_leaves_a_permanent_stretch():
    """Force 1.2 σy A: ε = σ/E + (σ − σy)(1/E_t − 1/E); after unloading only the plastic
    part remains."""
    L = 100.0
    project = _bar(L, plasticity=True)
    end = _face(project, (5, 5, L))
    sigma = 1.2 * SY
    project.loads = [Load(LoadKind.FORCE, [end], vector=(0, 0, sigma * AREA))]
    mesh = generate_mesh(project, "coarse")
    r = solve_nonlinear(project, mesh, {0: STEEL})
    plastic = (sigma - SY) * (1 / ET - 1 / E)
    top = mesh.face_nodes([end])
    assert r.displacement[top, 2] == pytest.approx((sigma / E + plastic) * L, rel=1e-4)
    assert r.nonlinear.permanent[top, 2] == pytest.approx(plastic * L, rel=1e-3)


def test_load_beyond_the_plastic_limit_is_reported():
    """Nearly no hardening: a bar carries little more than σy A. Asked for 1.1 σy A the
    run stops where the plastic strain runs away (50 %), i.e. at σ = σy + H · 0.5."""
    project = _bar(plasticity=True)
    end = _face(project, (5, 5, 100))
    project.loads = [Load(LoadKind.FORCE, [end], vector=(0, 0, 1.1 * SY * AREA))]
    mesh = generate_mesh(project, "coarse")
    soft = replace(STEEL, tangent_modulus=1e-4 * E)
    r = solve_nonlinear(project, mesh, {0: soft})
    assert not r.nonlinear.completed and r.nonlinear.limit_reason
    H = E * soft.tangent_modulus / (E - soft.tangent_modulus)
    assert r.nonlinear.load_factor == pytest.approx((SY + 0.5 * H) / (1.1 * SY), rel=0.01)


def test_large_deflection_of_a_cantilever_matches_the_elastica():
    """Slender cantilever, tip load with P L² / (E I) = 1 (dead load): the elastica gives
    tip deflections 0.3017 L down and 0.0564 L inwards; the linear answer is L/3."""
    L, b = 200.0, 5.0
    project = _project(_box(0, 0, 0, L, b, b), large_deformation=True)
    root, tip = _face(project, (0, b / 2, b / 2)), _face(project, (L, b / 2, b / 2))
    project.fixtures = [Fixture(FixtureKind.FIXED, [root])]
    inertia = b**4 / 12
    P = E * inertia / L**2
    project.loads = [Load(LoadKind.REMOTE, [tip], vector=(0, 0, -P), point=(L, b / 2, b / 2))]
    mesh = generate_mesh(project, "normal")
    r = solve_nonlinear(project, mesh, {0: STEEL})
    assert r.nonlinear.completed
    centre = mesh.face_nodes([tip])
    down = -r.displacement[centre, 2].mean() / L
    inwards = -r.displacement[centre, 0].mean() / L
    assert down == pytest.approx(0.3017, rel=0.03)
    assert inwards == pytest.approx(0.0564, rel=0.05)
    linear = solve_static(project, mesh, {0: STEEL})
    assert -linear.displacement[centre, 2].mean() / L == pytest.approx(1 / 3, rel=0.03)
    # a rigid rotation produces no stress: the Cauchy stress stays that of bending only
    assert np.abs(r.stress).max() < 2 * P * L * (b / 2) / inertia


def test_ties_pins_and_supports_below_yield_match_the_linear_run():
    """Bonded parts (eliminated ties), a pinned hole with rollers (equations, multiplier
    reactions) and an elastic support: under a small load the nonlinear run is the linear
    one, including the reactions."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Dir

    from kstruct.geometry.samples import _hole
    from kstruct.model import Connection, ConnectionKind

    lower, upper = _box(0, 0, 0, 10, 10, 50), _box(0, 0, 50, 10, 10, 50)
    project = Project(
        parts=[Part(0, "lower", lower, "s"), Part(1, "upper", upper, "s")], plasticity=True
    )

    def face(pid, point):
        return FaceRef(pid, face_at(project.parts[pid].shape, point))

    project.fixtures = [Fixture(FixtureKind.FIXED, [face(0, (5, 5, 0))])]
    project.connections = [
        Connection(ConnectionKind.BONDED, [face(0, (5, 5, 50))], [face(1, (5, 5, 50))])
    ]
    project.loads = [Load(LoadKind.FORCE, [face(1, (5, 5, 100))], vector=(20.0, 0, 500.0))]
    plate = BRepAlgoAPI_Cut(_box(0, 0, 0, 100, 20, 5), _hole(10, 10, -1, gp_Dir(0, 0, 1), 4, 7))
    pinned = _project(plate.Shape(), plasticity=True, large_deformation=True)
    pinned.fixtures = [
        Fixture(FixtureKind.PIN, [_face(pinned, (14, 10, 2.5))]),
        Fixture(FixtureKind.ROLLER, [_face(pinned, (50, 0, 2.5))]),
        Fixture(FixtureKind.ELASTIC, [_face(pinned, (50, 10, 0))], stiffness=1e5),
    ]
    pinned.loads = [Load(LoadKind.FORCE, [_face(pinned, (100, 10, 2.5))], vector=(300.0, 0, -5))]
    for model, mats in ((project, {0: STEEL, 1: STEEL}), (pinned, {0: STEEL})):
        mesh = generate_mesh(model, "coarse")
        linear = solve_static(model, mesh, mats)
        r = solve_nonlinear(model, mesh, mats)
        assert r.nonlinear.completed and not r.nonlinear.yielded
        scale = np.abs(linear.displacement).max()
        # (large deformation adds a geometric effect of order u/L, here ~1e-5)
        assert np.abs(r.displacement - linear.displacement).max() < 1e-3 * scale
        for a, b in zip(r.reactions, linear.reactions, strict=True):
            assert a == pytest.approx(b, rel=2e-3, abs=1e-2)
