"""Torque, bearing and centrifugal loads against closed-form solutions."""

import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.gp import gp_Dir

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.loads import bearing_forces
from kstruct.geometry.face_info import common_axis, face_at
from kstruct.geometry.samples import _box, _hole
from kstruct.geometry.tessellate import faces_of
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Material, Part, Project
from kstruct.project_io import load_project, save_project
from kstruct.report.describe import describe_load

STEEL = Material("s", "s", "steel", 210_000.0, 0.3, 7.85e-9, 250.0, 400.0)
G = STEEL.youngs_modulus / (2 * (1 + STEEL.poisson_ratio))


def _shaft(R=10.0, L=100.0):
    shape = _hole(0, 0, 0, gp_Dir(0, 0, 1), R, L)  # a solid cylinder along z
    project = Project(parts=[Part(0, "shaft", shape)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 0, 0)))])]
    return project, shape


def test_torsion_of_a_circular_shaft_matches_theory():
    R, L, T = 10.0, 100.0, 50_000.0  # 50 N·m
    project, shape = _shaft(R, L)
    tip = FaceRef(0, face_at(shape, (0, 0, L)))
    project.loads = [Load(LoadKind.TORQUE, [tip], vector=(0, 0, 1), magnitude=T, point=(0, 0, 0))]
    mesh = generate_mesh(project, "fine")
    r = solve_static(project, mesh, {0: STEEL})

    J = np.pi * R**4 / 2
    tau_max = T * R / J
    # von Mises of pure shear is sqrt(3)·tau; take the outer surface at mid-length
    outer = np.isclose(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1]), R, atol=0.05 * R)
    mid = outer & (np.abs(mesh.nodes[:, 2] - L / 2) < 0.1 * L)
    assert r.von_mises[mid].mean() == pytest.approx(np.sqrt(3) * tau_max, rel=0.04)
    # twist angle at the tip: theta = T L / (G J); tangential displacement = theta * r
    theta = T * L / (G * J)
    tip_nodes = mesh.face_nodes([tip])
    radial = mesh.nodes[tip_nodes, :2]
    u = r.displacement[tip_nodes, :2]
    rr = np.linalg.norm(radial, axis=1)
    tangential = (radial[:, 0] * u[:, 1] - radial[:, 1] * u[:, 0]) / rr
    far = rr > 0.5 * R
    assert tangential[far] / rr[far] == pytest.approx(theta, rel=0.03)
    # the fixture reacts with no net force (a pure couple)
    assert np.abs(r.reactions[0]).max() < 1e-6 * T / R


def test_torque_sign_follows_the_right_hand_rule():
    project, shape = _shaft()
    tip = FaceRef(0, face_at(shape, (0, 0, 100)))
    project.loads = [
        Load(LoadKind.TORQUE, [tip], vector=(0, 0, 1), magnitude=-1000.0, point=(0, 0, 0))
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_static(project, mesh, {0: STEEL})
    tip_nodes = mesh.face_nodes([tip])
    radial = mesh.nodes[tip_nodes, :2]
    u = r.displacement[tip_nodes, :2]
    rotation = radial[:, 0] * u[:, 1] - radial[:, 1] * u[:, 0]
    assert rotation[np.linalg.norm(radial, axis=1) > 5].mean() < 0  # clockwise about +z


def test_torque_axis_comes_from_a_cylindrical_face():
    project, shape = _shaft()
    side = FaceRef(0, face_at(shape, (10, 0, 50)))
    axis = common_axis([faces_of(shape)[side.face_index]])
    assert axis is not None
    assert abs(abs(axis[1][2]) - 1) < 1e-9
    # no point/vector stored: the solver takes the face's own axis
    project.loads = [Load(LoadKind.TORQUE, [side], magnitude=1000.0)]
    mesh = generate_mesh(project, "coarse")
    r = solve_static(project, mesh, {0: STEEL})
    assert r.von_mises.max() > 0
    # a flat face without an axis cannot carry a torque
    project.loads = [
        Load(LoadKind.TORQUE, [FaceRef(0, face_at(shape, (0, 0, 100)))], magnitude=1.0)
    ]
    with pytest.raises(SolverError):
        solve_static(project, mesh, {0: STEEL})


def _plate_with_hole():
    shape = _box(0, 0, 0, 80, 60, 10)
    shape = BRepAlgoAPI_Cut(shape, _hole(40, 30, -1, gp_Dir(0, 0, 1), 8, 12)).Shape()
    project = Project(parts=[Part(0, "plate", shape)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 5)))])]
    hole = FaceRef(0, face_at(shape, (48, 30, 5)))
    return project, shape, hole


def test_bearing_load_resultant_and_loaded_half():
    project, shape, hole = _plate_with_hole()
    F = (2000.0, 0.0, 0.0)
    project.loads = [Load(LoadKind.BEARING, [hole], vector=F)]
    mesh = generate_mesh(project, "coarse")
    f = bearing_forces(project, mesh, project.loads[0])
    assert f.sum(axis=0) == pytest.approx(F, abs=1e-6 * 2000)
    # only the wall facing +x is pushed (nodes with x > 40 on the hole)
    hole_nodes = mesh.face_nodes([hole])
    loaded = hole_nodes[np.abs(f[hole_nodes]).sum(axis=1) > 1e-9]
    assert np.all(mesh.nodes[loaded, 0] > 40 - 3.0)  # triangles straddling the boundary
    far = hole_nodes[mesh.nodes[hole_nodes, 0] < 40 - 3.0]
    assert len(far) > 0 and np.abs(f[far]).max() == 0.0
    r = solve_static(project, mesh, {0: STEEL})
    assert r.reactions[0] == pytest.approx([-2000.0, 0.0, 0.0], abs=1e-6)
    # not a hole: refused
    project.loads = [Load(LoadKind.BEARING, [FaceRef(0, face_at(shape, (40, 30, 10)))], vector=F)]
    with pytest.raises(SolverError):
        solve_static(project, mesh, {0: STEEL})


def test_centrifugal_rod_about_its_end():
    """Rod of length L rotating about an axis through its fixed end: total centrifugal
    force ρAω²L²/2 and axial stress σ(x) = ρω²(L² − x²)/2."""
    L, b = 200.0, 10.0
    shape = _box(0, 0, 0, L, b, b)
    project = Project(parts=[Part(0, "rod", shape)])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 5, 5)))])]
    omega = 3000 * 2 * np.pi / 60
    project.loads = [Load(LoadKind.CENTRIFUGAL, vector=(0, 0, 1), magnitude=omega, point=(0, 5, 5))]
    mesh = generate_mesh(project, "normal")
    r = solve_static(project, mesh, {0: STEEL})
    rho = STEEL.density
    total = rho * b * b * omega**2 * L**2 / 2
    assert r.reactions[0] == pytest.approx([-total, 0, 0], abs=1e-6 * total)  # holds it back
    mid = np.abs(mesh.nodes[:, 0] - L / 2) < 2.0
    assert r.stress[mid, 0].mean() == pytest.approx(
        rho * omega**2 * (L**2 - (L / 2) ** 2) / 2, rel=0.03
    )


def test_new_loads_round_trip_and_describe(tmp_path):
    project, shape, hole = _plate_with_hole()
    project.loads = [
        Load(LoadKind.TORQUE, [hole], vector=(0, 0, 1), magnitude=12_000.0, point=(40, 30, 0)),
        Load(LoadKind.BEARING, [hole], vector=(0, -500.0, 0)),
        Load(LoadKind.CENTRIFUGAL, vector=(0, 0, 1), magnitude=314.159, point=(40, 30, 0)),
    ]
    loaded = load_project(save_project(project, tmp_path / "l.kst"))
    assert loaded.loads == project.loads
    texts = [describe_load(loaded, ld) for ld in loaded.loads]
    assert texts[0].startswith("토크 12 N·m, Z 축")
    assert texts[1].startswith("베어링 하중 500.0 N")
    assert texts[2].startswith("원심력 3,000 rpm, Z 축")
