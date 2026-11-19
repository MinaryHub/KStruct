"""Modal and thermal verification against closed-form solutions."""

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh, solve_static
from kstruct.fem.modal import solve_modal
from kstruct.fem.thermal import solve_thermal
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Material, Part, Project

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
    thermal_conductivity=50.0,
)


def _bar(length=200.0, side=10.0) -> Project:
    return Project(parts=[Part(0, "bar", _box(0, 0, 0, length, side, side))])


def _face(project, point) -> FaceRef:
    return FaceRef(0, face_at(project.parts[0].shape, point))


def test_cantilever_first_bending_frequency():
    L, b = 200.0, 10.0
    project = _bar(L, b)
    project.fixtures = [Fixture(FixtureKind.FIXED, [_face(project, (0, 5, 5))])]
    mesh = generate_mesh(project, "normal")
    r = solve_modal(project, mesh, {0: STEEL}, n_modes=4)

    E, rho, area, inertia = STEEL.youngs_modulus, STEEL.density, b * b, b**4 / 12
    f1 = 1.875104**2 / (2 * np.pi) * np.sqrt(E * inertia / (rho * area * L**4))
    # square section: the first two modes are the same bending mode in y and z
    assert r.frequencies[0] == pytest.approx(f1, rel=0.03)
    assert r.frequencies[1] == pytest.approx(f1, rel=0.03)
    assert r.frequencies[2] > 3 * f1
    # first mode moves sideways (y or z), not along the beam axis
    assert r.participation[0, 0] < 0.01
    assert r.near(f1) == [0, 1]


def test_free_free_skips_rigid_modes():
    project = _bar(100.0, 10.0)
    mesh = generate_mesh(project, "coarse")
    r = solve_modal(project, mesh, {0: STEEL}, n_modes=2)
    assert r.rigid_modes == 6
    assert r.frequencies[0] > 1000  # first free-free bending mode of a short stiff bar


def test_linear_conduction():
    """Bar with 100 °C at one end, 0 °C at the other: linear profile, Q = k A ΔT / L."""
    L = 100.0
    project = _bar(L)
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (0, 5, 5))], magnitude=100.0),
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (L, 5, 5))], magnitude=0.0),
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_thermal(project, mesh, {0: STEEL})
    expected = 100.0 * (1 - mesh.nodes[:, 0] / L)
    assert r.temperature == pytest.approx(expected, abs=1e-8)
    q = 50.0 * 100e-6 * 100.0 / 0.1  # k A dT / L in W (A, L in metres)
    # what enters at the hot end leaves at the cold end (reported separately, not netted)
    assert r.heat_in == pytest.approx(q, rel=1e-6)
    assert r.heat_out == pytest.approx(q, rel=1e-6)
    assert r.heat_flux[:, 0] == pytest.approx(q / 100e-6, rel=1e-6)  # W/m²


def test_heat_input_with_convection():
    """1-D: Q enters one end, leaves by convection at the other. T_end = T_inf + Q/(hA)."""
    L, Q, h = 100.0, 2.0, 500.0
    project = _bar(L)
    project.loads = [
        Load(LoadKind.HEAT_FLOW, [_face(project, (0, 5, 5))], magnitude=Q),
        Load(LoadKind.CONVECTION, [_face(project, (L, 5, 5))], magnitude=h, ambient=25.0),
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_thermal(project, mesh, {0: STEEL})
    area = 100e-6  # m²
    cold = 25.0 + Q / (h * area)
    hot = cold + Q * (L / 1000) / (50.0 * area)
    end = np.isclose(mesh.nodes[:, 0], L)
    start = np.isclose(mesh.nodes[:, 0], 0)
    assert r.temperature[end] == pytest.approx(cold, rel=1e-6)
    assert r.temperature[start] == pytest.approx(hot, rel=1e-6)
    assert r.heat_in == pytest.approx(Q)
    assert r.heat_out == pytest.approx(Q, rel=1e-6)


def test_thermal_needs_a_reference():
    project = _bar()
    project.loads = [Load(LoadKind.HEAT_FLOW, [_face(project, (0, 5, 5))], magnitude=1.0)]
    with pytest.raises(SolverError):
        solve_thermal(project, generate_mesh(project, "coarse"), {0: STEEL})


def test_computed_temperature_drives_thermal_stress():
    """Both ends held at 120 °C -> uniform field, 100 K above the stress-free 20 °C: the
    roller-held bar grows freely by alpha * 100 * L without stress."""
    L = 100.0
    project = Project(
        parts=[Part(0, "bar", _box(0, 0, 0, 10, 10, L))],
        reference_temperature=20.0,
    )
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, 50))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, 50))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0))]),
    ]
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (5, 5, 0))], magnitude=120.0),
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (5, 5, L))], magnitude=120.0),
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_static(project, mesh, {0: STEEL})
    top = np.isclose(mesh.nodes[:, 2], L)
    assert r.displacement[top, 2] == pytest.approx(12e-6 * 100.0 * L, rel=1e-6)
    assert r.von_mises.max() < 1e-6
    assert r.thermal is not None


def test_restrained_bar_thermal_stress():
    """Rollers on both ends stop axial growth: sigma_z = -E alpha dT exactly."""
    L = 100.0
    project = Project(parts=[Part(0, "bar", _box(0, 0, 0, 10, 10, L))])
    project.fixtures = [
        Fixture(FixtureKind.ROLLER, [_face(project, (0, 5, 50))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 0, 50))]),
        Fixture(FixtureKind.ROLLER, [_face(project, (5, 5, 0)), _face(project, (5, 5, L))]),
    ]
    project.loads = [Load(LoadKind.TEMPERATURE, magnitude=100.0)]
    r = solve_static(project, generate_mesh(project, "coarse"), {0: STEEL})
    assert r.stress[:, 2] == pytest.approx(-210_000 * 12e-6 * 100, rel=1e-6)
    assert np.abs(r.stress[:, :2]).max() < 1e-6


def test_radiation_end_matches_the_stefan_boltzmann_balance():
    """Bar held at 500 °C at one end, radiating (ε = 0.8) to 20 °C surroundings at the
    other: k (T1 − T2) / L = ε σ (T2⁴ − T∞⁴) fixes the free end temperature."""
    from scipy.optimize import brentq

    L, T1, eps, amb = 100.0, 500.0, 0.8, 20.0
    project = _bar(L)
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (0, 5, 5))], magnitude=T1),
        Load(LoadKind.RADIATION, [_face(project, (L, 5, 5))], magnitude=eps, ambient=amb),
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_thermal(project, mesh, {0: STEEL})
    sigma = 5.670374419e-8

    def balance(T2):
        return 50.0 * (T1 - T2) / (L / 1000) - eps * sigma * (
            (T2 + 273.15) ** 4 - (amb + 273.15) ** 4
        )

    T2 = brentq(balance, amb, T1)
    end = np.isclose(mesh.nodes[:, 0], L)
    assert r.temperature[end] == pytest.approx(T2, abs=0.05)
    assert r.iterations > 1
    q = 50.0 * (T1 - T2) / (L / 1000) * 100e-6  # W through the bar
    # in at the hot end, out by radiation
    assert r.heat_in == pytest.approx(q, rel=1e-3)
    assert r.heat_out == pytest.approx(q, rel=1e-3)


def test_temperature_dependent_conductivity_bends_the_profile():
    """k = 50 − 0.05 T (W/mK): with 500 °C and 0 °C at the ends the Kirchhoff transform
    gives 50 T − 0.025 T² linear in x, so the midpoint sits at 209.4 °C, not 250."""
    from dataclasses import replace

    L = 100.0
    varying = replace(STEEL, conductivity_table=((0.0, 50.0), (500.0, 25.0)))
    project = _bar(L)
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (0, 5, 5))], magnitude=500.0),
        Load(LoadKind.FACE_TEMPERATURE, [_face(project, (L, 5, 5))], magnitude=0.0),
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_thermal(project, mesh, {0: varying})
    x = mesh.nodes[:, 0]
    phi = 50.0 * r.temperature - 0.025 * r.temperature**2
    assert phi == pytest.approx(18750.0 * (1 - x / L), abs=60.0)  # 0.3 % of the range
    mid = np.isclose(x, L / 2)
    assert r.temperature[mid] == pytest.approx(209.43, abs=1.0)
    assert r.iterations > 1
