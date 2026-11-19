"""Heat transfer over time: backward Euler against the Fourier-series solution."""

import math

import numpy as np
import pytest

from kstruct.fem import SolverError, generate_mesh
from kstruct.fem.thermal import solve_thermal
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Load, LoadKind, Material, Part, Project

K, RHO, CP = 50.0, 7850.0, 460.0  # W/(m·K), kg/m³, J/(kg·K)
STEEL = Material(
    "s", "s", "steel", 210_000.0, 0.3, RHO * 1e-12, 250.0, 400.0,
    thermal_conductivity=K, specific_heat=CP * 1e6,
)  # fmt: skip
ALPHA = K / (RHO * CP) * 1e6  # diffusivity, mm²/s


def _bar(length=100.0) -> tuple[Project, object]:
    shape = _box(0, 0, 0, length, 10, 10)
    return Project(parts=[Part(0, "bar", shape, "s")]), shape


def _series(x, t, length, T0, T1, terms=200):
    """Bar at T0, the end x = 0 held at T1 from t = 0, everything else insulated."""
    theta = 0.0
    for n in range(terms):
        lam = (2 * n + 1) * math.pi / (2 * length)
        theta += 4 / ((2 * n + 1) * math.pi) * math.sin(lam * x) * math.exp(-ALPHA * lam**2 * t)
    return T1 + (T0 - T1) * theta


def test_bar_heated_from_one_end_follows_the_series_solution():
    L, T0, T1 = 100.0, 20.0, 100.0
    project, shape = _bar(L)
    project.thermal_duration = 300.0
    project.initial_temperature = T0
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [FaceRef(0, face_at(shape, (0, 5, 5)))], magnitude=T1)
    ]
    mesh = generate_mesh(project, "coarse")
    r = solve_thermal(project, mesh, {0: STEEL})
    assert r.transient and r.times[0] == 0.0 and r.times[-1] == pytest.approx(300.0)
    assert r.history[0] == pytest.approx(T0)
    x = mesh.nodes[:, 0]
    for i in (len(r.times) // 2, len(r.times) - 1):
        t = r.times[i]
        expected = np.array([_series(xi, t, L, T0, T1) for xi in x])
        # first-order time stepping: within 2 % of the temperature rise at every node
        assert np.abs(r.history[i] - expected).max() < 0.02 * (T1 - T0)
    assert r.temperature == pytest.approx(r.history[-1])
    peak, when = r.peak()
    assert peak == pytest.approx(T1) and when > 0


def test_long_transient_ends_at_the_steady_state():
    """Hot face + convection: after many time constants the transient result equals the
    steady-state solve."""
    project, shape = _bar()
    project.loads = [
        Load(LoadKind.FACE_TEMPERATURE, [FaceRef(0, face_at(shape, (0, 5, 5)))], magnitude=150.0),
        Load(LoadKind.CONVECTION, [FaceRef(0, face_at(shape, (50, 5, 10)))], magnitude=25.0),
    ]
    mesh = generate_mesh(project, "coarse")
    steady = solve_thermal(project, mesh, {0: STEEL})
    project.thermal_duration = 60_000.0
    late = solve_thermal(project, mesh, {0: STEEL})
    assert late.temperature == pytest.approx(steady.temperature, abs=0.05)


def test_transient_needs_heat_capacity_but_no_reference():
    """Heat input alone has no steady state but warms the part over time: ΔT = Q t / (m c)
    on average. A material without specific heat is reported."""
    from dataclasses import replace

    project, shape = _bar()
    q = 5.0  # W
    project.loads = [Load(LoadKind.HEAT_FLOW, [FaceRef(0, face_at(shape, (0, 5, 5)))], magnitude=q)]
    mesh = generate_mesh(project, "coarse")
    with pytest.raises(SolverError):
        solve_thermal(project, mesh, {0: STEEL})  # steady: no temperature reference
    project.thermal_duration = 120.0
    r = solve_thermal(project, mesh, {0: STEEL})
    mass = RHO * (100 * 10 * 10) * 1e-9  # kg
    expected_rise = q * 120.0 / (mass * CP)
    from kstruct.fem.tet10 import mass_lumped_volume

    vol = mass_lumped_volume(mesh.nodes[mesh.tets])
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, mesh.tets.ravel(), vol.ravel())
    mean_rise = (w * (r.temperature - 20.0)).sum() / w.sum()
    assert mean_rise == pytest.approx(expected_rise, rel=1e-6)  # energy is conserved
    with pytest.raises(SolverError, match="비열"):
        solve_thermal(project, mesh, {0: replace(STEEL, specific_heat=0.0)})
