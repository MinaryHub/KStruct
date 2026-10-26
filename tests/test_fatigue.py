from types import SimpleNamespace

import numpy as np
import pytest

from kstruct.fem.fatigue import (
    HIGH_CYCLES,
    INFINITE_LIFE,
    MARIN_FACTOR,
    LoadCycle,
    fatigue_check,
)
from kstruct.model import Material

STEEL = Material("s", "s", "steel", 210_000, 0.3, 7.85e-9, 250, 400, fatigue_limit=200)
SE = MARIN_FACTOR * 200


def _result(sigma_zz: list[float]):
    """Fake static result: uniaxial stress sigma_zz at each node, one part."""
    stress = np.zeros((len(sigma_zz), 6))
    stress[:, 2] = sigma_zz
    vm = np.abs(stress[:, 2])
    mesh = SimpleNamespace(n_nodes=len(sigma_zz), part_nodes=lambda pid: np.arange(len(sigma_zz)))
    return SimpleNamespace(stress=stress, von_mises=vm, mesh=mesh, materials={0: STEEL})


def test_reversed_at_endurance_limit_has_unit_safety():
    r = fatigue_check(_result([SE, SE / 2]), LoadCycle.REVERSED)
    assert r.safety_factor == pytest.approx([1.0, 2.0])
    assert r.life[1] == INFINITE_LIFE


def test_pulsating_uses_goodman():
    peak = 200.0
    r = fatigue_check(_result([peak]), LoadCycle.PULSATING)
    expected = 1 / ((peak / 2) / SE + (peak / 2) / 400)
    assert r.safety_factor[0] == pytest.approx(expected)


def test_compressive_mean_is_not_credited():
    tension = fatigue_check(_result([SE]), LoadCycle.REVERSED).safety_factor[0]
    pulsating_compression = fatigue_check(_result([-2 * SE]), LoadCycle.PULSATING)
    # amplitude SE, compressive mean ignored -> same as fully reversed at SE
    assert pulsating_compression.safety_factor[0] == pytest.approx(tension)


def test_basquin_life_endpoints():
    r = fatigue_check(_result([0.9 * 400, SE * 1.0001]), LoadCycle.REVERSED)
    assert r.life[0] == pytest.approx(1e3, rel=1e-6)
    assert r.life[1] == pytest.approx(HIGH_CYCLES, rel=1e-2)


def test_marin_factors_follow_surface_strength_and_reliability():
    from kstruct.fem.fatigue import FatigueSettings

    machined = FatigueSettings("machined", "90").factors(400.0)
    assert machined["ka"] == pytest.approx(4.51 * 400**-0.265, rel=1e-6)
    assert machined["kc"] == pytest.approx(0.897) and machined["kb"] == pytest.approx(0.85)
    assert machined["total"] == pytest.approx(0.70, abs=0.02)  # the old constant, roughly
    forged = FatigueSettings("as_forged", "99").factors(400.0)
    assert forged["total"] < 0.75 * machined["total"]
    ground = FatigueSettings("ground", "50").factors(1200.0)
    assert ground["ka"] < 1.0 and ground["kc"] == 1.0
    assert FatigueSettings("machined", "90").factors(100.0)["ka"] == 1.0  # capped


def test_settings_change_the_endurance_limit():
    from kstruct.fem.fatigue import FatigueSettings

    base = fatigue_check(_result([SE]), LoadCycle.REVERSED)
    rough = fatigue_check(_result([SE]), LoadCycle.REVERSED, FatigueSettings("as_forged", "99"))
    assert rough.factors[0]["total"] < MARIN_FACTOR
    assert rough.safety_factor[0] < base.safety_factor[0]
    assert rough.endurance[0] == pytest.approx(200 * rough.factors[0]["total"])


def test_fatigue_settings_round_trip(tmp_path):
    from kstruct.geometry.samples import _box
    from kstruct.model import Part, Project
    from kstruct.project_io import load_project, save_project

    project = Project(parts=[Part(0, "p", _box(0, 0, 0, 1, 1, 1))])
    project.load_cycle = "pulsating"
    project.fatigue_surface = "hot_rolled"
    project.fatigue_reliability = "99"
    loaded = load_project(save_project(project, tmp_path / "f.kst"))
    assert (loaded.fatigue_surface, loaded.fatigue_reliability) == ("hot_rolled", "99")
