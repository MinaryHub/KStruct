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
