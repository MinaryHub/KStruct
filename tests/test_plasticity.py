"""J2 return mapping: uniaxial response and the consistent tangent."""

import numpy as np
import pytest

from kstruct.fem.plasticity import PointMaterial, elasticity, return_map, tangent_modulus
from kstruct.model import Material

E, NU, SY = 200_000.0, 0.3, 250.0


def _mat(n=1, H=2_000.0):
    return PointMaterial(np.full(n, E), np.full(n, NU), np.full(n, SY), np.full(n, H))


def test_elastic_below_yield():
    strain = np.array([[1e-4, -3e-5, -3e-5, 2e-5, 0, 0]])
    stress, D, ep, a = return_map(strain, np.zeros((1, 6)), np.zeros(1), _mat())
    assert stress[0] == pytest.approx(elasticity(np.array([E]), np.array([NU]))[0] @ strain[0])
    assert a[0] == 0 and not ep.any()


def test_uniaxial_plastic_flow_follows_the_bilinear_curve():
    """Uniaxial stress state (lateral strains solved so σ22 = σ33 = 0): beyond yield
    σ = σy + E_t (ε − εy) with E_t = E H / (E + H)."""
    H = 2_000.0
    Et = E * H / (E + H)
    mat = _mat(H=H)
    for eps in (0.002, 0.01, 0.05):
        lateral = -NU * eps
        for _ in range(50):  # Newton on the lateral strain for zero lateral stress
            strain = np.array([[eps, lateral, lateral, 0, 0, 0]])
            stress, D, _, _ = return_map(strain, np.zeros((1, 6)), np.zeros(1), mat)
            k = D[0, 1, 1] + D[0, 1, 2]
            lateral -= stress[0, 1] / k
        expected = SY + Et * (eps - SY / E) if eps > SY / E else E * eps
        assert stress[0, 0] == pytest.approx(expected, rel=1e-8)
        assert abs(stress[0, 1]) < 1e-6


def test_consistent_tangent_matches_finite_differences():
    rng = np.random.default_rng(1)
    mat = _mat(n=4)
    strain = rng.normal(scale=3e-3, size=(4, 6))
    ep = rng.normal(scale=5e-4, size=(4, 6))
    ep[:, :3] -= ep[:, :3].mean(axis=1, keepdims=True)  # plastic flow is isochoric
    alpha = np.abs(rng.normal(scale=1e-3, size=4))
    stress, D, _, a = return_map(strain, ep, alpha, mat)
    assert np.all(a > alpha)  # all four yield further
    h = 1e-8
    for j in range(6):
        bumped = strain.copy()
        bumped[:, j] += h
        s2, _, _, _ = return_map(bumped, ep, alpha, mat, tangent=False)
        assert (s2 - stress) / h == pytest.approx(D[:, :, j], rel=1e-4, abs=1e-2)


def test_tangent_modulus_default_and_override():
    from dataclasses import replace

    steel = Material("s", "s", "steel", 210_000.0, 0.3, 7.85e-9, 235.0, 360.0)
    assert tangent_modulus(steel) == pytest.approx(125.0 / (0.10 - 235.0 / 210_000.0))
    assert tangent_modulus(replace(steel, tangent_modulus=1_000.0)) == 1_000.0
    flat = replace(steel, tensile_strength=235.0)
    assert tangent_modulus(flat) == pytest.approx(21.0)  # E/10⁴ floor


def test_flow_stress_levels_off_at_the_tensile_strength():
    """Trilinear curve: beyond R_m the stress only creeps up with the tiny H_after, and
    the tangent stays consistent on that branch too."""
    H, Rm, H2 = 2_000.0, 400.0, 20.0
    mat = PointMaterial(
        np.full(1, E), np.full(1, NU), np.full(1, SY), np.full(1, H), np.full(1, Rm), np.full(1, H2)
    )
    Et2 = E * H2 / (E + H2)
    eps = 0.2  # far beyond the strain at R_m
    lateral = -0.5 * eps
    for _ in range(80):
        strain = np.array([[eps, lateral, lateral, 0, 0, 0]])
        stress, D, _, alpha = return_map(strain, np.zeros((1, 6)), np.zeros(1), mat)
        lateral -= stress[0, 1] / (D[0, 1, 1] + D[0, 1, 2])
    alpha_u = (Rm - SY) / H
    eps_u = Rm / E + alpha_u  # total strain where R_m is reached
    assert stress[0, 0] == pytest.approx(Rm + Et2 * (eps - eps_u), rel=1e-6)
    h = 1e-8
    bumped = strain.copy()
    bumped[0, 0] += h
    s2, _, _, _ = return_map(bumped, np.zeros((1, 6)), np.zeros(1), mat, tangent=False)
    assert (s2 - stress)[0] / h == pytest.approx(D[0, :, 0], rel=1e-4, abs=1e-2)
