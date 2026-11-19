"""Elastic–plastic material: von Mises (J2) yield with linear isotropic hardening.

Vectorised over integration points. Voigt order xx, yy, zz, xy, yz, zx with engineering
shear strains (as in fem/tet10.py); stresses carry the tensor shear components.

The uniaxial curve is trilinear: E up to the yield strength, then the tangent modulus
E_t up to the tensile strength R_m, then (almost) flat — a bilinear curve that kept
hardening beyond R_m overestimated collapse loads by a third on the sample bracket. The hardening modulus of the J2 model follows as H = E·E_t / (E − E_t). Without a
tangent modulus in the material data it is estimated from the tensile strength being
reached at 10 % strain (a typical uniform elongation of structural metals):
E_t = (R_m − R_e) / (0.10 − R_e/E), at least E/10⁴ so the tangent stays positive.

Return mapping (radial return) and the consistent tangent follow de Souza Neto, Perić &
Owen, *Computational Methods for Plasticity*, box 7.4 / eq. 7.120.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kstruct.model import Material

UNIFORM_ELONGATION = 0.10  # strain at which R_m is reached, when no E_t is given
MIN_TANGENT = 1e-4  # E_t is at least this fraction of E
_NORMAL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
# 2G·I_dev for engineering strains: deviatoric part of the elasticity matrix per unit G
_DEV = np.zeros((6, 6))
_DEV[:3, :3] = 2 * (np.eye(3) - 1 / 3)  # 2 (δij − 1/3)
_DEV[3:, 3:] = np.eye(3)  # shear: σ_ij = G γ_ij


def tangent_modulus(m: Material) -> float:
    """Slope E_t of the bilinear stress–strain curve beyond yield (MPa)."""
    E = m.youngs_modulus
    if getattr(m, "tangent_modulus", None):
        return float(min(max(m.tangent_modulus, MIN_TANGENT * E), 0.99 * E))
    room = UNIFORM_ELONGATION - m.yield_strength / E
    if m.tensile_strength > m.yield_strength and room > 0:
        Et = (m.tensile_strength - m.yield_strength) / room
    else:
        Et = MIN_TANGENT * E
    return float(min(max(Et, MIN_TANGENT * E), 0.99 * E))


@dataclass
class PointMaterial:
    """Per integration point material constants (arrays of length n)."""

    E: np.ndarray
    nu: np.ndarray
    yield_stress: np.ndarray  # inf: stays elastic (brittle materials)
    H: np.ndarray  # isotropic hardening modulus up to the tensile strength
    ultimate: np.ndarray | None = None  # flow stress cap R_m (inf: none)
    H_after: np.ndarray | None = None  # hardening modulus beyond R_m (tiny, keeps K > 0)

    def __post_init__(self) -> None:
        if self.ultimate is None:
            self.ultimate = np.full(len(self.E), np.inf)
        if self.H_after is None:
            self.H_after = self.H.copy()

    def subset(self, sl) -> PointMaterial:
        return PointMaterial(
            self.E[sl], self.nu[sl], self.yield_stress[sl], self.H[sl],
            self.ultimate[sl], self.H_after[sl],
        )  # fmt: skip


def point_material(materials: list[Material]) -> PointMaterial:
    E = np.array([m.youngs_modulus for m in materials], dtype=float)
    Et = np.array([tangent_modulus(m) for m in materials])
    flat = MIN_TANGENT * E
    return PointMaterial(
        E=E,
        nu=np.array([m.poisson_ratio for m in materials], dtype=float),
        # brittle materials crack rather than yield: they stay linear elastic here and
        # are judged on their strength as in the linear analysis
        yield_stress=np.array(
            [m.yield_strength if m.ductile else np.inf for m in materials], dtype=float
        ),
        H=E * Et / (E - Et),
        ultimate=np.array(
            [
                m.tensile_strength
                if m.ductile and m.tensile_strength > m.yield_strength
                else np.inf
                for m in materials
            ],
            dtype=float,
        ),
        H_after=E * flat / (E - flat),
    )


def elasticity(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """(n, 6, 6) isotropic elasticity matrices for engineering shear strains."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    D = np.zeros((len(E), 6, 6))
    D[:, :3, :3] = lam[:, None, None]
    D[:, np.arange(3), np.arange(3)] += 2 * G[:, None]
    D[:, np.arange(3, 6), np.arange(3, 6)] = G[:, None]
    return D


def _tensor_norm(s: np.ndarray) -> np.ndarray:
    """||s|| for Voigt stress rows (tensor shear components)."""
    return np.sqrt((s[:, :3] ** 2).sum(axis=1) + 2 * (s[:, 3:] ** 2).sum(axis=1))


def return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    mat: PointMaterial,
    tangent: bool = True,
):
    """Stress update for total strain `strain` (n, 6) from the committed plastic strain
    (n, 6, engineering shear) and equivalent plastic strain alpha (n,).

    Returns (stress (n, 6), D (n, 6, 6) consistent tangent or None, new plastic strain,
    new alpha). Nothing is modified in place.
    """
    D = elasticity(mat.E, mat.nu)
    G = mat.E / (2 * (1 + mat.nu))
    trial = np.einsum("nij,nj->ni", D, strain - plastic_strain)
    pressure = trial[:, :3].mean(axis=1)
    s = trial - pressure[:, None] * _NORMAL
    norm = _tensor_norm(s)
    q = np.sqrt(1.5) * norm
    # flow stress σ_f(α): linear hardening up to R_m (at α_u), then H_after
    capped = np.isfinite(mat.ultimate) & np.isfinite(mat.yield_stress)
    alpha_u = np.full(len(alpha), np.inf)  # no cap: never "beyond"
    alpha_u[capped] = (mat.ultimate[capped] - mat.yield_stress[capped]) / mat.H[capped]
    beyond = alpha >= alpha_u
    flow = mat.yield_stress + mat.H * alpha
    flow[beyond] = mat.ultimate[beyond] + mat.H_after[beyond] * (alpha - alpha_u)[beyond]
    f = q - flow
    plastic = f > 1e-10 * np.maximum(mat.yield_stress, 1.0)
    stress = trial.copy()
    eps_p = plastic_strain.copy()
    a = alpha.copy()
    Dt = D if tangent else None
    if np.any(plastic):
        i = np.flatnonzero(plastic)
        Gi, qi = G[i], q[i]
        Hi = np.where(beyond[i], mat.H_after[i], mat.H[i])
        dgamma = f[i] / (3 * Gi + Hi)
        # the step crosses R_m: finish it on the flat branch
        cross = ~beyond[i] & (alpha[i] + dgamma > alpha_u[i])
        if np.any(cross):
            Hc = mat.H_after[i][cross]
            dgamma[cross] = (
                qi[cross] - mat.ultimate[i][cross] - Hc * (alpha[i][cross] - alpha_u[i][cross])
            ) / (3 * Gi[cross] + Hc)
            Hi = np.where(cross, mat.H_after[i], Hi)
        n_hat = s[i] / norm[i, None]  # unit deviatoric direction (tensor components)
        stress[i] = trial[i] - (2 * Gi * dgamma * np.sqrt(1.5))[:, None] * n_hat
        flow = np.sqrt(1.5) * dgamma[:, None] * n_hat
        flow[:, 3:] *= 2  # engineering shear
        eps_p[i] += flow
        a[i] += dgamma
        if tangent:
            Dt = D.copy()
            ratio = dgamma / qi
            Dt[i] -= (3 * Gi * ratio)[:, None, None] * (Gi[:, None, None] * _DEV)
            Dt[i] += (6 * Gi**2 * (ratio - 1 / (3 * Gi + Hi)))[:, None, None] * np.einsum(
                "ni,nj->nij", n_hat, n_hat
            )
    return stress, Dt, eps_p, a
