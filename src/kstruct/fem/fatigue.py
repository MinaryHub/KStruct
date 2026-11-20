"""High-cycle fatigue check of a linear static result (stress-life method).

The static load is taken as the peak of a repeated load:
  - pulsating:  0 -> peak -> 0     (sigma_a = sigma_m = sigma_peak / 2)
  - reversed:  -peak -> +peak      (sigma_a = sigma_peak, sigma_m = 0)

Equivalent stress is the signed von Mises stress (sign of the hydrostatic part), so
compressive mean stress is not credited as harmful. Modified Goodman for mean stress,
Basquin S-N line from (1e3, 0.9 Su) to (1e6, Se). Se is the polished-specimen fatigue
limit times a combined Marin factor for an ordinary machined surface and size.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from kstruct.fem.static import FOS_CAP, StaticResult
from kstruct.i18n import N_

MARIN_FACTOR = 0.7  # legacy combined factor (machined surface × size), used without settings
LOW_CYCLES, HIGH_CYCLES = 1e3, 1e6
INFINITE_LIFE = 1e9  # shown as "무한 수명"

# Marin surface factor ka = a · Su^b (Su in MPa), Shigley's table
SURFACE_FACTORS = {
    "ground": (N_("연삭"), 1.58, -0.085),
    "machined": (N_("기계 가공·냉간 인발"), 4.51, -0.265),
    "hot_rolled": (N_("열간 압연"), 57.7, -0.718),
    "as_forged": (N_("단조 그대로"), 272.0, -0.995),
}
# reliability factor kc (normal distribution, 8 % standard deviation of the fatigue limit)
RELIABILITY_FACTORS = {"50": 1.0, "90": 0.897, "95": 0.868, "99": 0.814, "99.9": 0.753}
SIZE_FACTOR = 0.85  # kb for ordinary part sizes (8-50 mm equivalent diameter)


@dataclass(frozen=True)
class FatigueSettings:
    surface: str = "machined"  # key of SURFACE_FACTORS
    reliability: str = "90"  # key of RELIABILITY_FACTORS (%)
    size_factor: float = SIZE_FACTOR

    def factors(self, su: float) -> dict[str, float]:
        """ka, kb, kc and their product for a material of tensile strength su (MPa)."""
        _, a, b = SURFACE_FACTORS.get(self.surface, SURFACE_FACTORS["machined"])
        ka = min(1.0, a * su**b)
        kc = RELIABILITY_FACTORS.get(self.reliability, 1.0)
        return {"ka": ka, "kb": self.size_factor, "kc": kc, "total": ka * self.size_factor * kc}


class LoadCycle(Enum):
    NONE = "none"
    PULSATING = "pulsating"
    REVERSED = "reversed"


@dataclass
class FatigueResult:
    safety_factor: np.ndarray  # (N,) Goodman safety factor against the endurance limit
    life: np.ndarray  # (N,) predicted cycles to crack initiation (INFINITE_LIFE if below Se)
    endurance: dict[int, float]  # part id -> corrected endurance limit Se, MPa
    factors: dict[int, dict[str, float]] | None = None  # part id -> Marin factors used


def signed_von_mises(result: StaticResult) -> np.ndarray:
    hydro = result.stress[:, :3].sum(axis=1)
    return np.where(hydro < 0, -1.0, 1.0) * result.von_mises


def fatigue_check(
    result: StaticResult, cycle: LoadCycle, settings: FatigueSettings | None = None
) -> FatigueResult:
    peak = signed_von_mises(result)
    if cycle == LoadCycle.REVERSED:
        amp, mean = np.abs(peak), np.zeros_like(peak)
    else:
        amp, mean = np.abs(peak) / 2, peak / 2

    n = result.mesh.n_nodes
    fos = np.full(n, FOS_CAP)
    life = np.full(n, INFINITE_LIFE)
    endurance = {}
    factors = {}
    for pid, m in result.materials.items():
        nodes = result.mesh.part_nodes(pid)
        su = m.tensile_strength
        if settings is None:
            factor = MARIN_FACTOR
        else:
            factors[pid] = settings.factors(su)
            factor = factors[pid]["total"]
        se = factor * (m.fatigue_limit or 0.5 * su)
        endurance[pid] = se
        a, mt = amp[nodes], np.maximum(mean[nodes], 0.0)  # compressive mean: no benefit taken
        with np.errstate(divide="ignore", invalid="ignore"):
            demand = a / se + mt / su
            fos[nodes] = np.minimum(np.where(demand > 0, 1.0 / demand, FOS_CAP), FOS_CAP)
            # equivalent fully reversed amplitude (Goodman), then Basquin life
            ar = np.where(mt < su, a / (1.0 - mt / su), np.inf)
            b = np.log10(0.9 * su / se) / np.log10(HIGH_CYCLES / LOW_CYCLES)
            cycles = LOW_CYCLES * (0.9 * su / ar) ** (1.0 / b)
        cycles = np.where(ar <= se, INFINITE_LIFE, np.clip(cycles, 1.0, INFINITE_LIFE))
        life[nodes] = np.where(ar >= 0.9 * su, np.minimum(cycles, LOW_CYCLES), cycles)
    return FatigueResult(fos, life, endurance, factors or None)
