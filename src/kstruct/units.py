"""Display units. Everything is computed in mm-N-MPa-°C; only what the user sees and types
is converted. The unit system follows the standards pack ("SI" or "US") unless overridden.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_IN = 25.4
N_PER_LBF = 4.4482216152605
PSI_PER_MPA = 145.03773773


@dataclass(frozen=True)
class Quantity:
    unit: str
    to_display: float  # multiply internal value by this
    offset: float = 0.0  # added after scaling (temperature)

    def show(self, value: float) -> float:
        return value * self.to_display + self.offset

    def parse(self, value: float) -> float:
        return (value - self.offset) / self.to_display


SI = {
    "length": Quantity("mm", 1.0),
    "force": Quantity("N", 1.0),
    "stress": Quantity("MPa", 1.0),
    "temperature": Quantity("°C", 1.0),
    "temperature_change": Quantity("°C", 1.0),
    "power": Quantity("W", 1.0),
    "torque": Quantity("N·m", 1e-3),  # internal N·mm
    "speed": Quantity("rpm", 60.0 / (2 * 3.141592653589793)),  # internal rad/s
}
US = {
    "length": Quantity("in", 1 / MM_PER_IN),
    "force": Quantity("lbf", 1 / N_PER_LBF),
    "stress": Quantity("psi", PSI_PER_MPA),
    "temperature": Quantity("°F", 9 / 5, 32.0),
    "temperature_change": Quantity("°F", 9 / 5),
    "power": Quantity("W", 1.0),
    "torque": Quantity("lbf·in", 1 / (N_PER_LBF * MM_PER_IN)),
    "speed": Quantity("rpm", 60.0 / (2 * 3.141592653589793)),
}
SYSTEMS = {"SI": SI, "US": US}


def system_of(project) -> str:
    """Effective unit system: the project's choice, else its standards pack's default."""
    if project.units:
        return project.units
    from kstruct import standards

    return standards.pack(project.standard).units


def quantity(system: str, kind: str) -> Quantity:
    return SYSTEMS.get(system, SI)[kind]


def fmt(system: str, kind: str, value: float, digits: str | None = None) -> str:
    """Value with its unit. Without `digits`: four significant digits, never in exponent
    notation for ordinary magnitudes (",.4g" printed 14,500 psi as "1.45e+04 psi")."""
    q = quantity(system, kind)
    shown = q.show(value)
    if digits is not None:
        return f"{shown:{digits}} {q.unit}"
    if abs(shown) >= 1000:
        return f"{shown:,.0f} {q.unit}"
    return f"{shown:,.4g} {q.unit}"
