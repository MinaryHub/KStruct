"""Analysis project data model.

A project always holds a list of parts, even when only one is analysed, so that
assemblies can be added later without changing the model shape.

Unit system (consistent, used everywhere internally): mm, N, MPa, tonne, s.
Density is therefore in t/mm³ (steel ≈ 7.85e-9) and acceleration in mm/s² (g = 9810).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from OCP.TopoDS import TopoDS_Shape

GRAVITY = 9810.0  # mm/s²


@dataclass(frozen=True)
class FaceRef:
    """A B-rep face of a part, identified by its index in the part's face order."""

    part_id: int
    face_index: int


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    category: str  # "steel", "stainless", "aluminum", "cast_iron", "titanium", "copper", "plastic"
    youngs_modulus: float  # MPa
    poisson_ratio: float
    density: float  # t/mm³
    yield_strength: float  # MPa (0.2 % proof stress for metals without a yield point)
    tensile_strength: float  # MPa
    thermal_expansion: float = 0.0  # 1/K
    thermal_conductivity: float = 0.0  # W/(m·K) == mW/(mm·K) in the internal unit system
    specific_heat: float = 0.0  # J/(kg·K)
    fatigue_limit: float | None = None  # MPa, fully reversed, polished specimen
    max_service_temperature: float | None = None  # °C, approximate continuous-use limit
    ductile: bool = True
    designations: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


class FixtureKind(Enum):
    FIXED = "fixed"  # no translation at all
    ROLLER = "roller"  # can slide along the face, not move through it
    PIN = "pin"  # cylindrical face: can rotate about its axis, nothing else


@dataclass
class Fixture:
    kind: FixtureKind
    faces: list[FaceRef]


class LoadKind(Enum):
    FORCE = "force"  # total force vector (N) spread over the faces by area
    PRESSURE = "pressure"  # MPa, positive pushes into the surface
    GRAVITY = "gravity"  # acceleration vector (mm/s²) on all parts
    TEMPERATURE = "temperature"  # uniform temperature change (K) for thermal stress
    # thermal boundary conditions (steady-state heat transfer)
    FACE_TEMPERATURE = "face_temperature"  # faces held at magnitude °C
    HEAT_FLOW = "heat_flow"  # magnitude W entering through the faces
    CONVECTION = "convection"  # magnitude = h in W/(m²·K), to air at `ambient` °C


THERMAL_LOADS = (LoadKind.FACE_TEMPERATURE, LoadKind.HEAT_FLOW, LoadKind.CONVECTION)


@dataclass
class Load:
    kind: LoadKind
    faces: list[FaceRef] = field(default_factory=list)
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnitude: float = 0.0
    ambient: float = 20.0  # °C, convection only


class ConnectionKind(Enum):
    BONDED = "bonded"  # faces glued together (tie); linear
    CONTACT = "contact"  # may slide / separate; nonlinear (later)
    BOLT = "bolt"  # bolted joint with preload (later)


@dataclass
class Connection:
    kind: ConnectionKind
    faces_a: list[FaceRef]
    faces_b: list[FaceRef]
    friction: float = 0.0


@dataclass
class Part:
    id: int
    name: str
    shape: TopoDS_Shape
    material_id: str | None = None
    visible: bool = True


class AnalysisKind(Enum):
    STATIC = "static"  # strength (with thermal stress if thermal loads are present)
    MODAL = "modal"  # natural frequencies
    THERMAL = "thermal"  # steady-state temperature only
    BUCKLING = "buckling"  # load factor at which slender parts collapse sideways


@dataclass
class Project:
    source_path: Path | None = None
    parts: list[Part] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    mesh_quality: str = "normal"  # key of fem.mesh.MESH_PRESETS
    standard: str = "iso"  # standards pack id
    analysis: AnalysisKind = AnalysisKind.STATIC
    reference_temperature: float = 20.0  # °C, stress-free temperature for thermal stress
    operating_frequency: float = 0.0  # Hz, excitation to compare natural frequencies with
    n_modes: int = 6
    load_cycle: str = "none"  # fem.fatigue.LoadCycle value

    def part(self, part_id: int) -> Part:
        return next(p for p in self.parts if p.id == part_id)
