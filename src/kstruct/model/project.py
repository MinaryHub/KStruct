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
    # optional (°C, W/(m·K)) points; the thermal solver interpolates (and extrapolates
    # flat) between them instead of using thermal_conductivity
    conductivity_table: tuple[tuple[float, float], ...] | None = None
    specific_heat: float = 0.0  # J/(kg·K)
    fatigue_limit: float | None = None  # MPa, fully reversed, polished specimen
    max_service_temperature: float | None = None  # °C, approximate continuous-use limit
    ductile: bool = True
    # brittle materials: strength in compression (MPa); None = assume the tensile strength
    compressive_strength: float | None = None
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
    RADIATION = "radiation"  # magnitude = emissivity (0..1), to surroundings at `ambient` °C
    # rotating machinery
    TORQUE = "torque"  # magnitude N·mm about the axis (vector = unit direction, point on axis)
    BEARING = "bearing"  # force vector (N) on a cylindrical hole, cosine distribution
    CENTRIFUGAL = "centrifugal"  # magnitude = ω rad/s about the axis (vector, point); all parts


THERMAL_LOADS = (
    LoadKind.FACE_TEMPERATURE,
    LoadKind.HEAT_FLOW,
    LoadKind.CONVECTION,
    LoadKind.RADIATION,
)
TEMPERATURE_REFERENCES = (LoadKind.FACE_TEMPERATURE, LoadKind.CONVECTION, LoadKind.RADIATION)
AXIS_LOADS = (LoadKind.TORQUE, LoadKind.CENTRIFUGAL)  # need an axis (vector + point)


@dataclass
class Load:
    kind: LoadKind
    faces: list[FaceRef] = field(default_factory=list)
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnitude: float = 0.0
    ambient: float = 20.0  # °C, convection and radiation
    point: tuple[float, float, float] | None = None  # a point on the axis (torque, centrifugal)


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
class Bolt:
    """A bolt joining the hole face of one part to the hole face of another."""

    hole_a: FaceRef
    hole_b: FaceRef
    size: str = "M10"  # resources/bolts.json size name
    grade: str = "8.8"  # ISO 898-1 property class
    preload: float = 0.0  # N, assembly tension
    name: str = ""


@dataclass
class MeshRefinement:
    """Smaller elements on chosen faces (size in mm)."""

    faces: list[FaceRef]
    size: float


@dataclass
class LoadCase:
    """One set of loads that act together. A project has at least one; the strength
    analysis solves every case and reports the most unfavourable one."""

    name: str
    loads: list[Load] = field(default_factory=list)


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
    bolts: list[Bolt] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    cases: list[LoadCase] = field(default_factory=lambda: [LoadCase("하중 케이스 1")])
    active_case: int = 0  # the case shown in the wizard and used by thermal / buckling
    mesh_quality: str = "normal"  # key of fem.mesh.MESH_PRESETS
    mesh_refinements: list[MeshRefinement] = field(default_factory=list)
    standard: str = "iso"  # standards pack id
    analysis: AnalysisKind = AnalysisKind.STATIC
    reference_temperature: float = 20.0  # °C, stress-free temperature for thermal stress
    operating_frequency: float = 0.0  # Hz, excitation to compare natural frequencies with
    n_modes: int = 6
    load_cycle: str = "none"  # fem.fatigue.LoadCycle value
    fatigue_surface: str = "machined"  # fem.fatigue.SURFACE_FACTORS key
    fatigue_reliability: str = "90"  # fem.fatigue.RELIABILITY_FACTORS key (%)
    units: str | None = None  # "SI" / "US"; None follows the standards pack
    # recommended safety factors chosen for this project ("ductile" / "brittle" / "buckling");
    # missing keys follow the standards pack
    target_overrides: dict[str, float] = field(default_factory=dict)
    # materials the user defined for this project, by material id (saved in the .kst file)
    custom_materials: dict[str, Material] = field(default_factory=dict)

    def part(self, part_id: int) -> Part:
        return next(p for p in self.parts if p.id == part_id)

    def visible_parts(self) -> list[Part]:
        return [p for p in self.parts if p.visible]

    @property
    def case(self) -> LoadCase:
        if not self.cases:
            self.cases.append(LoadCase("하중 케이스 1"))
        self.active_case = min(max(self.active_case, 0), len(self.cases) - 1)
        return self.cases[self.active_case]

    @property
    def loads(self) -> list[Load]:
        """Loads of the active case (what the solvers and the wizard work with)."""
        return self.case.loads

    @loads.setter
    def loads(self, value: list[Load]) -> None:
        self.case.loads = list(value)

    def all_loads(self) -> list[Load]:
        return [ld for c in self.cases for ld in c.loads]


def with_case(project: Project, index: int) -> Project:
    """A shallow view of the project with another case active (nothing is copied, so it is
    safe to use from a worker while the UI keeps showing the chosen case)."""
    from dataclasses import replace

    return replace(project, active_case=index)


def active_project(project: Project) -> Project:
    """The project as analysed: hidden parts are left out together with every fixture,
    load, connection and bolt that refers to them (a shallow copy; parts are shared)."""
    hidden = {p.id for p in project.parts if not p.visible}
    if not hidden:
        return project

    def keep(refs: list[FaceRef]) -> bool:
        return not any(r.part_id in hidden for r in refs)

    out = Project(**{k: getattr(project, k) for k in project.__dataclass_fields__})
    out.parts = project.visible_parts()
    out.fixtures = [f for f in project.fixtures if keep(f.faces)]
    out.cases = [LoadCase(c.name, [ld for ld in c.loads if keep(ld.faces)]) for c in project.cases]
    out.connections = [c for c in project.connections if keep(c.faces_a) and keep(c.faces_b)]
    out.mesh_refinements = [r for r in project.mesh_refinements if keep(r.faces)]
    out.bolts = [b for b in project.bolts if keep([b.hole_a, b.hole_b])]
    return out


def excluded_by_hidden_parts(project: Project) -> list[str]:
    """Short Korean descriptions of what active_project() drops, for the checklist."""
    active = active_project(project)
    counts = {
        "고정": len(project.fixtures) - len(active.fixtures),
        "하중": len(project.all_loads()) - len(active.all_loads()),
        "연결": len(project.connections) - len(active.connections),
        "볼트": len(project.bolts) - len(active.bolts),
    }
    return [f"{name} {n}개" for name, n in counts.items() if n]
