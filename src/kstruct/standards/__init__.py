"""Material library and standards packs (data lives in resources/*.json).

A standards pack decides which designation a material is shown under (S235JR vs SS235 vs
A36 ...), the default unit system and the target safety factors used for the verdict.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from importlib import resources

from kstruct.model import Material, Project

CATEGORY_NAMES = {
    "steel": "강",
    "stainless": "스테인리스강",
    "aluminum": "알루미늄",
    "cast_iron": "주철",
    "titanium": "티타늄",
    "copper": "구리 합금",
    "plastic": "플라스틱",
}


@dataclass(frozen=True)
class StandardPack:
    id: str
    label: str
    names: str
    units: str  # "SI" or "US"
    target_ductile: float
    target_brittle: float
    target_buckling: float
    fatigue: str
    references: tuple[str, ...]

    def material_name(self, material: Material) -> str:
        return material.designations.get(self.names) or material.name

    def target(self, material: Material) -> float:
        return self.target_ductile if material.ductile else self.target_brittle


def _read(name: str) -> dict:
    return json.loads(resources.files("kstruct.resources").joinpath(name).read_text("utf-8"))


def material_from_record(m: dict) -> Material:
    """Material from the materials.json record format (everyday units: kg/m³, J/kg·K)."""
    return Material(
        id=m["id"],
        name=m["ko"],
        category=m["category"],
        youngs_modulus=float(m["E"]),
        poisson_ratio=float(m["nu"]),
        density=float(m["density"]) * 1e-12,  # kg/m³ -> t/mm³
        yield_strength=float(m["yield"]),
        tensile_strength=float(m["uts"]),
        thermal_expansion=float(m.get("alpha", 0.0)),
        thermal_conductivity=float(m.get("k", 0.0)),  # W/(m·K) == N/(s·K) in mm-N-s units
        conductivity_table=tuple((float(t), float(k)) for t, k in m["k_table"])
        if m.get("k_table")
        else None,
        specific_heat=float(m.get("cp", 0.0)) * 1e6,  # J/(kg·K) -> mm²/(s²·K)
        fatigue_limit=m.get("fatigue"),
        max_service_temperature=m.get("tmax"),
        ductile=bool(m["ductile"]),
        designations=dict(m.get("names", {})),
    )


def material_record(m: Material) -> dict:
    """Inverse of material_from_record (what the project file and user library store)."""
    return {
        "id": m.id,
        "ko": m.name,
        "category": m.category,
        "E": m.youngs_modulus,
        "nu": m.poisson_ratio,
        "density": m.density * 1e12,
        "yield": m.yield_strength,
        "uts": m.tensile_strength,
        "alpha": m.thermal_expansion,
        "k": m.thermal_conductivity,
        "k_table": [list(p) for p in m.conductivity_table] if m.conductivity_table else None,
        "cp": m.specific_heat * 1e-6,
        "fatigue": m.fatigue_limit,
        "tmax": m.max_service_temperature,
        "ductile": m.ductile,
        "names": dict(m.designations),
    }


def new_material_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


@cache
def materials() -> dict[str, Material]:
    """Built-in material library."""
    return {m["id"]: material_from_record(m) for m in _read("materials.json")["materials"]}


def project_materials(project: Project | None) -> dict[str, Material]:
    """Built-in materials plus the project's own (user-defined) ones."""
    out = dict(materials())
    if project is not None:
        out.update(project.custom_materials)
    return out


@cache
def packs() -> dict[str, StandardPack]:
    return {
        p["id"]: StandardPack(
            id=p["id"],
            label=p["label"],
            names=p["names"],
            units=p["units"],
            target_ductile=p["targets"]["ductile"],
            target_brittle=p["targets"]["brittle"],
            target_buckling=p["targets"].get("buckling", 3.0),
            fatigue=p["fatigue"],
            references=tuple(p["references"]),
        )
        for p in _read("standards.json")["packs"]
    }


def pack(pack_id: str) -> StandardPack:
    return packs().get(pack_id) or packs()[_read("standards.json")["default"]]


TARGET_KEYS = ("ductile", "brittle", "buckling")


def effective_pack(project: Project) -> StandardPack:
    """The project's standards pack with its own recommended safety factors applied."""
    base = pack(project.standard)
    o = project.target_overrides or {}
    return replace(
        base,
        target_ductile=float(o.get("ductile", base.target_ductile)),
        target_brittle=float(o.get("brittle", base.target_brittle)),
        target_buckling=float(o.get("buckling", base.target_buckling)),
    )


class Verdict(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


def verdict(safety_factor: float, target: float) -> Verdict:
    if safety_factor >= target:
        return Verdict.OK
    if safety_factor >= 1.0:
        return Verdict.WARN
    return Verdict.FAIL
