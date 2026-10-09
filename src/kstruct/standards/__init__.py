"""Material library and standards packs (data lives in resources/*.json).

A standards pack decides which designation a material is shown under (S235JR vs SS235 vs
A36 ...), the default unit system and the target safety factors used for the verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib import resources

from kstruct.model import Material

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
    fatigue: str
    references: tuple[str, ...]

    def material_name(self, material: Material) -> str:
        return material.designations.get(self.names) or material.name

    def target(self, material: Material) -> float:
        return self.target_ductile if material.ductile else self.target_brittle


def _read(name: str) -> dict:
    return json.loads(resources.files("kstruct.resources").joinpath(name).read_text("utf-8"))


@cache
def materials() -> dict[str, Material]:
    out = {}
    for m in _read("materials.json")["materials"]:
        out[m["id"]] = Material(
            id=m["id"],
            name=m["ko"],
            category=m["category"],
            youngs_modulus=m["E"],
            poisson_ratio=m["nu"],
            density=m["density"] * 1e-12,  # kg/m³ -> t/mm³
            yield_strength=m["yield"],
            tensile_strength=m["uts"],
            thermal_expansion=m["alpha"],
            thermal_conductivity=m["k"],  # W/(m·K) is numerically N/(s·K) in mm-N-s units
            specific_heat=m["cp"] * 1e6,  # J/(kg·K) -> mm²/(s²·K)
            fatigue_limit=m.get("fatigue"),
            max_service_temperature=m.get("tmax"),
            ductile=m["ductile"],
            designations=m["names"],
        )
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
            fatigue=p["fatigue"],
            references=tuple(p["references"]),
        )
        for p in _read("standards.json")["packs"]
    }


def pack(pack_id: str) -> StandardPack:
    return packs().get(pack_id) or packs()[_read("standards.json")["default"]]


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
