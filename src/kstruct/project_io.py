"""Save / load projects as .kst files: a zip with project.json and one BREP file per part."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape

from kstruct.model import (
    AnalysisKind,
    Bolt,
    Connection,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Part,
    Project,
)
from kstruct.standards import material_from_record, material_record

FORMAT_VERSION = 1
SUFFIX = ".kst"


class ProjectFileError(Exception):
    pass


def _refs(refs: list[FaceRef]) -> list[list[int]]:
    return [[r.part_id, r.face_index] for r in refs]


def _unrefs(data: list[list[int]]) -> list[FaceRef]:
    return [FaceRef(int(p), int(f)) for p, f in data]


def save_project(project: Project, path: str | Path) -> Path:
    path = Path(path)
    doc = {
        "format": "kstruct",
        "version": FORMAT_VERSION,
        "source_path": str(project.source_path) if project.source_path else None,
        "mesh_quality": project.mesh_quality,
        "standard": project.standard,
        "analysis": project.analysis.value,
        "reference_temperature": project.reference_temperature,
        "operating_frequency": project.operating_frequency,
        "n_modes": project.n_modes,
        "load_cycle": project.load_cycle,
        "units": project.units,
        "target_overrides": dict(project.target_overrides),
        "custom_materials": [material_record(m) for m in project.custom_materials.values()],
        "parts": [
            {"id": p.id, "name": p.name, "material_id": p.material_id, "visible": p.visible}
            for p in project.parts
        ],
        "fixtures": [{"kind": f.kind.value, "faces": _refs(f.faces)} for f in project.fixtures],
        "loads": [
            {
                "kind": ld.kind.value,
                "faces": _refs(ld.faces),
                "vector": list(ld.vector),
                "magnitude": ld.magnitude,
                "ambient": ld.ambient,
            }
            for ld in project.loads
        ],
        "bolts": [
            {
                "hole_a": _refs([b.hole_a])[0],
                "hole_b": _refs([b.hole_b])[0],
                "size": b.size,
                "grade": b.grade,
                "preload": b.preload,
                "name": b.name,
            }
            for b in project.bolts
        ],
        "connections": [
            {
                "kind": c.kind.value,
                "faces_a": _refs(c.faces_a),
                "faces_b": _refs(c.faces_b),
                "friction": c.friction,
            }
            for c in project.connections
        ],
    }
    with (
        tempfile.TemporaryDirectory() as tmp,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z,
    ):
        z.writestr("project.json", json.dumps(doc, ensure_ascii=False, indent=2))
        for p in project.parts:
            brep = Path(tmp) / f"part{p.id}.brep"
            BRepTools.Write_s(p.shape, str(brep))
            z.write(brep, f"parts/{p.id}.brep")
    return path


def load_project(path: str | Path) -> Project:
    path = Path(path)
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise ProjectFileError(f"프로젝트 파일을 열 수 없습니다: {path.name}") from e
    with z, tempfile.TemporaryDirectory() as tmp:
        doc = json.loads(z.read("project.json"))
        if doc.get("format") != "kstruct" or doc.get("version", 0) > FORMAT_VERSION:
            raise ProjectFileError("이 버전의 KStruct에서 열 수 없는 프로젝트 파일입니다.")
        project = Project(
            source_path=Path(doc["source_path"]) if doc.get("source_path") else None,
            mesh_quality=doc.get("mesh_quality", "normal"),
            standard=doc.get("standard", "iso"),
            analysis=AnalysisKind(doc.get("analysis", "static")),
            reference_temperature=doc.get("reference_temperature", 20.0),
            operating_frequency=doc.get("operating_frequency", 0.0),
            n_modes=doc.get("n_modes", 6),
            load_cycle=doc.get("load_cycle", "none"),
            units=doc.get("units"),
            target_overrides={k: float(v) for k, v in doc.get("target_overrides", {}).items()},
            custom_materials={
                m["id"]: material_from_record(m) for m in doc.get("custom_materials", [])
            },
        )
        for pd in doc["parts"]:
            brep = Path(z.extract(f"parts/{pd['id']}.brep", tmp))
            shape = TopoDS_Shape()
            if not BRepTools.Read_s(shape, str(brep), BRep_Builder()):
                raise ProjectFileError(f"부품 형상을 읽을 수 없습니다: {pd['name']}")
            project.parts.append(
                Part(pd["id"], pd["name"], shape, pd.get("material_id"), pd.get("visible", True))
            )
    project.fixtures = [
        Fixture(FixtureKind(f["kind"]), _unrefs(f["faces"])) for f in doc["fixtures"]
    ]
    project.loads = [
        Load(
            LoadKind(ld["kind"]),
            _unrefs(ld["faces"]),
            tuple(ld["vector"]),
            ld["magnitude"],
            ld.get("ambient", 20.0),
        )
        for ld in doc["loads"]
    ]
    project.bolts = [
        Bolt(
            _unrefs([b["hole_a"]])[0],
            _unrefs([b["hole_b"]])[0],
            b["size"],
            b["grade"],
            b["preload"],
            b.get("name", ""),
        )
        for b in doc.get("bolts", [])
    ]
    project.connections = [
        Connection(
            ConnectionKind(c["kind"]), _unrefs(c["faces_a"]), _unrefs(c["faces_b"]), c["friction"]
        )
        for c in doc["connections"]
    ]
    return project
