"""Triangle-mesh formats (STL, OBJ, glTF/GLB, 3MF, FBX) -> solid parts.

The triangles are turned into a B-rep solid with shared vertices and edges (no sewing
needed), then coplanar triangles are merged into single planar faces so the user can
click a whole flat side. Curved regions stay faceted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
)
from OCP.gp import gp_Pnt
from OCP.ShapeFix import ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Shell, TopoDS_Vertex, TopoDS_Wire

from kstruct.geometry.step_import import GeometryImportError
from kstruct.model import Part, Project

MESH_SUFFIXES = {".stl", ".obj", ".gltf", ".glb", ".3mf", ".fbx", ".ply", ".off"}
MAX_TRIANGLES = 40_000
# formats whose coordinates are metres by specification
_METRE_FORMATS = {".gltf", ".glb"}


def _read_triangles(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        from kstruct.geometry.fbx import FbxError, read_fbx

        try:
            return [(m.name, m.vertices, m.triangles) for m in read_fbx(path)]
        except FbxError as e:
            raise GeometryImportError(str(e)) from e

    import trimesh

    try:
        scene = trimesh.load(str(path), force="scene")
    except Exception as e:  # trimesh raises many exception types for bad files
        raise GeometryImportError(f"파일을 읽을 수 없습니다: {path.name}") from e
    scale = 1000.0 if suffix in _METRE_FORMATS else 1.0
    if suffix == ".3mf":
        unit = (scene.units or "millimeters").lower()
        scale = {"meters": 1000.0, "centimeters": 10.0, "inches": 25.4, "micron": 1e-3}.get(
            unit, 1.0
        )
    out = []
    for name, geom in _named(scene):
        out.append((name, np.asarray(geom.vertices) * scale, np.asarray(geom.faces)))
    return out


def _named(scene):
    import trimesh

    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        yield str(node), geom.copy().apply_transform(transform)


def _clean(vertices: np.ndarray, triangles: np.ndarray):
    """Merge duplicate vertices, drop degenerate triangles, fix winding; returns trimesh.

    Holes are deliberately not filled: automatic filling can silently change the shape
    (e.g. cut a corner off), which is worse than asking the user for a closed model.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices, triangles, process=True, validate=True)
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def mesh_to_solid(vertices: np.ndarray, triangles: np.ndarray) -> TopoDS_Shape:
    builder = BRep_Builder()
    vtx = [_vertex(builder, p) for p in vertices]
    edges: dict[tuple[int, int], object] = {}

    def edge(a: int, b: int):
        key = (min(a, b), max(a, b))
        if key not in edges:
            edges[key] = BRepBuilderAPI_MakeEdge(vtx[key[0]], vtx[key[1]]).Edge()
        e = edges[key]
        return e if a == key[0] else TopoDS.Edge(e.Reversed())

    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for a, b, c in triangles:
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        for e in (edge(a, b), edge(b, c), edge(c, a)):
            builder.Add(wire, e)
        face = BRepBuilderAPI_MakeFace(wire, True)
        if face.IsDone():
            builder.Add(shell, face.Face())
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    unify = ShapeUpgrade_UnifySameDomain(fix.Solid(), True, True, False)
    unify.Build()
    return unify.Shape()


def _vertex(builder: BRep_Builder, p: np.ndarray) -> TopoDS_Vertex:
    v = TopoDS_Vertex()
    builder.MakeVertex(v, gp_Pnt(*map(float, p)), 1e-7)
    return v


def load_mesh(path: str | Path) -> Project:
    path = Path(path)
    if path.suffix.lower() not in MESH_SUFFIXES:
        raise GeometryImportError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
    project = Project(source_path=path)
    for i, (name, vertices, triangles) in enumerate(_read_triangles(path)):
        mesh = _clean(vertices, triangles)
        if len(mesh.faces) > MAX_TRIANGLES:
            raise GeometryImportError(
                f"'{name}'의 삼각형이 {len(mesh.faces):,}개로 너무 많습니다 "
                f"(최대 {MAX_TRIANGLES:,}개). CAD에서 STEP으로 저장하거나, 메시를 단순화해 주세요."
            )
        if not mesh.is_watertight:
            raise GeometryImportError(
                f"'{name}' 형상에 구멍이나 틈이 있어 속이 찬 물체로 만들 수 없습니다. "
                "닫힌(watertight) 메시가 필요합니다. 가능하면 원본 CAD에서 STEP으로 저장해 주세요."
            )
        shape = mesh_to_solid(mesh.vertices, mesh.faces)
        project.parts.append(Part(id=i, name=name, shape=shape))
    if not project.parts:
        raise GeometryImportError("파일에 형상이 없습니다.")
    return project
