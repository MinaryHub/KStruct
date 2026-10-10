"""Volume meshing with Netgen (quadratic tetrahedra, TET10).

Netgen bundles its own OpenCASCADE build, so OCP shapes cannot be passed directly.
Each part is handed over as a STEP file. Netgen numbers faces in the same order as
geometry.faces_of() (verified in tests), so surface triangles map straight back to FaceRef.

Node orderings (Netgen = VTK conventions):
  TET10 midside nodes 4..9 sit on edges (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
  TRI6  midside nodes 3..5 sit on edges (1,2) (0,2) (0,1)
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kstruct.geometry.samples import write_step
from kstruct.geometry.tessellate import bounding_diagonal
from kstruct.model import FaceRef, Project

# model diagonal / element size
MESH_TIMEOUT = 600  # seconds per part before the meshing process is killed
MESH_PRESETS = {"coarse": 8.0, "normal": 15.0, "fine": 30.0, "very_fine": 50.0}


class MeshingError(Exception):
    pass


@dataclass
class FEMesh:
    nodes: np.ndarray  # (N, 3) float
    tets: np.ndarray  # (E, 10) int, 0-based
    tet_part: np.ndarray  # (E,) part id
    tris: np.ndarray  # (S, 6) int, boundary triangles, 0-based
    tri_part: np.ndarray  # (S,) part id
    tri_face: np.ndarray  # (S,) B-rep face index within the part

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def face_tris(self, refs: list[FaceRef]) -> np.ndarray:
        """Indices of boundary triangles lying on any of the given faces."""
        mask = np.zeros(len(self.tris), dtype=bool)
        for ref in refs:
            mask |= (self.tri_part == ref.part_id) & (self.tri_face == ref.face_index)
        return np.flatnonzero(mask)

    def face_nodes(self, refs: list[FaceRef]) -> np.ndarray:
        return np.unique(self.tris[self.face_tris(refs)])

    def part_nodes(self, part_id: int) -> np.ndarray:
        return np.unique(self.tets[self.tet_part == part_id])


def _mesh_shape(step_path: Path, maxh: float):
    from netgen.occ import OCCGeometry

    geo = OCCGeometry(str(step_path))
    mesh = geo.GenerateMesh(maxh=maxh, grading=0.3, closeedgefac=2.0)
    mesh.SecondOrder()
    # copy: Coordinates() is a view into netgen memory, freed with the mesh object
    nodes = np.array(mesh.Coordinates(), dtype=float, copy=True)
    tets = mesh.Elements3D().NumPy()["nodes"].astype(np.int64) - 1
    surf = mesh.Elements2D().NumPy()
    tris = surf["nodes"].astype(np.int64) - 1
    faces = surf["index"].astype(np.int64) - 1
    return nodes, tets, tris, faces


def _mesh_all(steps: list[Path], maxh: float, names: list[str]):
    """Mesh every part in worker processes (in parallel for assemblies).

    Netgen occasionally deadlocks inside its own thread pool; a separate process can be
    killed after MESH_TIMEOUT seconds and retried once, instead of freezing the app.
    Set KSTRUCT_INPROCESS_MESH=1 to mesh in this process (debugging).
    """
    if os.environ.get("KSTRUCT_INPROCESS_MESH") == "1":
        return [_mesh_or_raise(step, maxh, name) for step, name in zip(steps, names, strict=True)]
    workers = max(1, min(len(steps), os.cpu_count() or 1, 4))
    for attempt in (1, 2):
        pool = multiprocessing.get_context("spawn").Pool(workers)
        try:
            jobs = [pool.apply_async(_mesh_shape, (str(step), maxh)) for step in steps]
            results = []
            for job, name in zip(jobs, names, strict=True):
                try:
                    results.append(job.get(timeout=MESH_TIMEOUT))
                except multiprocessing.TimeoutError:
                    if attempt == 2:
                        raise MeshingError(
                            f"'{name}' 부품의 메시 생성이 {MESH_TIMEOUT // 60}분 안에 끝나지 않았습니다. "
                            "메시 품질을 '거침'이나 '보통'으로 바꿔 보세요."
                        ) from None
                    raise _Retry from None
                except Exception as e:  # netgen raises plain Exception
                    raise MeshingError(f"'{name}' 부품의 메시 생성에 실패했습니다: {e}") from e
            return results
        except _Retry:
            continue
        finally:
            pool.terminate()
            pool.join()
    raise AssertionError("unreachable")


class _Retry(Exception):
    pass


def _mesh_or_raise(step: Path, maxh: float, name: str):
    try:
        return _mesh_shape(step, maxh)
    except Exception as e:  # netgen raises plain Exception
        raise MeshingError(f"'{name}' 부품의 메시 생성에 실패했습니다: {e}") from e


def generate_mesh(project: Project, quality: str | None = None) -> FEMesh:
    quality = quality or project.mesh_quality
    divisions = MESH_PRESETS[quality]
    parts = [p for p in project.parts if p.visible]
    if not parts:
        raise MeshingError("메시를 만들 부품이 없습니다.")
    diag = max(bounding_diagonal(p.shape) for p in parts)
    maxh = diag / divisions

    nodes, tets, tet_part, tris, tri_part, tri_face = [], [], [], [], [], []
    offset = 0
    with tempfile.TemporaryDirectory() as tmp:
        steps = [write_step(p.shape, Path(tmp) / f"part{p.id}.step") for p in parts]
        meshes = _mesh_all(steps, maxh, [p.name for p in parts])
        for part, (n, t, s, f) in zip(parts, meshes, strict=True):
            if len(t) == 0:
                raise MeshingError(f"'{part.name}' 부품의 메시 생성에 실패했습니다.")
            nodes.append(n)
            tets.append(t + offset)
            tet_part.append(np.full(len(t), part.id))
            tris.append(s + offset)
            tri_part.append(np.full(len(s), part.id))
            tri_face.append(f)
            offset += len(n)

    return FEMesh(
        nodes=np.vstack(nodes),
        tets=np.vstack(tets),
        tet_part=np.concatenate(tet_part),
        tris=np.vstack(tris),
        tri_part=np.concatenate(tri_part),
        tri_face=np.concatenate(tri_face),
    )
