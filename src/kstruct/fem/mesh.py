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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kstruct.geometry.samples import write_step
from kstruct.geometry.tessellate import bounding_diagonal
from kstruct.i18n import tr
from kstruct.model import FaceRef, Project

MESH_TIMEOUT = 600  # seconds per part before the meshing process is killed


@dataclass(frozen=True)
class MeshPreset:
    divisions: float  # model diagonal / global element size
    grading: float  # Netgen size grading (smaller = smoother transitions, more elements)
    layers: int  # elements wanted through a wall: local size <= thickness / layers


MESH_PRESETS = {
    "coarse": MeshPreset(10.0, 0.3, 1),
    "normal": MeshPreset(20.0, 0.3, 2),
    "fine": MeshPreset(30.0, 0.25, 3),
    "very_fine": MeshPreset(40.0, 0.2, 4),
}


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
    notes: list[str] = field(default_factory=list)  # what the sizing did (shown in step 5)

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


def _mesh_shape(
    step_path: Path, maxh: float, face_maxh: dict[int, float] | None = None, grading=0.3
):
    from netgen.occ import OCCGeometry

    geo = OCCGeometry(str(step_path))
    if face_maxh:
        # per-face sizes only take effect when the geometry is rebuilt from the shape
        # after setting them (setting them on geo.shape alone is ignored)
        shape = geo.shape
        faces = shape.faces  # same order as geometry.faces_of (checked by tests)
        for index, h in face_maxh.items():
            if 0 <= index < len(faces):
                faces[index].maxh = float(h)
        geo = OCCGeometry(shape)
    mesh = geo.GenerateMesh(maxh=maxh, grading=grading, closeedgefac=2.0)
    mesh.SecondOrder()
    # copy: Coordinates() is a view into netgen memory, freed with the mesh object
    nodes = np.array(mesh.Coordinates(), dtype=float, copy=True)
    tets = mesh.Elements3D().NumPy()["nodes"].astype(np.int64) - 1
    surf = mesh.Elements2D().NumPy()
    tris = surf["nodes"].astype(np.int64) - 1
    faces = surf["index"].astype(np.int64) - 1
    return nodes, tets, tris, faces


def _mesh_all(
    steps: list[Path],
    maxh: float,
    names: list[str],
    face_maxh=None,
    grading=0.3,
    check: Callable[[], None] | None = None,
):
    """Mesh every part in worker processes (in parallel for assemblies).

    Netgen occasionally deadlocks inside its own thread pool; a separate process can be
    killed after MESH_TIMEOUT seconds and retried once, instead of freezing the app.
    `check` is polled while waiting and may raise (the user cancelled); the worker
    processes are then killed. Set KSTRUCT_INPROCESS_MESH=1 to mesh in this process
    (debugging).
    """
    face_maxh = face_maxh or [None] * len(steps)
    if os.environ.get("KSTRUCT_INPROCESS_MESH") == "1":
        return [
            _mesh_or_raise(step, maxh, name, fh, grading)
            for step, name, fh in zip(steps, names, face_maxh, strict=True)
        ]
    workers = max(1, min(len(steps), os.cpu_count() or 1, 4))
    for attempt in (1, 2):
        pool = multiprocessing.get_context("spawn").Pool(workers)
        try:
            jobs = [
                pool.apply_async(_mesh_shape, (str(step), maxh, fh, grading))
                for step, fh in zip(steps, face_maxh, strict=True)
            ]
            results = []
            deadline = time.monotonic() + MESH_TIMEOUT
            for job, name in zip(jobs, names, strict=True):
                try:
                    results.append(_wait(job, deadline, check))
                except multiprocessing.TimeoutError:
                    if attempt == 2:
                        raise MeshingError(
                            tr(
                                "'{name}' 부품의 메시 생성이 {minutes}분 안에 끝나지 않았습니다. "
                                "메시 품질을 '거침'이나 '보통'으로 바꿔 보세요.",
                                name=name,
                                minutes=MESH_TIMEOUT // 60,
                            )
                        ) from None
                    raise _Retry from None
                except _Stop as stop:  # check() raised: pass its exception on untouched
                    raise stop.error from None
                except Exception as e:  # netgen raises plain Exception
                    raise MeshingError(
                        tr("'{name}' 부품의 메시 생성에 실패했습니다: {e}", name=name, e=e)
                    ) from e
            return results
        except _Retry:
            continue
        finally:
            pool.terminate()
            pool.join()
    raise AssertionError("unreachable")


class _Retry(Exception):
    pass


class _Stop(Exception):
    def __init__(self, error: BaseException) -> None:
        self.error = error


def _wait(job, deadline: float, check: Callable[[], None] | None):
    """job.get() until the deadline, calling check() (which may raise) a few times a
    second so a cancel does not wait for Netgen."""
    while True:
        if check is not None:
            try:
                check()
            except Exception as e:
                raise _Stop(e) from e
        left = deadline - time.monotonic()
        if left <= 0:
            raise multiprocessing.TimeoutError
        try:
            return job.get(timeout=min(left, 0.25))
        except multiprocessing.TimeoutError:
            continue


def _mesh_or_raise(step: Path, maxh: float, name: str, face_maxh=None, grading=0.3):
    try:
        return _mesh_shape(step, maxh, face_maxh, grading)
    except Exception as e:  # netgen raises plain Exception
        raise MeshingError(
            tr("'{name}' 부품의 메시 생성에 실패했습니다: {e}", name=name, e=e)
        ) from e


def face_sizes(
    project: Project, quality: str | None = None
) -> tuple[float, list[dict[int, float]], list[str]]:
    """Global element size and, per visible part, the faces that get a smaller size:
    thin walls (preset.layers elements through the thickness) and the user's refinements.
    Returns (maxh, per-part {face index: size}, notes for the user)."""
    from kstruct.geometry.thickness import face_thickness

    preset = MESH_PRESETS[quality or project.mesh_quality]
    parts = [p for p in project.parts if p.visible]
    diag = max(bounding_diagonal(p.shape) for p in parts)
    maxh = diag / preset.divisions
    per_part, notes = [], []
    for part in parts:
        sizes: dict[int, float] = {}
        thin = []
        for index, t in face_thickness(part.shape).items():
            wanted = t / preset.layers
            if wanted < maxh:
                sizes[index] = max(wanted, 1e-3 * diag)
                thin.append(t)
        if thin:
            prefix = f"{part.name}: " if len(parts) > 1 else ""
            notes.append(
                tr(
                    "{prefix}얇은 부분(두께 약 {t_min:.3g}~{t_max:.3g} mm) {n}개 면의 "
                    "요소를 {size:.3g} mm로 줄여 두께 방향으로 {layers}개 이상 "
                    "넣었습니다.",
                    prefix=prefix,
                    t_min=min(thin),
                    t_max=max(thin),
                    n=len(thin),
                    size=min(sizes.values()),
                    layers=preset.layers,
                )
            )
        for r in project.mesh_refinements:
            for ref in r.faces:
                if ref.part_id == part.id:
                    sizes[ref.face_index] = min(sizes.get(ref.face_index, maxh), r.size)
        per_part.append(sizes)
    if project.mesh_refinements:
        n = sum(len(r.faces) for r in project.mesh_refinements)
        notes.append(tr("사용자가 지정한 면 {n}개에 더 작은 요소를 썼습니다.", n=n))
    return maxh, per_part, notes


def generate_mesh(
    project: Project, quality: str | None = None, check: Callable[[], None] | None = None
) -> FEMesh:
    """`check` is polled while the parts are meshed and may raise to abandon the job."""
    quality = quality or project.mesh_quality
    preset = MESH_PRESETS[quality]
    parts = [p for p in project.parts if p.visible]
    if not parts:
        raise MeshingError(tr("메시를 만들 부품이 없습니다."))
    maxh, per_part, notes = face_sizes(project, quality)

    nodes, tets, tet_part, tris, tri_part, tri_face = [], [], [], [], [], []
    offset = 0
    with tempfile.TemporaryDirectory() as tmp:
        steps = [write_step(p.shape, Path(tmp) / f"part{p.id}.step") for p in parts]
        meshes = _mesh_all(steps, maxh, [p.name for p in parts], per_part, preset.grading, check)
        for part, (n, t, s, f) in zip(parts, meshes, strict=True):
            if len(t) == 0:
                raise MeshingError(tr("'{name}' 부품의 메시 생성에 실패했습니다.", name=part.name))
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
        notes=notes,
    )
