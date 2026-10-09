"""3D viewer: CAD faces (pick / multi-select / annotate), FE mesh, and result contours."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from vtkmodules.vtkCommonCore import VTK_FONT_FILE
from vtkmodules.vtkRenderingCore import vtkCellPicker

from kstruct.fem.mesh import FEMesh
from kstruct.geometry import tessellate
from kstruct.geometry.tessellate import bounding_diagonal
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Project

PART_COLORS = ["#9fb4c7", "#c7b29f", "#a8c79f", "#c79fbf", "#9fc7c3", "#c7c39f"]
SELECT_COLOR = "#ffd23f"
FIXTURE_COLORS = {
    FixtureKind.FIXED: "#2e9e5b",
    FixtureKind.ROLLER: "#2aa1b3",
    FixtureKind.PIN: "#7b5cc4",
}
LOAD_COLOR = "#e8743b"
BOLT_COLOR = "#4a4f57"
CLICK_TOLERANCE_PX = 4  # press/release further apart than this is a rotate, not a click

# Netgen TRI6 (mid nodes on edges 12, 02, 01) -> VTK quadratic triangle (01, 12, 20)
_TRI6_TO_VTK = [0, 1, 2, 5, 3, 4]


@dataclass
class ResultField:
    values: np.ndarray  # (N,) per node
    title: str  # scalar bar title, with unit
    cmap: str = "turbo"
    clim: tuple[float, float] | None = None
    unit: str = ""


class ModelViewer(QWidget):
    selection_changed = Signal(list)  # list[FaceRef]
    probed = Signal(str)  # text describing the value under the cursor in result mode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter)

        self.plotter.set_background("#f4f6f8", top="#dfe5ec")
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing("ssaa")

        self._project: Project | None = None
        self._meshes: dict[int, pv.PolyData] = {}
        self._actors: dict[int, list] = {}
        self._actor_part: dict[str, int] = {}  # vtk actor address -> part id
        self._hidden: set[int] = set()
        self._selection: list[FaceRef] = []
        self._face_colors: dict[FaceRef, str] = {}
        self._overlay: list = []  # arrows etc.
        self._mesh_actor = None
        self._result_actor = None
        self._result_extras: list = []
        self._result_points: np.ndarray | None = None
        self._result_tree: cKDTree | None = None
        self._result_field: ResultField | None = None
        self._probe_actor = None
        self.selection_enabled = True
        self._press_pos: tuple[int, int] | None = None

        # deformation animation: surface points = base + factor(t) * delta
        self._anim_surface: pv.PolyData | None = None
        self._anim_base: np.ndarray | None = None
        self._anim_delta: np.ndarray | None = None
        self._anim_symmetric = False  # modes swing both ways, static loads go 0 -> 1
        self._anim_phase = 0.0
        self._animated = False
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)
        self._anim_timer.timeout.connect(self._anim_step)

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)

    # --- CAD model -----------------------------------------------------------------

    def load_project(self, project: Project) -> None:
        self.plotter.clear_actors()
        self._project = project
        self._meshes.clear()
        self._actors.clear()
        self._actor_part.clear()
        self._hidden.clear()
        self._selection.clear()
        self._face_colors.clear()
        self._overlay.clear()
        self._result_extras.clear()
        self._mesh_actor = self._result_actor = self._probe_actor = None
        for part in project.parts:
            mesh = tessellate(part.shape)
            # per-cell colors, so highlighting recolors cells instead of overlaying a
            # coincident mesh (which z-fights)
            mesh.cell_data["rgb"] = np.zeros((mesh.n_cells, 3), dtype=np.uint8)
            body = self.plotter.add_mesh(
                mesh, scalars="rgb", rgb=True, ambient=0.35, diffuse=0.7, specular=0.2
            )
            edges = self.plotter.add_mesh(
                # faces are tessellated separately, so their boundaries are exactly the B-rep edges
                mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                ),
                color="#2b3440",
                line_width=1.5,
                pickable=False,
            )
            self._meshes[part.id] = mesh
            self._actors[part.id] = [body, edges]
            self._actor_part[_address(body)] = part.id
        self._repaint()
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def set_part_visible(self, part_id: int, visible: bool) -> None:
        (self._hidden.discard if visible else self._hidden.add)(part_id)
        if self._result_actor is None:
            for actor in self._actors.get(part_id, []):
                actor.SetVisibility(visible)
        self.plotter.render()

    def fit(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    @property
    def selection(self) -> list[FaceRef]:
        return list(self._selection)

    def set_selection(self, refs: list[FaceRef]) -> None:
        self._selection = list(dict.fromkeys(refs))
        self._repaint()
        self.selection_changed.emit(self.selection)

    def annotate(self, fixtures: list[Fixture], loads: list[Load], bolts=()) -> None:
        """Color fixture/load/bolt faces and draw load arrows."""
        self._face_colors.clear()
        for b in bolts:
            self._face_colors[b.hole_a] = BOLT_COLOR
            self._face_colors[b.hole_b] = BOLT_COLOR
        for fx in fixtures:
            for ref in fx.faces:
                self._face_colors[ref] = FIXTURE_COLORS[fx.kind]
        for ld in loads:
            for ref in ld.faces:
                self._face_colors[ref] = LOAD_COLOR
        for actor in self._overlay:
            self.plotter.remove_actor(actor)
        self._overlay.clear()
        if self._project is not None:
            for ld in loads:
                self._draw_load_arrow(ld)
        self._repaint()

    def _face_center_normal(self, refs: list[FaceRef]) -> tuple[np.ndarray, np.ndarray]:
        centers, normals, areas = [], [], []
        for ref in refs:
            mesh = self._meshes[ref.part_id]
            cells = mesh.extract_cells(mesh.cell_data["face_id"] == ref.face_index)
            surf = cells.extract_surface(algorithm="dataset_surface").compute_normals(
                cell_normals=True, point_normals=False, auto_orient_normals=False
            )
            sizes = surf.compute_cell_sizes(length=False, volume=False)["Area"]
            centers.append((surf.cell_centers().points * sizes[:, None]).sum(0))
            normals.append((surf.cell_data["Normals"] * sizes[:, None]).sum(0))
            areas.append(sizes.sum())
        c = np.sum(centers, axis=0) / sum(areas)
        n = np.sum(normals, axis=0)
        return c, n / (np.linalg.norm(n) or 1.0)

    def _draw_load_arrow(self, load: Load) -> None:
        diag = max(bounding_diagonal(p.shape) for p in self._project.parts)
        length = 0.15 * diag
        if load.kind == LoadKind.GRAVITY:
            center = np.mean([m.center for m in self._meshes.values()], axis=0)
            direction = np.asarray(load.vector, dtype=float)
        elif load.kind in (LoadKind.FORCE, LoadKind.PRESSURE) and load.faces:
            center, normal = self._face_center_normal(load.faces)
            if load.kind == LoadKind.PRESSURE:
                direction = -normal * np.sign(load.magnitude or 1)
            else:
                direction = np.asarray(load.vector, dtype=float)
        else:
            return
        norm = np.linalg.norm(direction)
        if norm == 0:
            return
        direction = direction / norm
        arrow = pv.Arrow(start=center - direction * length, direction=direction, scale=length)
        self._overlay.append(self.plotter.add_mesh(arrow, color=LOAD_COLOR, pickable=False))

    def _repaint(self) -> None:
        selected = set(self._selection)
        for part_id, mesh in self._meshes.items():
            rgb = mesh.cell_data["rgb"]
            rgb[:] = pv.Color(PART_COLORS[part_id % len(PART_COLORS)]).int_rgb
            face_id = mesh.cell_data["face_id"]
            for ref, color in self._face_colors.items():
                if ref.part_id == part_id:
                    rgb[face_id == ref.face_index] = pv.Color(color).int_rgb
            for ref in selected:
                if ref.part_id == part_id:
                    rgb[face_id == ref.face_index] = pv.Color(SELECT_COLOR).int_rgb
            rgb.VTKObject.Modified()
        self.plotter.render()

    # --- FE mesh -------------------------------------------------------------------

    def show_mesh(self, mesh: FEMesh | None) -> None:
        if self._mesh_actor is not None:
            self.plotter.remove_actor(self._mesh_actor)
            self._mesh_actor = None
        if mesh is not None:
            surf = _surface_grid(mesh, mesh.nodes).extract_surface(
                algorithm="dataset_surface", nonlinear_subdivision=0
            )
            self._mesh_actor = self.plotter.add_mesh(
                surf, style="wireframe", color="#1f3a5a", line_width=0.6, pickable=False
            )
        self.plotter.render()

    # --- results -------------------------------------------------------------------

    def show_result(
        self,
        mesh: FEMesh,
        displacement: np.ndarray,
        field: ResultField,
        scale: float,
        symmetric: bool = False,
    ):
        self.clear_result(render=False)
        points = mesh.nodes + scale * displacement
        grid = _surface_grid(mesh, points)
        grid.point_data["value"] = field.values
        surf = grid.extract_surface(algorithm="dataset_surface", nonlinear_subdivision=2)
        # subdivision is linear in the node coordinates, so the undeformed surface has the
        # same topology and animation can interpolate point positions directly
        base = _surface_grid(mesh, mesh.nodes).extract_surface(
            algorithm="dataset_surface", nonlinear_subdivision=2
        )
        self._anim_surface = surf
        self._anim_base = np.array(base.points)
        self._anim_delta = np.array(surf.points) - self._anim_base
        self._anim_symmetric = symmetric
        self._anim_phase = 0.0
        self._result_actor = self.plotter.add_mesh(
            surf,
            scalars="value",
            cmap=field.cmap,
            clim=field.clim,
            smooth_shading=True,
            scalar_bar_args={
                "title": field.title,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.15,
                "height": 0.7,
                "fmt": "%.3g",
                "color": "#2b3440",
            },
        )
        _use_korean_font(self.plotter.scalar_bars[field.title])
        # undeformed outline for reference
        outline = (
            _surface_grid(mesh, mesh.nodes)
            .extract_surface(algorithm="dataset_surface")
            .extract_feature_edges(45)
        )
        self._result_extras.append(
            self.plotter.add_mesh(outline, color="#8894a2", line_width=1, pickable=False)
        )
        boundary = np.unique(mesh.tris)
        self._result_points = boundary
        self._result_tree = cKDTree(points[boundary])
        self._result_field = field
        for actors in self._actors.values():
            for actor in actors:
                actor.SetVisibility(False)
        for actor in self._overlay:
            actor.SetVisibility(False)
        if self._animated:
            self._anim_timer.start()
        self.plotter.render()

    def set_animated(self, on: bool) -> None:
        self._animated = on
        if on and self._result_actor is not None:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
            if self._anim_surface is not None:
                self._anim_surface.points = self._anim_base + self._anim_delta
                self.plotter.render()

    def _anim_step(self) -> None:
        if self._anim_surface is None:
            return
        self._anim_phase += 0.2
        if self._anim_symmetric:
            factor = np.sin(self._anim_phase)
        else:
            factor = 0.5 * (1 - np.cos(self._anim_phase))
        self._anim_surface.points = self._anim_base + factor * self._anim_delta
        self.plotter.render()

    def clear_result(self, render: bool = True) -> None:
        self._anim_timer.stop()
        self._anim_surface = None
        if self._result_actor is not None:
            self.plotter.remove_actor(self._result_actor)
            self.plotter.remove_scalar_bar()
            self._result_actor = None
        for actor in self._result_extras:
            self.plotter.remove_actor(actor)
        self._result_extras.clear()
        if self._probe_actor is not None:
            self.plotter.remove_actor(self._probe_actor)
            self._probe_actor = None
        self._result_field = None
        for part_id, actors in self._actors.items():
            for actor in actors:
                actor.SetVisibility(part_id not in self._hidden)
        for actor in self._overlay:
            actor.SetVisibility(True)
        if render:
            self.plotter.render()

    def _probe(self, x: int, y: int) -> None:
        if not self._picker.Pick(x, y, 0, self.plotter.renderer):
            return
        if self._picker.GetActor() is None or _address(self._picker.GetActor()) != _address(
            self._result_actor
        ):
            return
        pos = np.array(self._picker.GetPickPosition())
        _, i = self._result_tree.query(pos)
        node = self._result_points[i]
        value = self._result_field.values[node]
        if self._probe_actor is not None:
            self.plotter.remove_actor(self._probe_actor)
        self._probe_actor = self.plotter.add_point_labels(
            [pos],
            [f"{value:.4g} {self._result_field.unit}"],
            point_size=12,
            point_color="#e8743b",
            font_size=14,
            shape_opacity=0.85,
            always_visible=True,
            pickable=False,
        )
        self.probed.emit(f"{self._result_field.title}: {value:.4g} {self._result_field.unit}")
        self.plotter.render()

    # --- mouse ---------------------------------------------------------------------

    def _on_press(self, *_args) -> None:
        self._press_pos = self.plotter.iren.get_event_position()

    def _on_release(self, *_args) -> None:
        pos = self.plotter.iren.get_event_position()
        if self._press_pos is None:
            return
        moved = np.hypot(pos[0] - self._press_pos[0], pos[1] - self._press_pos[1])
        self._press_pos = None
        if moved > CLICK_TOLERANCE_PX:
            return
        if self._result_actor is not None:
            self._probe(*pos)
            return
        if not self.selection_enabled:
            return
        ref = self._pick(*pos)
        additive = bool(self.plotter.iren.interactor.GetControlKey())
        if additive and ref is not None:
            if ref in self._selection:
                self._selection.remove(ref)
            else:
                self._selection.append(ref)
            self.set_selection(self._selection)
        else:
            self.set_selection([ref] if ref is not None else [])

    def _pick(self, x: int, y: int) -> FaceRef | None:
        if not self._picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        actor = self._picker.GetActor()
        part_id = self._actor_part.get(_address(actor)) if actor else None
        cell = self._picker.GetCellId()
        if part_id is None or cell < 0:
            return None
        return FaceRef(part_id, int(self._meshes[part_id].cell_data["face_id"][cell]))


def _surface_grid(mesh: FEMesh, points: np.ndarray) -> pv.UnstructuredGrid:
    tris = mesh.tris[:, _TRI6_TO_VTK]
    cells = np.hstack([np.full((len(tris), 1), 6), tris]).ravel()
    types = np.full(len(tris), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, points)


def _use_korean_font(scalar_bar) -> None:
    """VTK's built-in fonts have no Hangul; render scalar bar text with the bundled font."""
    font = str(resources.files("kstruct.resources").joinpath("fonts/NotoSansKR.ttf"))
    for prop in (scalar_bar.GetTitleTextProperty(), scalar_bar.GetLabelTextProperty()):
        prop.SetFontFamily(VTK_FONT_FILE)
        prop.SetFontFile(font)


def _address(actor) -> str:
    return actor.GetAddressAsString("vtkActor")
