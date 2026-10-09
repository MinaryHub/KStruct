"""3D model viewer with B-rep face picking."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from kstruct.geometry import tessellate
from kstruct.model import FaceRef, Project

PART_COLORS = ["#9fb4c7", "#c7b29f", "#a8c79f", "#c79fbf", "#9fc7c3", "#c7c39f"]
HIGHLIGHT_COLOR = "#ffd23f"
CLICK_TOLERANCE_PX = 4  # press/release further apart than this is a rotate, not a click


class ModelViewer(QWidget):
    face_picked = Signal(object)  # FaceRef | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter)

        self.plotter.set_background("#f4f6f8", top="#dfe5ec")
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing("ssaa")

        self._meshes: dict[int, pv.PolyData] = {}
        self._actors: dict[int, list] = {}
        self._actor_part: dict[str, int] = {}  # vtk actor address -> part id
        self._highlight: FaceRef | None = None
        self._press_pos: tuple[int, int] | None = None

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)

    def load_project(self, project: Project) -> None:
        self.plotter.clear_actors()
        self._meshes.clear()
        self._actors.clear()
        self._actor_part.clear()
        self._highlight = None
        for part in project.parts:
            mesh = tessellate(part.shape)
            # per-cell colors, so highlighting recolors cells instead of overlaying a
            # coincident mesh (which z-fights)
            base = pv.Color(PART_COLORS[part.id % len(PART_COLORS)]).int_rgb
            mesh.cell_data["rgb"] = np.tile(np.array(base, dtype=np.uint8), (mesh.n_cells, 1))
            body = self.plotter.add_mesh(
                mesh,
                scalars="rgb",
                rgb=True,
                ambient=0.35,
                diffuse=0.7,
                specular=0.2,
                pickable=True,
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
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def set_part_visible(self, part_id: int, visible: bool) -> None:
        for actor in self._actors.get(part_id, []):
            actor.SetVisibility(visible)
        self.plotter.render()

    def fit(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def highlight(self, ref: FaceRef | None) -> None:
        if self._highlight is not None:
            self._paint(self._highlight, PART_COLORS[self._highlight.part_id % len(PART_COLORS)])
        self._highlight = ref
        if ref is not None:
            self._paint(ref, HIGHLIGHT_COLOR)
        self.plotter.render()

    def _paint(self, ref: FaceRef, color: str) -> None:
        mesh = self._meshes[ref.part_id]
        rgb = mesh.cell_data["rgb"]
        rgb[mesh.cell_data["face_id"] == ref.face_index] = pv.Color(color).int_rgb
        rgb.VTKObject.Modified()

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
        ref = self._pick(*pos)
        self.highlight(ref)
        self.face_picked.emit(ref)

    def _pick(self, x: int, y: int) -> FaceRef | None:
        if not self._picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        actor = self._picker.GetActor()
        part_id = self._actor_part.get(_address(actor)) if actor else None
        cell = self._picker.GetCellId()
        if part_id is None or cell < 0:
            return None
        return FaceRef(part_id, int(self._meshes[part_id].cell_data["face_id"][cell]))


def _address(actor) -> str:
    return actor.GetAddressAsString("vtkActor")
