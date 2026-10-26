"""Result tools: section cut, CSV/VTU export, node-order mappings."""

import numpy as np
import pytest
import pyvista as pv

from kstruct import standards
from kstruct.fem import generate_mesh, solve_static
from kstruct.fem.modal import solve_modal
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box
from kstruct.model import FaceRef, Fixture, FixtureKind, Load, LoadKind, Part, Project
from kstruct.report.export import export_csv, export_vtu, result_columns
from kstruct.report.summary import summarize
from kstruct.ui.viewer import linear_volume_grid, section_surface, volume_grid

STEEL = standards.materials()["s235"]


def _beam():
    shape = _box(0, 0, 0, 100, 20, 20)
    project = Project(parts=[Part(0, "beam", shape, "s235")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 10, 10)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (100, 10, 10)))], vector=(0, 0, -500.0))
    ]
    mesh = generate_mesh(project, "coarse")
    return project, mesh, solve_static(project, mesh, {0: STEEL})


def test_vtk_tet_node_order_puts_mid_nodes_on_the_right_edges():
    project, mesh, _ = _beam()
    grid = volume_grid(mesh, mesh.nodes)
    cell = grid.get_cell(0)
    pts = np.asarray(cell.points)
    # VTK quadratic tetra: node 4..9 on edges 01 12 20 03 13 23
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for mid, (a, b) in zip(range(4, 10), edges, strict=True):
        assert pts[mid] == pytest.approx((pts[a] + pts[b]) / 2, abs=1e-9)
    assert grid.volume == pytest.approx(100 * 20 * 20, rel=1e-6)
    linear = linear_volume_grid(mesh, mesh.nodes)
    assert linear.n_cells == 8 * len(mesh.tets)
    assert linear.volume == pytest.approx(100 * 20 * 20, rel=1e-6)


def test_section_surface_shows_interior_values():
    project, mesh, result = _beam()
    cut = section_surface(
        mesh, mesh.nodes, result.von_mises, normal=(-1, 0, 0), origin=(50, 10, 10)
    )
    assert cut.n_points > 0 and cut.points[:, 0].max() == pytest.approx(50.0, abs=1e-6)
    # the cut face carries the bending stress: positive on top, varying through the depth
    on_cut = np.isclose(cut.points[:, 0], 50.0)
    values = np.asarray(cut.point_data["value"])[on_cut]
    assert values.max() > 3 * values.min()  # top fibre vs. neutral axis (shear only)


def test_csv_and_vtu_export(tmp_path):
    project, mesh, result = _beam()
    summary = summarize(project, result, standards.effective_pack(project))
    csv = export_csv(tmp_path / "r.csv", project, result, summary)
    lines = csv.read_text("utf-8").splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["x [mm]", "y [mm]", "z [mm]"]
    assert "von Mises [MPa]" in header and "safety factor [-]" in header
    assert len(lines) == mesh.n_nodes + 1
    vm_col = header.index("von Mises [MPa]")
    values = np.array([float(line.split(",")[vm_col]) for line in lines[1:]])
    assert values.max() == pytest.approx(result.von_mises.max(), rel=1e-4)

    vtu = export_vtu(tmp_path / "r.vtu", project, result, summary)
    grid = pv.read(str(vtu))
    assert grid.n_cells == len(mesh.tets) and "von Mises" in grid.point_data
    assert np.asarray(grid.point_data["safety factor"]).min() == pytest.approx(
        result.safety_factor.min(), rel=1e-6
    )

    # US display units are honoured in the column names and values
    project.standard = "astm"
    cols = dict(result_columns(project, result))
    assert "|u| [in]" in cols and cols["von Mises [psi]"].max() == pytest.approx(
        result.von_mises.max() * 145.0377, rel=1e-4
    )


def test_modal_and_thermal_columns():
    project, mesh, _ = _beam()
    modal = solve_modal(project, mesh, {0: STEEL}, n_modes=2)
    names = [n for n, _ in result_columns(project, modal)]
    assert len(names) == 5 and names[3].startswith("mode 1 (")
