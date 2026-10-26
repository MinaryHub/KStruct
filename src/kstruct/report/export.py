"""Export results as a CSV table (node values) or a VTU file (for ParaView etc.).

Values are written in the project's display units; the CSV header names the unit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kstruct.fem.mesh import FEMesh
from kstruct.units import quantity, system_of


def result_columns(project, result, summary=None) -> list[tuple[str, np.ndarray]]:
    """(column name with unit, per-node values) for the result's kind."""
    from kstruct.fem.buckling import BucklingResult
    from kstruct.fem.modal import ModalResult
    from kstruct.fem.static import StaticResult
    from kstruct.fem.thermal import ThermalResult

    us = system_of(project)
    length, stress, temp = (quantity(us, k) for k in ("length", "stress", "temperature"))
    mesh: FEMesh = result.mesh
    cols: list[tuple[str, np.ndarray]] = [
        (f"x [{length.unit}]", length.show(mesh.nodes[:, 0])),
        (f"y [{length.unit}]", length.show(mesh.nodes[:, 1])),
        (f"z [{length.unit}]", length.show(mesh.nodes[:, 2])),
    ]
    if isinstance(result, ThermalResult):
        cols.append((f"T [{temp.unit}]", temp.show(result.temperature)))
        cols.append(("heat flux [W/m2]", result.heat_flux_magnitude))
        return cols
    if isinstance(result, ModalResult):
        for i, (f, shape) in enumerate(zip(result.frequencies, result.shapes, strict=True)):
            cols.append((f"mode {i + 1} ({f:.2f} Hz) |phi| [-]", np.linalg.norm(shape, axis=1)))
        return cols
    if isinstance(result, BucklingResult):
        for i, (lam, shape) in enumerate(zip(result.factors, result.shapes, strict=True)):
            cols.append(
                (f"buckling mode {i + 1} (x{lam:.3f}) |phi| [-]", np.linalg.norm(shape, axis=1))
            )
        result = result.static
    assert isinstance(result, StaticResult)
    u = result.displacement
    for k, name in enumerate("xyz"):
        cols.append((f"u{name} [{length.unit}]", length.show(u[:, k])))
    cols.append((f"|u| [{length.unit}]", length.show(result.displacement_magnitude)))
    for k, name in enumerate(("sxx", "syy", "szz", "sxy", "syz", "szx")):
        cols.append((f"{name} [{stress.unit}]", stress.show(result.stress[:, k])))
    cols.append((f"von Mises [{stress.unit}]", stress.show(result.von_mises)))
    cols.append((f"max principal [{stress.unit}]", stress.show(result.principal[:, 0])))
    cols.append(("safety factor [-]", result.safety_factor))
    if result.thermal is not None:
        cols.append((f"T [{temp.unit}]", temp.show(result.thermal.temperature)))
    if summary is not None and getattr(summary, "fatigue", None) is not None:
        cols.append(("fatigue safety factor [-]", summary.fatigue.safety_factor))
        cols.append(("fatigue life [cycles]", summary.fatigue.life))
    return cols


def export_csv(path: str | Path, project, result, summary=None) -> Path:
    path = Path(path)
    cols = result_columns(project, result, summary)
    data = np.column_stack([np.asarray(v, dtype=float) for _, v in cols])
    header = ",".join(name for name, _ in cols)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")
    return path


def export_vtu(path: str | Path, project, result, summary=None) -> Path:
    """Quadratic tetrahedra with every column as point data (ParaView, VisIt, PyVista)."""
    from kstruct.ui.viewer import volume_grid

    path = Path(path)
    grid = volume_grid(result.mesh, result.mesh.nodes)
    for name, values in result_columns(project, result, summary):
        grid.point_data[name.split(" [")[0]] = np.asarray(values, dtype=float)
    grid.save(str(path))
    return path
