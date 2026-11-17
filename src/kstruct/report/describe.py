"""Plain-language, unit-aware descriptions of fixtures, loads and connections.

Shared by the wizard panels and the PDF report so both always say the same thing.
"""

from __future__ import annotations

from collections.abc import Callable

from kstruct.model import (
    GRAVITY,
    Connection,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Project,
)
from kstruct.units import fmt, quantity, system_of

FIXTURE_NAMES = {
    FixtureKind.FIXED: "완전 고정",
    FixtureKind.ROLLER: "미끄럼 지지",
    FixtureKind.PIN: "핀 지지",
}
CONNECTION_NAMES = {
    ConnectionKind.BONDED: "접합",
    ConnectionKind.CONTACT: "접촉",
    ConnectionKind.BOLT: "볼트",
}

FaceNamer = Callable[[list[FaceRef]], str]


def plain_faces(project: Project) -> FaceNamer:
    """Face names without geometry lookups (used by the report)."""
    multi = len(project.parts) > 1

    def name(refs: list[FaceRef]) -> str:
        return ", ".join(
            (f"{project.part(r.part_id).name} " if multi else "") + f"면#{r.face_index + 1}"
            for r in refs
        )

    return name


def describe_fixture(project: Project, fx: Fixture, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    return f"{FIXTURE_NAMES[fx.kind]}: {faces(fx.faces)}"


def describe_connection(project: Project, c: Connection, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    name = CONNECTION_NAMES[c.kind]
    if c.kind == ConnectionKind.CONTACT:
        name += f" (마찰 μ={c.friction:g})" if c.friction > 0 else " (마찰 없음)"
    return f"{name}: {faces(c.faces_a)} ↔ {faces(c.faces_b)}"


def describe_load(project: Project, ld: Load, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    system = system_of(project)
    if ld.kind == LoadKind.FORCE:
        fq = quantity(system, "force")
        fx, fy, fz = (fq.show(c) for c in ld.vector)
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        return f"힘 {mag:,.1f} {fq.unit} ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}): {faces(ld.faces)}"
    if ld.kind == LoadKind.PRESSURE:
        return f"압력 {fmt(system, 'stress', ld.magnitude)}: {faces(ld.faces)}"
    if ld.kind == LoadKind.GRAVITY:
        g = (sum(c * c for c in ld.vector) ** 0.5) / GRAVITY
        return f"자중 {g:.2g} g"
    if ld.kind == LoadKind.TORQUE:
        return f"토크 {fmt(system, 'torque', ld.magnitude)}{_axis(ld)}: {faces(ld.faces)}"
    if ld.kind == LoadKind.BEARING:
        fq = quantity(system, "force")
        fx, fy, fz = (fq.show(c) for c in ld.vector)
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        return (
            f"베어링 하중 {mag:,.1f} {fq.unit} ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}): {faces(ld.faces)}"
        )
    if ld.kind == LoadKind.CENTRIFUGAL:
        return f"원심력 {fmt(system, 'speed', ld.magnitude, ',.0f')}{_axis(ld)}"
    if ld.kind == LoadKind.FACE_TEMPERATURE:
        return f"고정 온도 {fmt(system, 'temperature', ld.magnitude, '.1f')}: {faces(ld.faces)}"
    if ld.kind == LoadKind.HEAT_FLOW:
        return f"열 입력 {fmt(system, 'power', ld.magnitude)}: {faces(ld.faces)}"
    if ld.kind == LoadKind.CONVECTION:
        air = fmt(system, "temperature", ld.ambient, ".0f")
        return f"대류 h={ld.magnitude:.3g} W/m²K, 공기 {air}: {faces(ld.faces)}"
    if ld.kind == LoadKind.RADIATION:
        around = fmt(system, "temperature", ld.ambient, ".0f")
        return f"복사 ε={ld.magnitude:.2g}, 주변 {around}: {faces(ld.faces)}"
    return f"온도 변화 {fmt(system, 'temperature_change', ld.magnitude, '+.1f')}"


def _axis(ld: Load) -> str:
    norm = sum(c * c for c in ld.vector) ** 0.5
    if norm == 0:
        return ", 원통면의 축"
    for i, name in enumerate("XYZ"):
        c = ld.vector[i] / norm
        if abs(c) > 1 - 1e-6:  # rounding the components called (0.9, 0.44, 0) "X"
            return f", {name} 축" if c > 0 else f", −{name} 축"
    x, y, z = (c / norm for c in ld.vector)
    return f", 축 방향 ({x:+.2f}, {y:+.2f}, {z:+.2f})"
