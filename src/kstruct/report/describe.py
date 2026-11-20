"""Plain-language, unit-aware descriptions of fixtures, loads and connections.

Shared by the wizard panels and the PDF report so both always say the same thing.
"""

from __future__ import annotations

from collections.abc import Callable

from kstruct.i18n import N_, tr, tr_data
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
    FixtureKind.FIXED: N_("완전 고정"),
    FixtureKind.ROLLER: N_("미끄럼 지지"),
    FixtureKind.PIN: N_("핀 지지"),
    FixtureKind.DISPLACEMENT: N_("강제 변위"),
    FixtureKind.ELASTIC: N_("탄성 지지"),
}
CONNECTION_NAMES = {
    ConnectionKind.BONDED: N_("접합"),
    ConnectionKind.CONTACT: N_("접촉"),
    ConnectionKind.BOLT: N_("볼트"),
}

FaceNamer = Callable[[list[FaceRef]], str]


def plain_faces(project: Project) -> FaceNamer:
    """Face names without geometry lookups (used by the report)."""
    multi = len(project.parts) > 1

    def name(refs: list[FaceRef]) -> str:
        return ", ".join(
            tr("{part} 면#{n}", part=tr_data(project.part(r.part_id).name), n=r.face_index + 1)
            if multi
            else tr("면#{n}", n=r.face_index + 1)
            for r in refs
        )

    return name


def describe_fixture(project: Project, fx: Fixture, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    system = system_of(project)
    detail = ""
    if fx.kind == FixtureKind.DISPLACEMENT:
        parts = [
            f"{axis} {fmt(system, 'length', d, '+.4g')}"
            for axis, d in zip("XYZ", fx.displacement, strict=True)
            if d is not None
        ]
        detail = f" ({', '.join(parts)})" if parts else ""
    elif fx.kind == FixtureKind.ELASTIC:
        detail = f" ({fmt(system, 'stiffness', fx.stiffness)})"
    return f"{tr(FIXTURE_NAMES[fx.kind])}{detail}: {faces(fx.faces)}"


def describe_connection(project: Project, c: Connection, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    name = tr(CONNECTION_NAMES[c.kind])
    if c.kind == ConnectionKind.CONTACT:
        name = (
            tr("접촉 (마찰 μ={mu:g})", mu=c.friction) if c.friction > 0 else tr("접촉 (마찰 없음)")
        )
    return f"{name}: {faces(c.faces_a)} ↔ {faces(c.faces_b)}"


def describe_load(project: Project, ld: Load, faces: FaceNamer | None = None) -> str:
    faces = faces or plain_faces(project)
    system = system_of(project)
    if ld.kind == LoadKind.FORCE:
        fq = quantity(system, "force")
        fx, fy, fz = (fq.show(c) for c in ld.vector)
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        return tr(
            "힘 {mag:,.1f} {unit} ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}): {faces}",
            mag=mag,
            unit=fq.unit,
            fx=fx,
            fy=fy,
            fz=fz,
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.PRESSURE:
        return tr(
            "압력 {value}: {faces}",
            value=fmt(system, "stress", ld.magnitude),
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.GRAVITY:
        g = (sum(c * c for c in ld.vector) ** 0.5) / GRAVITY
        return tr("자중 {g:.2g} g", g=g)
    if ld.kind == LoadKind.TORQUE:
        return tr(
            "토크 {value}{axis}: {faces}",
            value=fmt(system, "torque", ld.magnitude),
            axis=_axis(ld),
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.BEARING:
        fq = quantity(system, "force")
        fx, fy, fz = (fq.show(c) for c in ld.vector)
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        return tr(
            "베어링 하중 {mag:,.1f} {unit} ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}): {faces}",
            mag=mag,
            unit=fq.unit,
            fx=fx,
            fy=fy,
            fz=fz,
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.REMOTE:
        fq = quantity(system, "force")
        fx, fy, fz = (fq.show(c) for c in ld.vector)
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        lq = quantity(system, "length")
        px, py, pz = (lq.show(c) for c in (ld.point or (0.0, 0.0, 0.0)))
        return tr(
            "원격 하중 {mag:,.1f} {unit} ({fx:+.0f}, {fy:+.0f}, {fz:+.0f}), "
            "작용점 ({px:.4g}, {py:.4g}, {pz:.4g}) {length}: {faces}",
            mag=mag,
            unit=fq.unit,
            fx=fx,
            fy=fy,
            fz=fz,
            px=px,
            py=py,
            pz=pz,
            length=lq.unit,
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.MOMENT:
        mq = quantity(system, "torque")
        mx, my, mz = (mq.show(c) for c in ld.vector)
        mag = (mx**2 + my**2 + mz**2) ** 0.5
        return tr(
            "모멘트 {mag:,.4g} {unit} ({mx:+.4g}, {my:+.4g}, {mz:+.4g}): {faces}",
            mag=mag,
            unit=mq.unit,
            mx=mx,
            my=my,
            mz=mz,
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.CENTRIFUGAL:
        return tr(
            "원심력 {value}{axis}",
            value=fmt(system, "speed", ld.magnitude, ",.0f"),
            axis=_axis(ld),
        )
    if ld.kind == LoadKind.FACE_TEMPERATURE:
        return tr(
            "고정 온도 {value}: {faces}",
            value=fmt(system, "temperature", ld.magnitude, ".1f"),
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.HEAT_FLOW:
        return tr(
            "열 입력 {value}: {faces}",
            value=fmt(system, "power", ld.magnitude),
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.CONVECTION:
        air = fmt(system, "temperature", ld.ambient, ".0f")
        return tr(
            "대류 h={h:.3g} W/m²K, 공기 {air}: {faces}",
            h=ld.magnitude,
            air=air,
            faces=faces(ld.faces),
        )
    if ld.kind == LoadKind.RADIATION:
        around = fmt(system, "temperature", ld.ambient, ".0f")
        return tr(
            "복사 ε={e:.2g}, 주변 {around}: {faces}",
            e=ld.magnitude,
            around=around,
            faces=faces(ld.faces),
        )
    return tr(
        "온도 변화 {value}",
        value=fmt(system, "temperature_change", ld.magnitude, "+.1f"),
    )


def _axis(ld: Load) -> str:
    norm = sum(c * c for c in ld.vector) ** 0.5
    if norm == 0:
        return tr(", 원통면의 축")
    for i, name in enumerate("XYZ"):
        c = ld.vector[i] / norm
        if abs(c) > 1 - 1e-6:  # rounding the components called (0.9, 0.44, 0) "X"
            return tr(", {name} 축", name=name) if c > 0 else tr(", −{name} 축", name=name)
    x, y, z = (c / norm for c in ld.vector)
    return tr(", 축 방향 ({x:+.2f}, {y:+.2f}, {z:+.2f})", x=x, y=y, z=z)
