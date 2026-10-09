"""Analysis project data model.

A project always holds a list of parts, even when only one is analysed, so that
assemblies can be added later without changing the model shape.

Unit system (consistent, used everywhere internally): mm, N, MPa, tonne, s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from OCP.TopoDS import TopoDS_Shape


@dataclass
class FaceRef:
    """A B-rep face of a part, identified by its index in the part's face order."""

    part_id: int
    face_index: int


class ConnectionKind(Enum):
    BONDED = "bonded"  # faces glued together (tie); linear
    CONTACT = "contact"  # may slide / separate; nonlinear (later)
    BOLT = "bolt"  # bolted joint with preload (later)


@dataclass
class Connection:
    kind: ConnectionKind
    faces_a: list[FaceRef]
    faces_b: list[FaceRef]
    friction: float = 0.0


@dataclass
class Part:
    id: int
    name: str
    shape: TopoDS_Shape
    material_id: str | None = None
    visible: bool = True


@dataclass
class Project:
    source_path: Path | None = None
    parts: list[Part] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)

    def part(self, part_id: int) -> Part:
        return next(p for p in self.parts if p.id == part_id)
