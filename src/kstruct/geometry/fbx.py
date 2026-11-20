"""Minimal FBX geometry reader (binary 7.x and ASCII).

Only what structural analysis needs: mesh vertices and polygons of every Geometry, placed
with its Model's local transforms (translation / rotation XYZ / scaling, parent chain), in
the file's unit (GlobalSettings.UnitScaleFactor, centimetres by default).
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kstruct.i18n import tr

BINARY_MAGIC = b"Kaydara FBX Binary  \x00"


class FbxError(Exception):
    """A message for the user; `internal` ones are replaced by a generic message."""

    def __init__(self, message: str, internal: bool = False) -> None:
        super().__init__(message)
        self.internal = internal


@dataclass
class Node:
    name: str
    props: list = field(default_factory=list)
    children: list[Node] = field(default_factory=list)

    def find(self, name: str) -> Node | None:
        return next((c for c in self.children if c.name == name), None)

    def all(self, name: str) -> list[Node]:
        return [c for c in self.children if c.name == name]


# --- binary ---------------------------------------------------------------------------

_ARRAY_TYPES = {b"f": "f", b"d": "d", b"l": "q", b"i": "i", b"b": "?"}
_SCALAR_TYPES = {b"Y": "h", b"C": "?", b"I": "i", b"F": "f", b"D": "d", b"L": "q"}


def _read_binary(data: bytes) -> Node:
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    head = "<QQQ" if wide else "<III"
    head_size = struct.calcsize(head)

    def read_prop(pos: int):
        t = data[pos : pos + 1]
        pos += 1
        if t in _SCALAR_TYPES:
            fmt = "<" + _SCALAR_TYPES[t]
            return struct.unpack_from(fmt, data, pos)[0], pos + struct.calcsize(fmt)
        if t in _ARRAY_TYPES:
            length, encoding, size = struct.unpack_from("<III", data, pos)
            pos += 12
            raw = data[pos : pos + size]
            if encoding == 1:
                raw = zlib.decompress(raw)
            arr = np.frombuffer(raw, dtype="<" + _ARRAY_TYPES[t], count=length)
            return arr, pos + size
        if t in (b"S", b"R"):
            (size,) = struct.unpack_from("<I", data, pos)
            pos += 4
            raw = data[pos : pos + size]
            return (raw.decode("utf-8", "replace") if t == b"S" else raw), pos + size
        raise FbxError(f"unknown FBX property type {t!r}", internal=True)

    def read_node(pos: int) -> tuple[Node | None, int]:
        end, n_props, _ = struct.unpack_from(head, data, pos)
        pos += head_size
        name_len = data[pos]
        pos += 1
        if end == 0:
            return None, pos
        name = data[pos : pos + name_len].decode("ascii", "replace")
        pos += name_len
        node = Node(name)
        for _ in range(n_props):
            value, pos = read_prop(pos)
            node.props.append(value)
        while pos < end:
            child, pos = read_node(pos)
            if child is None:
                break
            node.children.append(child)
        return node, end

    root = Node("root")
    pos = 27
    while pos < len(data) - head_size:
        node, pos = read_node(pos)
        if node is None:
            break
        root.children.append(node)
    return root


# --- ASCII ----------------------------------------------------------------------------

_TOKEN = re.compile(r'"[^"]*"|[^\s,{}:"]+|[{}:,]')


def _ascii_value(tok: str):
    if tok.startswith('"'):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _read_ascii(text: str) -> Node:
    lines = [ln.split(";", 1)[0] for ln in text.splitlines()]
    tokens = _TOKEN.findall("\n".join(lines))
    root = Node("root")
    stack = [root]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "}":
            stack.pop()
            i += 1
            continue
        if i + 1 < len(tokens) and tokens[i + 1] == ":":
            node = Node(tok)
            i += 2
            while (
                i < len(tokens)
                and tokens[i] not in ("{", "}")
                and not (i + 1 < len(tokens) and tokens[i + 1] == ":")
            ):
                if tokens[i] != ",":
                    node.props.append(_ascii_value(tokens[i]))
                i += 1
            stack[-1].children.append(node)
            if i < len(tokens) and tokens[i] == "{":
                stack.append(node)
                i += 1
            continue
        i += 1
    # ASCII arrays: "Vertices: *24 { a: 1,2,3 }" -> values live in child "a"
    _flatten_ascii_arrays(root)
    return root


def _flatten_ascii_arrays(node: Node) -> None:
    for c in node.children:
        a = c.find("a")
        if a is not None and c.props and isinstance(c.props[0], str) and c.props[0].startswith("*"):
            c.props = [np.asarray(a.props)]
            c.children.remove(a)
        _flatten_ascii_arrays(c)


# --- geometry -------------------------------------------------------------------------


def _properties70(node: Node | None) -> dict[str, list]:
    out = {}
    p70 = node.find("Properties70") if node else None
    for p in p70.all("P") if p70 else []:
        out[p.props[0]] = p.props[4:]
    return out


def _rotation(deg) -> np.ndarray:
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # FBX default rotation order XYZ (applied X first)


def _local_matrix(model: Node) -> np.ndarray:
    p = _properties70(model)
    T = np.eye(4)
    T[:3, 3] = p.get("Lcl Translation", [0, 0, 0])
    R = np.eye(4)
    R[:3, :3] = _rotation(p.get("Lcl Rotation", [0, 0, 0]))
    S = np.diag([*p.get("Lcl Scaling", [1, 1, 1]), 1.0])
    return T @ R @ S


def _object_name(node: Node) -> str:
    """Binary: 'Name\x00\x01Model'; ASCII: 'Model::Name'."""
    raw = str(node.props[1]) if len(node.props) > 1 else ""
    return raw.split(chr(0))[0].split("::")[-1]


@dataclass
class FbxMesh:
    name: str
    vertices: np.ndarray  # (n, 3) in mm
    triangles: np.ndarray  # (m, 3)


def read_fbx(path: str | Path) -> list[FbxMesh]:
    data = Path(path).read_bytes()
    try:
        return _read(data)
    except FbxError as e:
        if e.internal:
            raise FbxError(tr("FBX 파일이 손상되었거나 지원하지 않는 형식입니다.")) from e
        raise
    except (struct.error, zlib.error, IndexError, ValueError, TypeError, KeyError) as e:
        # malformed content: counts that do not divide by three, indices out of range,
        # text where a number belongs
        raise FbxError(tr("FBX 파일이 손상되었거나 지원하지 않는 형식입니다.")) from e


def _read(data: bytes) -> list[FbxMesh]:
    root = (
        _read_binary(data)
        if data.startswith(BINARY_MAGIC)
        else _read_ascii(data.decode("utf-8", "replace"))
    )

    unit_cm = _properties70(root.find("GlobalSettings")).get("UnitScaleFactor", [1.0])[0]
    to_mm = 10.0 * float(unit_cm)

    objects = root.find("Objects")
    if objects is None:
        raise FbxError(tr("FBX 파일에 형상이 없습니다."))
    models = {m.props[0]: m for m in objects.all("Model")}
    parent: dict[int, int] = {}
    conns = root.find("Connections")
    for c in conns.all("C") if conns else []:
        if c.props[0] == "OO":
            parent.setdefault(c.props[1], c.props[2])

    meshes = []
    for geom in objects.all("Geometry"):
        verts_node, index_node = geom.find("Vertices"), geom.find("PolygonVertexIndex")
        if verts_node is None or index_node is None:
            continue
        verts = np.asarray(verts_node.props[0], dtype=float).reshape(-1, 3)
        index = np.asarray(index_node.props[0], dtype=np.int64)

        world = np.eye(4)
        # the name is that of the mesh's own node (the nearest named ancestor), not of
        # the group at the top of the hierarchy, which every mesh under it shares
        node_id, name, own = parent.get(geom.props[0]), _object_name(geom), ""
        while node_id in models:
            model = models[node_id]
            own = own or _object_name(model)
            world = _local_matrix(model) @ world
            node_id = parent.get(node_id)
        name = own or name
        verts = (np.c_[verts, np.ones(len(verts))] @ world.T)[:, :3] * to_mm

        tris, poly = [], []
        for idx in index:
            if idx < 0:
                poly.append(~idx)
                tris.extend((poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1))
                poly = []
            else:
                poly.append(idx)
        if tris:
            tris = np.array(tris, dtype=np.int64)
            if tris.min() < 0 or tris.max() >= len(verts):
                raise FbxError(tr("FBX 파일의 면이 없는 꼭짓점을 가리킵니다(파일 손상)."))
            meshes.append(FbxMesh(name or "Mesh", verts, tris))
    if not meshes:
        raise FbxError(tr("FBX 파일에 메시 형상이 없습니다."))
    return meshes
