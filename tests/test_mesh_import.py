import struct
import zlib

import numpy as np
import pytest
import trimesh

from kstruct.geometry import GeometryImportError, faces_of, load_cad
from kstruct.geometry.face_info import face_info
from kstruct.geometry.fbx import BINARY_MAGIC, read_fbx


def _volume(project) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(project.parts[0].shape, props)
    return props.Mass()


@pytest.mark.parametrize(
    ("suffix", "scale"), [(".stl", 1), (".obj", 1), (".3mf", 1), (".glb", 1e-3)]
)
def test_box_mesh_becomes_six_faced_solid(tmp_path, suffix, scale):
    box = trimesh.creation.box(extents=np.array([10, 20, 30]) * scale)
    path = tmp_path / f"box{suffix}"
    box.export(path)
    project = load_cad(path)
    assert len(project.parts) == 1
    faces = faces_of(project.parts[0].shape)
    assert len(faces) == 6  # coplanar triangles merged
    assert sorted(round(face_info(f).area) for f in faces) == [200, 200, 300, 300, 600, 600]
    assert _volume(project) == pytest.approx(6000, rel=1e-6)


def test_open_mesh_is_rejected(tmp_path):
    box = trimesh.creation.box(extents=[10, 10, 10])
    box.update_faces(np.arange(len(box.faces)) > 0)  # punch out one triangle
    box.remove_unreferenced_vertices()
    path = tmp_path / "open.stl"
    box.export(path)
    with pytest.raises(GeometryImportError, match="watertight"):
        load_cad(path)


def test_cylinder_mesh_meshes_and_solves(tmp_path):
    from kstruct.fem import generate_mesh

    cyl = trimesh.creation.cylinder(radius=5, height=40, sections=24)
    path = tmp_path / "cyl.stl"
    cyl.export(path)
    project = load_cad(path)
    mesh = generate_mesh(project, "coarse")
    assert len(mesh.tets) > 0


# --- FBX ------------------------------------------------------------------------------

CUBE_V = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)
# quads, last index of each polygon bit-inverted (FBX convention), outward winding
CUBE_Q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
CUBE_I = [i for q in CUBE_Q for i in (*q[:3], ~q[3])]


def _ascii_fbx() -> str:
    verts = ",".join(f"{v:g}" for v in (CUBE_V * 2).ravel())  # 2 cm cube
    idx = ",".join(str(i) for i in CUBE_I)
    return f"""; FBX 7.4.0 project file
FBXHeaderExtension:  {{ FBXVersion: 7400 }}
GlobalSettings:  {{
    Properties70:  {{
        P: "UnitScaleFactor", "double", "Number", "",1
    }}
}}
Objects:  {{
    Geometry: 100, "Geometry::Cube", "Mesh" {{
        Vertices: *24 {{ a: {verts} }}
        PolygonVertexIndex: *24 {{ a: {idx} }}
    }}
    Model: 200, "Model::Block", "Mesh" {{
        Properties70:  {{
            P: "Lcl Translation", "Lcl Translation", "", "A",5,0,0
        }}
    }}
}}
Connections:  {{
    C: "OO",100,200
    C: "OO",200,0
}}
"""


def _binary_fbx() -> bytes:
    """Version 7500 binary FBX with the same content (zlib-compressed arrays)."""

    def prop(v) -> bytes:
        if isinstance(v, str):
            raw = v.encode()
            return b"S" + struct.pack("<I", len(raw)) + raw
        if isinstance(v, int):
            return b"L" + struct.pack("<q", v)
        if isinstance(v, float):
            return b"D" + struct.pack("<d", v)
        arr = np.asarray(v)
        code, dtype = (b"d", "<f8") if arr.dtype.kind == "f" else (b"i", "<i4")
        data = zlib.compress(arr.astype(dtype).tobytes())
        return code + struct.pack("<III", len(arr), 1, len(data)) + data

    def node(name, props=(), children=(), offset=0) -> bytes:
        pbytes = b"".join(prop(p) for p in props)
        body_start = offset + 25 + len(name) + len(pbytes)
        kids, pos = b"", body_start
        for child in children:
            k = child(pos)
            kids += k
            pos += len(k)
        if children:
            kids += b"\x00" * 25
        end = body_start + len(kids)
        head = (
            struct.pack("<QQQ", end, len(props), len(pbytes)) + bytes([len(name)]) + name.encode()
        )
        return head + pbytes + kids

    def n(name, props=(), children=()):
        return lambda offset: node(name, props, children, offset)

    p70 = lambda *ps: n("Properties70", (), [n("P", p) for p in ps])  # noqa: E731
    top = [
        n("GlobalSettings", (), [p70(("UnitScaleFactor", "double", "Number", "", 1.0))]),
        n(
            "Objects",
            (),
            [
                n(
                    "Geometry",
                    (100, "Cube\x00\x01Geometry", "Mesh"),
                    [
                        n("Vertices", ((CUBE_V * 2).ravel(),)),
                        n("PolygonVertexIndex", (np.array(CUBE_I),)),
                    ],
                ),
                n(
                    "Model",
                    (200, "Block\x00\x01Model", "Mesh"),
                    [p70(("Lcl Translation", "Lcl Translation", "", "A", 5.0, 0.0, 0.0))],
                ),
            ],
        ),
        n("Connections", (), [n("C", ("OO", 100, 200)), n("C", ("OO", 200, 0))]),
    ]
    out = BINARY_MAGIC + b"\x1a\x00" + struct.pack("<I", 7500)
    for t in top:
        out += t(len(out))
    return out + b"\x00" * 25


@pytest.mark.parametrize("kind", ["ascii", "binary"])
def test_fbx_reader(tmp_path, kind):
    path = tmp_path / "cube.fbx"
    if kind == "ascii":
        path.write_text(_ascii_fbx())
    else:
        path.write_bytes(_binary_fbx())
    meshes = read_fbx(path)
    assert len(meshes) == 1
    m = meshes[0]
    assert m.name == "Block"
    assert len(m.triangles) == 12
    # 2 cm cube moved 5 cm along x, in mm
    assert m.vertices.min(axis=0) == pytest.approx([50, 0, 0])
    assert m.vertices.max(axis=0) == pytest.approx([70, 20, 20])

    project = load_cad(path)
    assert len(faces_of(project.parts[0].shape)) == 6
    assert _volume(project) == pytest.approx(8000, rel=1e-6)


def test_garbage_fbx(tmp_path):
    path = tmp_path / "bad.fbx"
    path.write_bytes(BINARY_MAGIC + b"\x1a\x00" + struct.pack("<I", 7500) + b"\xff" * 40)
    with pytest.raises(GeometryImportError):
        load_cad(path)
