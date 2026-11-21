# PyInstaller spec for KStruct (Windows, macOS, Linux).
# Build:  uv run pyinstaller packaging/kstruct.spec --noconfirm
# Output: dist/KStruct/  (one-folder build; wrap with an installer per OS)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []
for package in ("OCP", "netgen", "pyngcore", "vtkmodules", "pyvista", "pyvistaqt", "trimesh", "pyamg"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files("kstruct", includes=["resources/**/*"])
datas.append((str(ROOT / "THIRD_PARTY_NOTICES.md"), "."))  # shown in the About dialog
hiddenimports += collect_submodules("kstruct")
if (ROOT / "src" / "kstruct" / "_build.py").is_file():  # version stamp from packaging/build.py
    hiddenimports.append("kstruct._build")
else:
    print("kstruct.spec: no build stamp (run packaging/build.py); freezing the plain release version")
hiddenimports += ["scipy.sparse.linalg", "scipy.spatial", "lxml.etree", "networkx"]

# delvewheel/auditwheel put shared libraries in "<dist>.libs" folders beside the packages
# (OpenCASCADE for OCP and for netgen, VTK, numpy, scipy); ship them unchanged
import site

for sp in map(Path, site.getsitepackages()):
    for libs in sp.glob("*.libs"):
        binaries += [(str(f), libs.name) for f in libs.iterdir() if f.is_file()]

# netgen-occt installs plain-named OpenCASCADE libraries outside site-packages (venv bin/lib);
# find them through the wheel RECORD and put them on the frozen app's library path
from importlib.metadata import distribution

occt = distribution("netgen-occt")
for f in occt.files or []:
    if f.suffix.lower() in (".dll", ".so", ".dylib") or ".so." in f.name:
        binaries.append((str(f.locate().resolve()), "."))

# MKL for the PARDISO solver (Windows/Linux x86-64; fem/pardiso.py loads it with ctypes,
# so PyInstaller cannot see it). Only what PARDISO loads: the runtime, core, LP64
# interface and OpenMP threading layers, every CPU-specific kernel (MKL picks one at run
# time), OpenMP and tbbmalloc. The wheels hold ~1 GB more (MPI, ScaLAPACK, GPU offload).
import re

MKL_KEEP = re.compile(
    r"^(lib)?(mkl_(rt|core|intel_lp64|intel_thread|sequential|def|mc3|avx\d*|vml_\w+)"
    r"|iomp5(md)?|tbbmalloc)[.\d]*\.(dll|so[.\d]*|dylib)$",
    re.IGNORECASE,
)
for dist in ("mkl", "intel-openmp", "tbb"):
    try:
        files = distribution(dist).files or []
    except Exception:  # not installed on this platform (macOS)
        continue
    for f in files:
        if MKL_KEEP.match(f.name) and "_db" not in f.name:
            binaries.append((str(f.locate().resolve()), "."))

a = Analysis(
    [str(ROOT / "src" / "kstruct" / "app.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter",
        "IPython",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtQuick",
        "PySide6.QtQml",
    ],
    noarchive=False,
)
# netgen looks up its OCC libraries through the netgen-occt wheel metadata, whose paths are
# wrong inside the bundle; without the metadata it falls back to the libraries shipped above
a.datas = [d for d in a.datas if "netgen_occt-" not in d[0]]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KStruct",
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="KStruct")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="KStruct.app", bundle_identifier="kr.smic.kstruct")
