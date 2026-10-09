# KStruct — Third-party software notices

KStruct includes the following open-source components. Each is used unmodified and
shipped as separate, replaceable shared libraries in the installation folder, so the
LGPL-licensed libraries can be replaced by the user with compatible versions.

| Component | License | Source |
|---|---|---|
| Qt 6 / PySide6 | LGPL-3.0 | https://code.qt.io/ , https://pypi.org/project/PySide6/ |
| Open CASCADE Technology (via cadquery-ocp, netgen-occt) | LGPL-2.1 with OCCT exception | https://dev.opencascade.org/ |
| OCP bindings (cadquery-ocp) | Apache-2.0 | https://github.com/CadQuery/OCP |
| Netgen mesher | LGPL-2.1 | https://github.com/NGSolve/netgen |
| VTK | BSD-3-Clause | https://vtk.org/ |
| PyVista, pyvistaqt | MIT | https://github.com/pyvista |
| NumPy | BSD-3-Clause | https://numpy.org/ |
| SciPy | BSD-3-Clause | https://scipy.org/ |
| PyAMG | MIT | https://github.com/pyamg/pyamg |
| trimesh | MIT | https://github.com/mikedh/trimesh |
| lxml | BSD-3-Clause | https://lxml.de/ |
| NetworkX | BSD-3-Clause | https://networkx.org/ |
| Matplotlib (colormaps) | Matplotlib License (PSF-based) | https://matplotlib.org/ |
| Noto Sans KR font | SIL Open Font License 1.1 | https://fonts.google.com/noto/specimen/Noto+Sans+KR |
| Python | PSF License | https://www.python.org/ |

The full license texts are available at the URLs above; the font license is included
at `kstruct/resources/fonts/OFL.txt`.

To obtain the exact source code of the LGPL components used in a given release, use the
versions listed in `uv.lock` of that release; they are the unmodified upstream releases.
