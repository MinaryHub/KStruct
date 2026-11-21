"""Render the logo (src/kstruct/resources/icons/kstruct.svg) to every icon file:

    uv run python packaging/make_icons.py

- src/kstruct/resources/icons/kstruct_<size>.png   window / taskbar icon, header logo
- packaging/windows/kstruct.ico                     KStruct.exe and the installer
- packaging/macos/kstruct.icns                      KStruct.app
- docs/logo.png                                     README

Rendered with Qt's SVG renderer (SVG Tiny: no clipPath, filters or CSS), so the SVG
must stay within that subset; the PNGs are committed so builds need no SVG support.
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "src" / "kstruct" / "resources" / "icons" / "kstruct.svg"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def main() -> None:
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice, QRectF
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    app = QGuiApplication(sys.argv)  # noqa: F841
    renderer = QSvgRenderer(str(SVG))
    if not renderer.isValid():
        raise SystemExit(f"cannot read {SVG}")

    images = {}
    for size in (*SIZES, 1024):
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        renderer.render(p, QRectF(0, 0, size, size))
        p.end()
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        img.save(buf, "PNG")
        images[size] = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")

    out = SVG.parent
    for size in SIZES:
        images[size].save(out / f"kstruct_{size}.png", optimize=True)
    ico_sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    images[256].save(ROOT / "packaging" / "windows" / "kstruct.ico", sizes=ico_sizes)
    images[1024].save(ROOT / "packaging" / "macos" / "kstruct.icns")
    images[512].save(ROOT / "docs" / "logo.png", optimize=True)
    print("icons written")


if __name__ == "__main__":
    main()
