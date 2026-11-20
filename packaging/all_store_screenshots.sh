#!/usr/bin/env bash
# All store screenshots for one language: ./packaging/all_store_screenshots.sh ko|en
#   ko -> docs/screenshots/<size>/, en -> docs/screenshots/en/<size>/
# Linux needs a display: run under `xvfb-run -a -s "-screen 0 3000x2000x24"`.
set -euo pipefail
lang="${1:-ko}"
base="docs/screenshots"
[ "$lang" = "ko" ] || base="docs/screenshots/$lang"
scenes=(01_welcome 02_setup 03_stress 04_safety 05_bolts 06_modal 07_thermal 08_nonlinear)
shoot() {  # scale, width, height, folder
  mkdir -p "$base/$4"
  for s in "${scenes[@]}"; do
    KSTRUCT_LANG="$lang" QT_SCALE_FACTOR="$1" \
      uv run python packaging/store_screenshots.py "$s" "$base/$4/$s.png" "$2" "$3"
  done
  KSTRUCT_LANG="$lang" QT_SCALE_FACTOR="$1" \
    uv run python packaging/store_screenshots.py 03_stress_dark "$base/$4/09_dark_theme.png" "$2" "$3"
}
shoot 1.5 1280 720 1920x1080
shoot 2 1440 900 2880x1800
