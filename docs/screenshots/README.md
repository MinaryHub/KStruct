# 앱 등록용 스크린샷

스크린샷은 **영어 화면**입니다(`KSTRUCT_LANG=en`). 한국어 화면이 필요하면 아래 명령의 `ko`로 `ko/` 아래에 만듭니다.

| 폴더 | 크기 | 용도 |
|---|---|---|
| `1920x1080/` | 1920×1080 (16:9) | Microsoft Store, 웹사이트, 일반 등록 |
| `2880x1800/` | 2880×1800 (16:10) | Mac App Store(1440×900의 2배 해상도) |

| 파일 | 장면 | 설명 문구(예시) |
|---|---|---|
| `01_welcome.png` | 시작 화면 | CAD 파일을 열고 7단계만 따라 하면 부품이 견디는지 알려 줍니다. |
| `02_setup.png` | 하중 지정 | 3D 화면에서 면을 클릭해 고정과 하중을 지정합니다. |
| `03_stress.png` | 강도 결과(응력) | 응력 분포와 함께 "안전·주의·위험"을 쉬운 말로 판정합니다. |
| `04_safety.png` | 안전율 | 어디가 얼마나 여유가 있는지 안전율 지도로 보여 줍니다. |
| `05_bolts.png` | 볼트 체결·접촉 | 볼트 예압과 판 사이 접촉까지 계산합니다. |
| `06_modal.png` | 진동 | 고유진동수와 진동 모양, 운전 속도와의 공진 여부를 확인합니다. |
| `07_thermal.png` | 열 | 열 입력·대류·복사로 온도 분포를 구합니다. |
| `08_nonlinear.png` | 항복 이후(비선형) | 재료가 항복한 뒤의 영구 변형과 무너지는 하중까지 계산합니다. |
| `09_dark_theme.png` | 어두운 화면 | 밝은/어두운 화면을 고를 수 있습니다. |

예제 모델(브래킷 S355 3 kN, 볼트 체결 판 S355 + M10 10.9 세 개)로 실제 해석한 화면이며, 다시 만들려면:

```
KSTRUCT_LANG=en QT_SCALE_FACTOR=1.5 xvfb-run -a uv run python packaging/store_screenshots.py 03_stress docs/screenshots/1920x1080/03_stress.png 1280 720
KSTRUCT_LANG=en QT_SCALE_FACTOR=2   xvfb-run -a uv run python packaging/store_screenshots.py 03_stress docs/screenshots/2880x1800/03_stress.png 1440 900
```

장면 이름: `01_welcome 02_setup 03_stress 04_safety 05_bolts 06_modal 07_thermal 08_nonlinear`, 어두운 화면은 장면 이름 뒤에 `_dark`(예: `03_stress_dark` → `09_dark_theme.png`).

전부 다시 만들기:

```
xvfb-run -a -s "-screen 0 3000x2000x24" ./packaging/all_store_screenshots.sh      # 영어 → 이 폴더
xvfb-run -a -s "-screen 0 3000x2000x24" ./packaging/all_store_screenshots.sh ko   # 한국어 → ko/
```

| File | Caption (English, example) |
|---|---|
| `01_welcome.png` | Open a CAD file and follow seven steps to find out whether the part holds. |
| `02_setup.png` | Click faces in the 3D view to place supports and loads. |
| `03_stress.png` | Stress distribution with a plain-language verdict: safe, caution or unsafe. |
| `04_safety.png` | A safety-factor map shows where the part has margin and how much. |
| `05_bolts.png` | Bolt preload and contact between plates are included. |
| `06_modal.png` | Natural frequencies, mode shapes and resonance with the operating speed. |
| `07_thermal.png` | Temperature distribution from heat input, convection and radiation. |
| `08_nonlinear.png` | Beyond yield: permanent deformation and the load at which the part gives way. |
| `09_dark_theme.png` | Light and dark appearance. |

