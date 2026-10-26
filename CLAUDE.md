# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

KStruct: 기계 부품용 3D 유한요소해석 데스크톱 앱(비전문가 대상, 상용 배포 목표). 사용자와의 대화와 UI 문구는 한국어.

## 명령어

uv로 관리하며 Python 3.12 고정(OCP·Netgen 휠 호환 때문에 3.13+ 사용 금지).

```
uv sync                                   # 의존성 설치
uv run kstruct [model.step|model.kst]     # 앱 실행
uv run kstruct --selftest                 # 화면 없이 메싱+해석 점검 (배포본 검증용)
uv run kstruct --selftest-gui out.png     # 창 열고 예제 해석 후 스크린샷
uv run pytest                             # 전체 테스트 (~1분, Qt 창이 뜸)
uv run pytest tests/test_fem_static.py::test_cantilever_tip_deflection   # 단일 테스트
uv run ruff check --fix src tests && uv run ruff format src tests
uv run pyinstaller packaging/kstruct.spec --noconfirm   # dist/KStruct/ 생성 후 --selftest로 확인
```

CI(`.github/workflows/build.yml`)가 Windows/macOS/Linux에서 테스트 → PyInstaller → 배포본 selftest → 설치 파일(Inno Setup / dmg / tar.gz)을 만든다. 로컬 Windows에서는 Inno Setup 설치가 실패해 설치 파일 단계는 CI에서만 검증됨.

## 아키텍처

데이터 흐름: `geometry`(CAD/메시 → `Project`) → `fem.mesh`(Netgen TET10) → `fem.static|modal|thermal|buckling` → `report.summary`(판정·쉬운 설명) → `ui`/`report.pdf`.

- **단위계**: 내부는 mm·N·MPa·tonne·s 고정. 밀도 t/mm³(재료 JSON은 kg/m³로 저장하고 로드 시 변환), 열량 1 W = 1000 N·mm/s, 대류 h[W/m²K]×1e-3. 변환은 `standards/__init__.py`와 `fem/thermal.py`에만 있다.
- **면 식별(FaceRef)**: `geometry.faces_of(shape)`의 리스트 인덱스가 `FaceRef.face_index`. Netgen의 면 번호도 같은 순서임이 테스트로 보장된다. 뷰어 삼각형의 `cell_data["face_id"]`, 메시 경계 삼각형의 `FEMesh.tri_face`가 모두 이 인덱스라서 클릭 → 고정/하중 → 메시 절점이 연결된다. 면 순서를 바꾸는 형상 처리를 하면 기존 FaceRef가 깨진다.
- **Netgen**은 자체 OCCT 사본을 쓰므로 부품을 STEP 임시 파일로 넘긴다(`fem/mesh.py`). `mesh.Coordinates()`는 Netgen 메모리 뷰라 반드시 복사할 것(안 하면 다음 부품 메싱 때 좌표가 깨짐).
- **솔버는 자체 구현**(GPL 회피): `fem/tet10.py`가 요소 수식, `fem/assembly.py`가 조립·선형해(작으면 SuperLU, 크면 pyamg CG). SuperLU는 반드시 `MMD_AT_PLUS_A` + `SymmetricMode` + `DiagPivotThresh=0`(SPD라 대각 피벗 안전). 기본 부분 피벗은 페널티 tie가 있으면 채움이 30배로 폭증한다. 여러 우변은 `make_solver`로 한 번 분해 후 재사용. 경계조건(`fem/constraints.py`)은 축 정렬 구속은 DOF 제거, 경사 미끄럼/핀/부품 간 접합(tie)은 페널티 방정식. 반력은 각 고정이 실제로 구속한 DOF/방정식만 합산한다(모서리 공유 DOF 이중 집계 방지).
- **열응력**: 정적해석에 열 조건(`THERMAL_LOADS`)이 있으면 먼저 `solve_thermal`로 온도장을 구하고 `reference_temperature` 대비 ΔT로 열하중을 만든다. 피로(`fem/fatigue.py`)는 정적 결과를 반복 하중의 최대값으로 보고 `report.summary.summarize`에서 계산된다.
- **접촉**(`fem/contact.py`): 마찰 없는 node-to-surface 페널티 + 능동 집합 반복. 해제/결합 판정은 틈(gap)이 아니라 스프링 힘 부호로 한다(페널티가 강해 인장 쌍의 틈이 반올림 수준이라서). 접촉으로만 붙은 부품을 위해 1e-9 크기 안정화 스프링을 쓰고, 그 부품이 미끄러지면 `SolverError("contact unstable")`. 진동·좌굴·열 해석에서는 접촉을 접합으로 취급.
- **볼트**(`fem/bolts.py`): 머리·너트 아래 고리 면(없으면 구멍 벽)의 평균 변위를 잇는 축/전단 스프링. 예압은 "잠금" 방식: 예압만 걸린 상태에서 각 볼트가 정확히 P가 되도록 당김력 T를 `locked_pulls`로 풀고, 축력은 항상 모델 안의 실제 힘 T + k·신장량으로 보고한다(평형 보장). 예압이 있으면 두 부품 사이에 접촉/접합 연결이 필수.
- **좌굴**(`fem/buckling.py`): ARPACK `mode="buckling"`(K는 SPD, K_G는 부정치). 압축이 거의 없으면 반복 상한에서 끊고 "좌굴 없음"으로 처리.
- **표시 단위**(`units.py`): 계산은 항상 SI 내부 단위. 입력·목록·결과 필드·보고서만 `system_of(project)`(프로젝트 설정, 없으면 기준 팩의 `units`)로 변환한다.
- **여러 부품**: 처음부터 `Project.parts` 리스트 + `connections`. STEP의 솔리드마다 Part 하나, 맞닿은 면은 `geometry/contacts.py`가 자동 접합(바깥 법선이 마주보고 실제 겹침 면적이 있어야 함).
- **메시 포맷**(`geometry/mesh_import.py`, `geometry/fbx.py`): 삼각형을 공유 위상 B-rep 솔리드로 만들고 같은 평면 삼각형을 병합. 열린 메시는 자동 보정하지 않고 거부한다(보정이 형상을 몰래 바꾸므로). FBX 리더는 자체 구현.
- **기준 팩/재료**: `resources/standards.json`, `resources/materials.json`. 국가별 명칭·권장 안전율(좌굴 하중 배수 포함)은 데이터로만 추가한다. 계산 코드에 국가 기준을 하드코딩하지 말 것. 볼트 규격·등급은 `resources/bolts.json`.
- **도움말**: 모든 UI 요소는 `resources/help/ko.json`의 키를 참조(`HelpButton`, `with_help`, `topic(key).short`). `tests/test_ui.py`가 UI 코드에 쓰인 키가 JSON에 다 있는지 검사하므로 새 접두어를 쓰면 그 테스트의 `PREFIXES`도 갱신.
- **UI**: `ui/main_window.py`가 Project·mesh·result를 소유하고, `ui/steps.py`의 7개 패널이 이를 편집해 `changed`를 보낸다. 메싱·해석은 `ui/worker.run_job`으로 백그라운드 실행. 결과 종류(Static/Modal/Thermal)에 따라 결과 패널 필드가 바뀐다.
- **3D 텍스트 한글**: VTK 기본 글꼴에 한글이 없어 색 막대에 번들 Noto Sans KR(OFL)을 지정한다(`viewer._use_korean_font`).
- **결과 표시 주의**: 2차 요소 보간이 범위를 벗어나 거짓 색을 만들 수 있어 안전율 같은 잘린 값은 표시 전에 clim으로 자른다.
- **응력 특이점**(`fem/singularity.py`): 필렛 없는 오목 모서리(경계 삼각형 쌍의 이면각으로 판정)와 완전 고정면 가장자리 절점을 표시하고, 그 주변(요소 1.5개 또는 부품 대각선 3% 중 큰 값)을 제외한 "특이점 제외(핫스팟)" 응력·안전율을 `report.summary`가 함께 보고한다. 연성재 판정은 핫스팟 값(`PartSummary.governing_fos`), 취성재는 최대값 기준. 예제 브래킷의 판정이 메시 등급과 무관하게 같아야 한다(`tests/test_improvements.py`).
- **숨긴 부품**: 해석·메시·자동 접합은 항상 `model.active_project(project)`(보이는 부품 + 그 부품만 참조하는 고정·하중·연결·볼트)를 쓴다. 숨긴 부품에 걸린 조건은 6단계 체크리스트에 "제외" 표시.
- **권장 안전율·사용자 재료**: 판정 기준은 `standards.effective_pack(project)`(`Project.target_overrides` 반영), 재료 조회는 `standards.project_materials(project)`(내장 + `Project.custom_materials`). `standards.pack()`/`materials()`를 UI·요약·보고서에서 직접 쓰지 말 것. 사용자 재료는 .kst에 저장되고 `paths.user_materials_path()` 라이브러리에도 복사된다.
- **하중 케이스**: `Project.cases`(`LoadCase`) + `active_case`. `project.loads`는 활성 케이스의 리스트(프로퍼티)라 기존 코드가 그대로 동작하고, 강도 해석은 `solve_cases`가 `with_case(project, i)`(얕은 뷰, 복사 없음)로 모든 케이스를 푼다. UI는 `case_results`에 케이스별 (이름, 결과, 요약)을 들고 `select_case`로 바꿔 보여주며, 열·좌굴은 활성 케이스만 쓴다. 프로젝트 파일 v2는 `cases`를 저장하고 v1의 평면 `loads`도 읽는다.
- **회전 하중**(`fem/loads.py`): 토크는 선택 면에 반지름에 비례하는 접선 트랙션(합 모멘트 = T 정확), 베어링 하중은 구멍 원통면의 하중 방향 절반에 코사인 분포(합력 = F가 되도록 잔차를 보정), 원심력은 ρω²r 체적력. 축은 `Load.point` + `vector`(토크·원심력, 없으면 선택 원통면의 축), 베어링은 항상 원통면 축. 내부 단위 N·mm, rad/s이며 표시는 `units`의 `torque`(N·m / lbf·in)·`speed`(rpm).
- **조건 문구**: 고정·하중·연결 설명은 `report/describe.py` 한 곳에서 만들어 패널과 PDF가 같은 문장을 쓴다. 보고서 수치는 모두 `units.fmt`를 거친다.
- **로그·진단**(`paths.py`): `setup_logging()`이 사용자 데이터 폴더(`KSTRUCT_DATA_DIR`로 재지정 가능)에 회전 로그를 쓰고, `install_excepthook` + `MainWindow.install_crash_dialog`가 처리되지 않은 예외를 로그·대화상자로 보낸다. "문제 보고"는 로그+환경+.kst를 zip으로 묶는다(`ui/dialogs.py`). 버전은 `kstruct.__version__` 하나이며 pyproject와 일치하는지 테스트가 확인한다.

## OCP(cadquery-ocp 8.x, OCCT 8) 주의

정적 메서드는 대개 `_s` 접미사지만 예외가 있다(`TopoDS.Face`, `Bnd_Box.CornerMin`). `TopTools_IndexedMapOfShape` 대신 `OCP.collections.IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher`. 모르면 `dir()`로 확인. OCCT 표준출력 로그는 `geometry/__init__.py`에서 끈다.

## 패키징 주의

`packaging/kstruct.spec`: delvewheel `*.libs` 폴더와 `netgen-occt`의 DLL(venv의 bin에 설치됨)을 직접 포함하고, Netgen이 런타임에 잘못된 경로를 찾지 않도록 `netgen_occt-*.dist-info`는 번들에서 제외한다. matplotlib은 PyVista가 import하므로 제외하면 안 된다.

## 결과물 원칙

결과 화면·보고서에는 "참고용, 최종 판단은 자격 있는 기술자" 고지를 유지한다. 새 해석 기능은 이론해와 비교하는 테스트를 함께 추가한다(`tests/test_fem_*.py` 참고).
