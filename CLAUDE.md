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
uv run python packaging/build.py                # 버전 스탬프(빌드 번호 +1) → PyInstaller → selftest → 설치 파일
uv run pyinstaller packaging/kstruct.spec --noconfirm   # 프리즈만 (dist/KStruct/), 버전 스탬프 없음
```

CI(`.github/workflows/build.yml`)가 Windows/macOS/Linux에서 테스트 → PyInstaller → 배포본 selftest → 설치 파일(Inno Setup / dmg / tar.gz)을 만든다. 로컬 Windows에서는 Inno Setup 설치가 실패해 설치 파일 단계는 CI에서만 검증됨.

## 아키텍처

데이터 흐름: `geometry`(CAD/메시 → `Project`) → `fem.mesh`(Netgen TET10) → `fem.static|modal|thermal|buckling` → `report.summary`(판정·쉬운 설명) → `ui`/`report.pdf`.

- **단위계**: 내부는 mm·N·MPa·tonne·s 고정. 밀도 t/mm³(재료 JSON은 kg/m³로 저장하고 로드 시 변환), 열량 1 W = 1000 N·mm/s, 대류 h[W/m²K]×1e-3. 변환은 `standards/__init__.py`와 `fem/thermal.py`에만 있다.
- **면 식별(FaceRef)**: `geometry.faces_of(shape)`의 리스트 인덱스가 `FaceRef.face_index`. Netgen의 면 번호도 같은 순서임이 테스트로 보장된다. 뷰어 삼각형의 `cell_data["face_id"]`, 메시 경계 삼각형의 `FEMesh.tri_face`가 모두 이 인덱스라서 클릭 → 고정/하중 → 메시 절점이 연결된다. 면 순서를 바꾸는 형상 처리를 하면 기존 FaceRef가 깨진다.
- **Netgen**은 자체 OCCT 사본을 쓰므로 부품을 STEP 임시 파일로 넘긴다(`fem/mesh.py`). `mesh.Coordinates()`는 Netgen 메모리 뷰라 반드시 복사할 것(안 하면 다음 부품 메싱 때 좌표가 깨짐).
- **솔버는 자체 구현**(GPL 회피): `fem/tet10.py`가 요소 수식, `fem/assembly.py`가 조립·선형해(작으면 SuperLU, 크면 pyamg CG). SuperLU는 반드시 `MMD_AT_PLUS_A` + `SymmetricMode` + `DiagPivotThresh=0`(SPD라 대각 피벗 안전). 기본 부분 피벗은 페널티 tie가 있으면 채움이 30배로 폭증한다. 여러 우변은 `make_solver`로 한 번 분해 후 재사용. 경계조건(`fem/constraints.py`)은 축 정렬 구속은 DOF 제거, 경사 미끄럼/핀/부품 간 접합(tie)은 페널티 방정식. 반력은 각 고정이 실제로 구속한 DOF/방정식만 합산한다(모서리 공유 DOF 이중 집계 방지).
- **열 해석**(`fem/thermal.py`): 정상 상태 전도. 복사(`LoadKind.RADIATION`, magnitude = 방사율, `ambient` = 주변 온도)는 등가 대류 계수 εσ(T²+T∞²)(T+T∞)로, 온도 의존 열전도율(`Material.conductivity_table`, 재료 JSON `k_table`)은 요소 평균 온도 보간으로 넣고 둘 중 하나라도 있으면 `MAX_ITERATIONS`까지 고정점 반복(변화 < 범위의 1e-4). 온도 기준(`TEMPERATURE_REFERENCES`: 고정 온도·대류·복사) 없이는 `SolverError`. 과도 해석은 없음.
- **열응력**: 정적해석에 열 조건(`THERMAL_LOADS`)이 있으면 먼저 `solve_thermal`로 온도장을 구하고 `reference_temperature` 대비 ΔT로 열하중을 만든다. 피로(`fem/fatigue.py`)는 정적 결과를 반복 하중의 최대값으로 보고 `report.summary.summarize`에서 계산된다. 피로 한도 보정은 `FatigueSettings`(표면 `SURFACE_FACTORS`, 신뢰도 `RELIABILITY_FACTORS`, 크기 0.85)로 하고 `Project.fatigue_surface/fatigue_reliability`에 저장; settings 없이 부르면 옛 상수 `MARIN_FACTOR`(0.7).
- **접촉**(`fem/contact.py`, `ContactSolver`): 마찰 없는 node-to-surface 페널티(`CONTACT_PENALTY_FACTOR` 1e3, 접합 tie의 1e6보다 무름) + 능동 집합 반복. 해제 판정은 틈이 아니라 **평형 잔차 K u − f의 슬레이브 절점 성분**(= 스프링 힘; α·틈은 반올림을 α배 증폭하므로 쓰지 말 것)으로 하고, 임계값은 최대 접촉 압력의 0.1 %. 결합은 요소 크기·변위 크기 기준 관통. 쌍마다 상태 변경을 `MAX_FLIPS`회로 제한하고 상태가 순환하면 멈춘다(채터링 방지). 분해는 한 번 하고 능동 집합이 같으면 재사용, 바뀐 쌍이 적으면 Woodbury 저랭크 갱신(재분해와의 선택은 `make_solver`가 붙이는 `solve.breakeven` = 0.25·nnz(LU)/n 기준. **벽시계 시간으로 고르지 말 것**: 반올림 경로가 실행마다 달라져 마찰 결과가 195~1500 N으로 흔들렸다). K_c의 잔차는 페널티 행이 반올림을 증폭해 정확도 지표로 쓸 수 없고, 반복 개선은 발산한다. 접촉으로만 붙은 부품을 위해 1e-9 크기 안정화 스프링을 쓰고, 그 부품이 미끄러지면 `SolverError("contact unstable")`. 진동·좌굴·열 해석에서는 접촉을 접합으로 취급. **마찰**(`Connection.friction`, 쿨롱): 닫힌 쌍은 접선 스프링 2개(같은 α)로 고착, 스프링 힘이 μN을 넘으면 스프링을 빼고 μN 크기의 마찰력을 외력으로 건다(방향은 처음 미끄러질 때의 스프링 힘 방향, 면 전체가 한 방향으로 μΣN을 넘으면 모든 쌍을 그 합력 방향으로 한 번에 미끄럼 처리). 미끄럼 쌍의 방향을 변위로 다시 정하면 이웃의 탄성 반응(노이즈)에 따라 매 반복 뒤집히므로 하지 말 것. 미끄럼↔고착 반복이 `MAX_FRICTION_FLIPS`를 넘으면 **미끄럼 상태로** 고정(고착으로 고정하면 무한한 힘을 버팀). 매 `solve`마다 마찰 상태는 초기화(열림/닫힘은 유지). `StaticResult.contact`는 (닫힘, 전체, 미끄럼) 수. 반복 상한을 다 쓰면 `contact_converged=False`가 되고 요약 첫 메모로 경고한다(예압+마찰 겹침 이음은 현재 수렴하지 않음 — `test_friction_relieves_the_bolt_of_part_of_the_shear`가 strict xfail).
- **볼트**(`fem/bolts.py`): 머리·너트 아래 고리 면(없으면 구멍 벽)의 평균 변위를 잇는 축/전단 스프링. 예압은 "잠금" 방식: 예압만 걸린 상태에서 각 볼트가 정확히 P가 되도록 당김력 T를 `locked_pulls`로 풀고, 축력은 항상 모델 안의 실제 힘 T + k·신장량으로 보고한다(평형 보장). 예압이 있으면 두 부품 사이에 접촉/접합 연결이 필수.
- **좌굴**(`fem/buckling.py`): ARPACK `mode="buckling"`(K는 SPD, K_G는 부정치). 압축이 거의 없으면 반복 상한에서 끊고 "좌굴 없음"으로 처리.
- **표시 단위**(`units.py`): 계산은 항상 SI 내부 단위. 입력·목록·결과 필드·보고서만 `system_of(project)`(프로젝트 설정, 없으면 기준 팩의 `units`)로 변환한다.
- **여러 부품**: 처음부터 `Project.parts` 리스트 + `connections`. STEP의 솔리드마다 Part 하나(이름은 XCAF `read_named_solids`로 읽고 어셈블리는 배치를 적용해 평탄화; STEP의 UTF-8 이름은 바이트당 한 코드포인트로 돌아오므로 `_decode_name`으로 복원), 맞닿은 면은 `geometry/contacts.py`가 자동 접합(바깥 법선이 마주보고 실제 겹침 면적이 있어야 함).
- **메시 포맷**(`geometry/mesh_import.py`, `geometry/fbx.py`): 삼각형을 공유 위상 B-rep 솔리드로 만들고 같은 평면 삼각형을 병합. 열린 메시는 자동 보정하지 않고 거부한다(보정이 형상을 몰래 바꾸므로). FBX 리더는 자체 구현.
- **기준 팩/재료**: `resources/standards.json`, `resources/materials.json`. 국가별 명칭·권장 안전율(좌굴 하중 배수 포함)은 데이터로만 추가한다. 계산 코드에 국가 기준을 하드코딩하지 말 것. 볼트 규격·등급은 `resources/bolts.json`.
- **도움말**: 모든 UI 요소는 `resources/help/ko.json`의 키를 참조(`HelpButton`, `with_help`, `topic(key).short`). `tests/test_ui.py`가 UI 코드에 쓰인 키가 JSON에 다 있는지 검사하므로 새 접두어를 쓰면 그 테스트의 `PREFIXES`도 갱신.
- **UI**: `ui/main_window.py`가 Project·mesh·result를 소유하고, `ui/steps.py`의 7개 패널이 이를 편집해 `changed`를 보낸다. 메싱·해석은 `ui/worker.run_job`으로 백그라운드 실행. 결과 종류(Static/Modal/Thermal)에 따라 결과 패널 필드가 바뀐다.
- **단면·내보내기**: 뷰어는 평소 TRI6 겉면만 그리지만 단면 보기는 `viewer.linear_volume_grid`(TET10을 선형 사면체 8개로 분할; VTK clip은 비선형 셀을 못 다룸)를 잘라 보여준다. VTU 내보내기는 `volume_grid`(VTK 2차 사면체, 중간 절점 순서 `_TET10_TO_VTK`로 재배열 — Netgen과 VTK의 중간 절점 순서가 다르다). 결과 열 정의는 `report/export.py.result_columns` 한 곳.
- **3D 텍스트 한글**: VTK 기본 글꼴에 한글이 없어 색 막대에 번들 Noto Sans KR(OFL)을 지정한다(`viewer._use_korean_font`).
- **결과 표시 주의**: 2차 요소 보간이 범위를 벗어나 거짓 색을 만들 수 있어 안전율 같은 잘린 값은 표시 전에 clim으로 자른다.
- **응력 특이점**(`fem/singularity.py`): 필렛 없는 오목 모서리(경계 삼각형 쌍의 이면각으로 판정)와 완전 고정면 가장자리 절점을 표시하고, 그 주변(요소 1.5개 또는 부품 대각선 3% 중 큰 값)을 제외한 "특이점 제외(핫스팟)" 응력·안전율을 `report.summary`가 함께 보고한다. 연성재 판정은 핫스팟 값(`PartSummary.governing_fos`), 취성재는 최대값 기준. 예제 브래킷의 판정이 메시 등급과 무관하게 같아야 한다(`tests/test_improvements.py`).
- **숨긴 부품**: 해석·메시·자동 접합은 항상 `model.active_project(project)`(보이는 부품 + 그 부품만 참조하는 고정·하중·연결·볼트)를 쓴다. 숨긴 부품에 걸린 조건은 6단계 체크리스트에 "제외" 표시.
- **권장 안전율·사용자 재료**: 판정 기준은 `standards.effective_pack(project)`(`Project.target_overrides` 반영), 재료 조회는 `standards.project_materials(project)`(내장 + `Project.custom_materials`). `standards.pack()`/`materials()`를 UI·요약·보고서에서 직접 쓰지 말 것. 사용자 재료는 .kst에 저장되고 `paths.user_materials_path()` 라이브러리에도 복사된다.
- **메시 크기**: `MESH_PRESETS`는 `MeshPreset(분할수, grading, 두께 방향 요소 수)`. `fem.mesh.face_sizes`가 `geometry/thickness.py`(테셀레이션에 자체 Möller–Trumbore 레이 캐스팅, 면별 중앙값 두께)로 얇은 면의 크기를 두께/요소수로 줄이고 `Project.mesh_refinements`를 합친 뒤 Netgen에 면별 `maxh`로 넘긴다. Netgen은 `geo.shape.faces[i].maxh`를 설정한 뒤 **`OCCGeometry(shape)`로 다시 감싸야** 반영된다(원래 geo에 바로 설정하면 무시됨). netgen은 OCP와 다른 OCCT 사본이라 주 프로세스에서 import하면 `faces_of`가 빈 목록을 돌려주므로 반드시 워커 프로세스 안에서만 import.
- **하중 케이스**: `Project.cases`(`LoadCase`) + `active_case`. `project.loads`는 활성 케이스의 리스트(프로퍼티)라 기존 코드가 그대로 동작하고, 강도 해석은 `solve_cases`가 `with_case(project, i)`(얕은 뷰, 복사 없음)로 모든 케이스를 푼다. UI는 `case_results`에 케이스별 (이름, 결과, 요약)을 들고 `select_case`로 바꿔 보여주며, 열·좌굴은 활성 케이스만 쓴다. 프로젝트 파일 v2는 `cases`를 저장하고 v1의 평면 `loads`도 읽는다.
- **회전 하중**(`fem/loads.py`): 토크는 선택 면에 반지름에 비례하는 접선 트랙션(합 모멘트 = T 정확), 베어링 하중은 구멍 원통면의 하중 방향 절반에 코사인 분포(합력 = F가 되도록 잔차를 보정), 원심력은 ρω²r 체적력. 축은 `Load.point` + `vector`(토크·원심력, 없으면 선택 원통면의 축), 베어링은 항상 원통면 축. 내부 단위 N·mm, rad/s이며 표시는 `units`의 `torque`(N·m / lbf·in)·`speed`(rpm).
- **조건 문구**: 고정·하중·연결 설명은 `report/describe.py` 한 곳에서 만들어 패널과 PDF가 같은 문장을 쓴다. 보고서 수치는 모두 `units.fmt`를 거친다.
- **프로젝트 파일의 메시**: `save_project(project, path, mesh)`는 `mesh.npz`와 `mesh.json`(키 = 보이는 부품 brep 바이트 해시 + 메시 등급 + 세분화)을 넣고, `load_project_mesh`는 키가 현재 프로젝트와 같을 때만 돌려준다. 결과는 저장하지 않는다.
- **환경설정·자동 저장**(`ui/prefs.py`): 데이터 폴더의 `prefs.json`(최근 파일, 마지막 폴더, 창 geometry·splitter)과 `autosave.kst`(+`.json` 메타). `MainWindow._dirty`가 변경을 추적해 3분마다 자동 저장하고, 시작 시 `offer_recovery()`가 복구를 묻는다. `closeEvent`는 dirty면 저장 여부를 묻는데 **테스트에서는 `window.ask_before_close = False`**로 꺼야 한다(pytest-qt가 teardown에서 창을 닫을 때 모달이 떠서 멈춤). 예제 파일은 최근 목록에 넣지 않는다.
- **로그·진단**(`paths.py`): `setup_logging()`이 사용자 데이터 폴더(`KSTRUCT_DATA_DIR`로 재지정 가능)에 회전 로그를 쓰고, `install_excepthook` + `MainWindow.install_crash_dialog`가 처리되지 않은 예외를 로그·대화상자로 보낸다. "문제 보고"는 로그+환경+.kst를 zip으로 묶는다(`ui/dialogs.py`). 릴리스 버전은 `kstruct.RELEASE_VERSION`(pyproject와 일치, 테스트 확인)이고, `packaging/build.py`가 빌드마다 `packaging/build_number.txt`를 +1 하거나(CI는 `KSTRUCT_BUILD_NUMBER` = run number) `src/kstruct/_build.py`(git 무시)에 스탬프를 써서 `__version__` = `릴리스.빌드번호`(예 0.1.0.57)가 된다. 정보 창·로그·보고서·설치 파일 이름이 모두 이 값을 쓴다(`build_info()`는 날짜·커밋 포함). `kstruct --version`으로 확인.

## OCP(cadquery-ocp 8.x, OCCT 8) 주의

정적 메서드는 대개 `_s` 접미사지만 예외가 있다(`TopoDS.Face`, `Bnd_Box.CornerMin`). `TopTools_IndexedMapOfShape` 대신 `OCP.collections.IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher`. 모르면 `dir()`로 확인. OCCT 표준출력 로그는 `geometry/__init__.py`에서 끈다.

## 패키징 주의

`packaging/kstruct.spec`: delvewheel `*.libs` 폴더와 `netgen-occt`의 DLL(venv의 bin에 설치됨)을 직접 포함하고, Netgen이 런타임에 잘못된 경로를 찾지 않도록 `netgen_occt-*.dist-info`는 번들에서 제외한다. matplotlib은 PyVista가 import하므로 제외하면 안 된다.

## 결과물 원칙

결과 화면·보고서에는 "참고용, 최종 판단은 자격 있는 기술자" 고지를 유지한다. 새 해석 기능은 이론해와 비교하는 테스트를 함께 추가한다(`tests/test_fem_*.py` 참고).
