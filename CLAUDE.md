# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

사용자와의 대화 및 UI 기본 언어는 한국어. 현재 Phase 0 완료(STEP 열기, 3D 표시, 면 선택, 도움말 시스템). 다음은 Phase 1: Netgen 메싱 → CalculiX 정적해석 → 결과 파싱, 이론해 검증.

## 명령어

uv로 관리하며 Python 3.12 고정(OCP·Netgen 휠 호환 때문에 3.13+ 사용 금지).

```
uv sync                                   # 의존성 설치
uv run kstruct [model.step]               # 앱 실행
uv run pytest                             # 전체 테스트
uv run pytest tests/test_geometry.py::test_box_round_trip   # 단일 테스트
uv run ruff check --fix src tests && uv run ruff format src tests
```

OpenCASCADE는 STEP 읽기/쓰기 시 C++ 로그를 stdout에 대량 출력한다. 테스트 출력을 볼 때는 `| grep -vE '^\*|^$|WorkSession|Step File|Transfer'`로 걸러 볼 것.

## 확정된 요구사항

- 대상: **기계설계 부품**의 3D 솔리드 유한요소해석(FEA). 건축 골조 해석이 아님.
- 입력: SolidWorks(.sldprt/.sldasm), Inventor(.ipt/.iam), CATIA(.CATPart/.CATProduct), 중립 포맷(STEP, IGES), 메시 포맷(STL, OBJ, FBX, glTF, 3MF).
- 배포: Windows, macOS, Linux **설치형 데스크톱 앱**.
- 기준: 국제기준(ISO 등)이 기본, 국가별 기준(KS, JIS, ASTM/ASME, EN, GB 등)은 선택 옵션.
- **상용 수준** 제품: 라이선스가 상용 배포와 호환되어야 하고, 배포본의 소스 보호(예: Nuitka 컴파일)를 고려한다.
- 초기 MVP는 단일 부품이지만, 데이터 모델은 처음부터 **여러 부품(어셈블리)**을 전제로 설계한다(부품 목록 + 부품 간 연결 정의). 연결은 접합(bonded/tie) → 마찰·분리 접촉 → 볼트 순으로 확장.
- 비전문가 대상: 단계별 마법사, 모든 기능에 `?` 툴팁 + 클릭 팝업 도움말, OK/주의/NG 판정.

## 설계 시 유의할 제약

- 네이티브 CAD 포맷(SolidWorks/Inventor/CATIA)은 무료 라이브러리로 읽을 수 없다. 상용 SDK(CAD Exchanger, HOOPS Exchange 등)가 필요하거나 사용자가 STEP으로 내보내야 한다.
- 메시 포맷(FBX/STL/OBJ)은 B-rep 면 정보가 없어 면 선택 기반의 하중·구속 지정이 제한되며, 닫힌(watertight) 메시일 때만 솔리드 메싱이 가능하다.
- 라이브러리 라이선스: GPL 라이브러리(예: Gmsh)를 링크하면 앱 전체가 GPL이 된다. 상용 배포 가능성을 고려해 LGPL/BSD/MIT 계열을 우선 사용하고, GPL 솔버는 별도 프로세스로만 호출한다.
- 국가별 기준은 코드에 하드코딩하지 않고 "기준 팩" 데이터(재료 DB, 안전계수, 검토식)로 분리해 추가·교체할 수 있게 한다.
- 도움말 텍스트는 코드에 넣지 않고 언어별 JSON(예: `help/ko.json`)에 키 단위로 관리한다.
- 결과 화면과 보고서에는 "참고용이며 최종 판단은 전문가 검토 필요" 고지를 포함한다.

## 아키텍처

- `src/kstruct/model/`: `Project`(parts + connections). 단일 부품도 항상 `parts` 리스트에 담는다. 내부 단위계는 mm·N·MPa·tonne·s로 고정.
- 면 식별: `geometry.faces_of(shape)`가 반환하는 리스트의 인덱스가 곧 `FaceRef.face_index`다(TopExp.MapShapes 순서, 안정적). 뷰어의 삼각형은 `cell_data["face_id"]`로 이 인덱스를 들고 있어서 클릭 → B-rep 면 → 이후 고정/하중 지정으로 이어진다. 테셀레이션은 면마다 정점을 따로 두므로 메시 경계 모서리 = B-rep 모서리.
- 뷰어 하이라이트는 겹친 메시를 얹지 않고 셀 RGB 배열(`cell_data["rgb"]`)을 다시 칠한다(z-fighting 방지).
- `ui/help.py` + `resources/help/ko.json`: 모든 UI 요소는 도움말 키를 참조(`HelpButton`, `with_help`, `topic(key).short`를 툴팁으로). `tests/test_ui.py`가 코드에 쓰인 `step./view./file./help.` 키가 JSON에 모두 있는지 검사하므로, 새 키 접두어를 쓰면 그 정규식도 갱신할 것.
- OCP는 cadquery-ocp 8.x(OCCT 8): 정적 메서드는 `_s` 접미사지만 예외가 있음(`TopoDS.Face`, `Bnd_Box.CornerMin` 등). `TopTools_IndexedMapOfShape` 대신 `OCP.collections.IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher`. 모르면 `dir()`로 확인.
- Netgen(`netgen-mesher`, 아직 의존성에 미추가)은 자체 OCCT 사본을 쓴다. OCP 객체를 직접 넘길 수 없으므로 STEP/BREP 파일로 전달해야 한다. 같은 프로세스에서 함께 import되는 것은 확인됨.

## 계획된 구성 (미구현)

- 솔버는 추상화 계층 뒤에 둔다. 초기 구현은 CalculiX(GPL, 별도 프로세스 호출 — 배포 시 GPL 소스 제공 의무 준수)이고, 이후 자체 솔버나 상용 솔버로 교체할 수 있어야 한다.

- 배포: Nuitka 컴파일 + OS별 설치 파일.
- 파이프라인: 형상 입력(OpenCASCADE/OCP로 STEP·IGES, Assimp/trimesh로 메시 포맷) → Netgen 4면체 2차 메싱 → CalculiX `.inp` 생성 후 별도 프로세스 실행 → `.frd` 결과 파싱 → 기준 팩으로 안전율·판정 → 시각화/보고서.
- geometry/mesh/solver 계층은 UI 없이 동작·테스트 가능해야 하며, 이론해(외팔보, 인장봉, 원공 응력집중)와 비교하는 검증 테스트를 둔다.
- 7단계 마법사(`ui/main_window.py`의 `WIZARD_STEPS`)는 현재 1단계만 활성화되어 있다.
