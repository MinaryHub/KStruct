# 배포·코드 서명 안내

> KStruct는 판매·외부 배포용 제품이 아니므로 이 문서의 서명·공증·릴리스는 **필요할 때만** 쓰는 선택 사항입니다. 인증서를 설정하지 않으면 빌드는 지금처럼 서명 없이 만들어지고, 사내에서 쓰는 데는 문제가 없습니다(처음 실행할 때 Windows/macOS가 한 번 경고할 수 있음).

서명하지 않은 설치 파일은 Windows SmartScreen("Windows의 PC 보호")과 macOS Gatekeeper("확인되지 않은 개발자")에 막힙니다. 빌드는 서명 없이도 되지만, 고객에게 배포하려면 아래 인증서가 필요합니다. 인증서 발급과 계약은 회사가 직접 해야 하는 일이라 코드에는 **비밀값이 있으면 서명하고, 없으면 건너뛰는** 단계만 준비되어 있습니다.

## 1. 필요한 것

| 플랫폼 | 필요한 것 | 비고 |
|---|---|---|
| Windows | 코드 서명 인증서 (.pfx) — OV 또는 EV | EV는 SmartScreen 평판이 바로 생기지만 보통 하드웨어 토큰/클라우드 HSM이 필요. 이 경우 `KSTRUCT_SIGN_COMMAND`로 업체 도구를 부른다 |
| macOS | Apple Developer Program 가입, "Developer ID Application" 인증서 (.p12) | 공증(notarization)에 Apple ID와 앱 암호가 필요 |

## 2. GitHub 저장소 비밀값 (Settings → Secrets and variables → Actions)

| 이름 | 내용 |
|---|---|
| `WINDOWS_CERT_PFX_BASE64` | .pfx 파일을 base64로 인코딩한 문자열 (`base64 -w0 cert.pfx`) |
| `WINDOWS_CERT_PASSWORD` | .pfx 암호 |
| `MACOS_CERT_P12_BASE64` | Developer ID Application 인증서(.p12)의 base64 |
| `MACOS_CERT_PASSWORD` | .p12 암호 |
| `MACOS_SIGN_IDENTITY` | `Developer ID Application: 회사명 (TEAMID)` |
| `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD` | 공증용 (앱 암호는 appleid.apple.com에서 발급) |

비밀값이 하나도 없으면 지금처럼 서명 없는 설치 파일이 만들어지고, 빌드 로그에 `note: unsigned build`가 남습니다.

## 3. 릴리스

```
git tag v0.2.0 && git push origin v0.2.0
```

태그를 푸시하면 CI가 세 OS의 설치 파일을 만들고(서명·공증 포함), GitHub Release를 만들어 설치 파일과 `SHA256SUMS.txt`를 올립니다. 릴리스 노트는 커밋 목록으로 자동 작성되므로 필요하면 편집합니다.

## 4. 로컬에서 서명하며 빌드

```
# Windows
set KSTRUCT_SIGN_PFX=C:\certs\kstruct.pfx
set KSTRUCT_SIGN_PASSWORD=...
uv run python packaging/build.py

# macOS
export KSTRUCT_MACOS_IDENTITY="Developer ID Application: 회사명 (TEAMID)"
export KSTRUCT_NOTARY_PROFILE=kstruct-notary   # xcrun notarytool store-credentials 로 만든 프로필
uv run python packaging/build.py
```

## 5. 아직 코드 밖에서 해야 할 일

- **EULA(최종 사용자 사용권 계약)**: 결과에 대한 책임 한계, 사용 범위, 오픈소스 고지 참조를 담아야 하며 법률 검토가 필요합니다. 확정되면 `packaging/windows/kstruct.iss`의 `LicenseFile`을 EULA로 바꾸고(오픈소스 고지는 `[Files]`로 계속 설치), 정보 창에도 연결합니다.
- **CI 실행 환경**: 현재 GitHub Actions 작업이 러너를 배정받지 못하고 2초 만에 끝납니다(결제·사용 한도 또는 Actions 권한). 이것이 풀려야 위 서명·릴리스가 실제로 동작합니다.
