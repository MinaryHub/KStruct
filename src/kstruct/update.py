"""Updates from GitHub Releases, plus the contact / sponsor links.

The repository is private, so the GitHub API answers 404 without a token: a personal
access token with read access to the repository (fine-grained: "Contents: read") is
taken from the environment variable KSTRUCT_GITHUB_TOKEN or the preferences.

Flow: `check()` reads /releases/latest and returns the release when its tag is newer
than this build's RELEASE_VERSION; `download()` fetches the installer for this platform
and checks it against the release's SHA256SUMS.txt (written by CI); `launch_installer()`
starts it. Nothing here touches Qt, so it can be tested without a window.

All network access goes through `_request` (one seam for the tests). Release assets of
a private repository are fetched through the API, which redirects to a signed storage
URL: the token must not follow that redirect (the storage rejects a second credential,
and it should not see the token anyway).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

from kstruct import RELEASE_VERSION, __version__

log = logging.getLogger(__name__)
API = "https://api.github.com"
TIMEOUT = 15  # seconds per request
CHUNK = 1 << 16
TOKEN_ENV = "KSTRUCT_GITHUB_TOKEN"
CHECK_INTERVAL_HOURS = 24  # automatic check at start-up at most this often


class UpdateError(Exception):
    """A failure the user can act on; the message is Korean UI text."""


class Cancelled(Exception):
    pass


@cache
def links() -> dict:
    """resources/links.json: repository, issue page, sponsor page."""
    text = resources.files("kstruct").joinpath("resources/links.json").read_text("utf-8")
    return json.loads(text)


# --- versions ---------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...]:
    """'v0.2.0', '0.2.0', '0.1.0.57' -> integer tuple; trailing text ('-rc1') ignored."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", text or "")
    if not match:
        raise ValueError(f"not a version: {text!r}")
    return tuple(int(p) for p in match.group(1).split("."))


def is_newer(tag: str, current: str = RELEASE_VERSION) -> bool:
    """Releases are tagged with the release version only (v0.2.0); the build number of
    this copy does not take part."""
    a, b = parse_version(tag), parse_version(current)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


# --- releases ---------------------------------------------------------------------------


@dataclass
class Asset:
    name: str
    api_url: str  # .../releases/assets/<id> (works for private repositories)
    size: int = 0


@dataclass
class Release:
    tag: str
    name: str
    notes: str
    page: str  # html_url
    assets: list[Asset] = field(default_factory=list)

    @property
    def version(self) -> str:
        return self.tag.lstrip("vV")

    def asset(self, name: str) -> Asset | None:
        return next((a for a in self.assets if a.name == name), None)


def installer_pattern(system: str | None = None, machine: str | None = None) -> re.Pattern:
    """File name of this platform's installer, as packaging/build.py names it."""
    system = system or platform.system()
    if system == "Windows":
        return re.compile(r"^KStruct-Setup-[\d.]+\.exe$", re.I)
    if system == "Darwin":
        return re.compile(r"^KStruct-[\d.]+-macOS\.dmg$", re.I)
    arch = re.escape(machine or platform.machine())
    return re.compile(rf"^KStruct-[\d.]+-linux-{arch}\.tar\.gz$", re.I)


def pick_installer(release: Release, system=None, machine=None) -> Asset | None:
    pattern = installer_pattern(system, machine)
    return next((a for a in release.assets if pattern.match(a.name)), None)


def token(saved: str | None = None) -> str | None:
    return os.environ.get(TOKEN_ENV) or saved or None


def _headers(tok: str | None, accept: str = "application/vnd.github+json") -> dict:
    headers = {
        "Accept": accept,
        "User-Agent": f"KStruct/{__version__}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


@cache
def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError):
        return ssl.create_default_context()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # handled by _request
        return None


def _request(url: str, headers: dict):
    """Open url and return the response; a redirect to another host is followed without
    the Authorization header. HTTP errors raise urllib.error.HTTPError."""
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_ssl_context()), _NoRedirect()
    )
    for _ in range(5):
        try:
            return opener.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308) or not e.headers.get("Location"):
                raise
            target = urllib.parse.urljoin(url, e.headers["Location"])
            if urllib.parse.urlsplit(target).netloc != urllib.parse.urlsplit(url).netloc:
                headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
                headers["Accept"] = "application/octet-stream"
            url = target
    raise UpdateError("업데이트 서버가 너무 여러 번 다른 주소로 보냈습니다.")


def _explain(error: Exception, tok: str | None) -> UpdateError:
    if isinstance(error, urllib.error.HTTPError):
        if error.code == 404:
            return UpdateError(
                "릴리스를 찾을 수 없습니다. 저장소가 비공개라서 읽을 수 있는 GitHub 토큰이 "
                "필요합니다."
                if not tok
                else "릴리스를 찾을 수 없습니다. 아직 릴리스가 없거나, 토큰에 이 저장소를 "
                "읽을 권한이 없습니다."
            )
        if error.code == 401:
            return UpdateError("GitHub 토큰이 맞지 않거나 만료되었습니다. 토큰을 다시 입력하세요.")
        if error.code in (403, 429):
            return UpdateError(
                "GitHub 요청 한도에 걸렸거나 접근이 거부되었습니다. 잠시 뒤 다시 시도하세요."
            )
        return UpdateError(f"업데이트 서버 오류 (HTTP {error.code}).")
    if isinstance(error, (urllib.error.URLError, TimeoutError, OSError)):
        return UpdateError(
            "업데이트 서버에 연결할 수 없습니다. 인터넷(또는 사내 프록시) 연결을 확인하세요."
        )
    return UpdateError(f"업데이트 확인 중 오류: {error}")


def latest_release(tok: str | None = None, repo: str | None = None) -> Release:
    repo = repo or links()["repository"]
    try:
        with _request(f"{API}/repos/{repo}/releases/latest", _headers(tok)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except UpdateError:
        raise
    except Exception as e:  # noqa: BLE001 (network errors of every kind become UI text)
        raise _explain(e, tok) from e
    return Release(
        tag=data.get("tag_name", ""),
        name=data.get("name") or data.get("tag_name", ""),
        notes=data.get("body") or "",
        page=data.get("html_url", ""),
        assets=[Asset(a["name"], a["url"], int(a.get("size", 0))) for a in data.get("assets", [])],
    )


def check(tok: str | None = None, current: str = RELEASE_VERSION) -> Release | None:
    """The latest release when it is newer than this copy, else None."""
    release = latest_release(tok)
    try:
        newer = is_newer(release.tag, current)
    except ValueError:
        log.warning("release tag %r is not a version", release.tag)
        return None
    return release if newer else None


# --- download & install -----------------------------------------------------------------


def _fetch(asset: Asset, tok: str | None, out: Path, progress=None, cancelled=None) -> None:
    headers = _headers(tok, "application/octet-stream")
    partial = out.with_name(out.name + ".part")
    try:
        with _request(asset.api_url, headers) as response, partial.open("wb") as f:
            total = int(response.headers.get("Content-Length") or asset.size or 0)
            done = 0
            while chunk := response.read(CHUNK):
                if cancelled is not None and cancelled():
                    raise Cancelled
                f.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        os.replace(partial, out)
    except (UpdateError, Cancelled):
        partial.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        raise _explain(e, tok) from e


def expected_sha256(sums: str, name: str) -> str | None:
    """Look up `name` in sha256sum output ('<hex>  <name>' or '<hex> *<name>')."""
    for line in sums.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[1].lstrip("*").strip() == name:
            return parts[0].lower()
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    release: Release,
    folder: Path,
    tok: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Download this platform's installer into `folder` and verify its checksum."""
    asset = pick_installer(release)
    if asset is None:
        raise UpdateError(
            f"{release.tag} 릴리스에 이 운영체제({platform.system()})용 설치 파일이 없습니다."
        )
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / asset.name
    _fetch(asset, tok, out, progress, cancelled)
    sums_asset = release.asset("SHA256SUMS.txt")
    if sums_asset is None:
        log.warning("release %s has no SHA256SUMS.txt: %s not verified", release.tag, asset.name)
        return out
    sums_path = folder / f"SHA256SUMS-{release.version}.txt"
    _fetch(sums_asset, tok, sums_path)
    expected = expected_sha256(sums_path.read_text("utf-8", errors="replace"), asset.name)
    actual = sha256_of(out)
    if expected is None or expected != actual:
        out.unlink(missing_ok=True)
        raise UpdateError(
            "받은 설치 파일이 릴리스의 체크섬과 맞지 않아 지웠습니다(다운로드가 깨졌거나 "
            "파일이 바뀌었을 수 있음). 다시 시도하세요."
        )
    return out


def frozen() -> bool:
    """Running from the installed (PyInstaller) app rather than a source checkout."""
    return bool(getattr(sys, "frozen", False))


def launch_installer(path: Path) -> bool:
    """Start the installer. Returns True when the app should quit so that the installer
    can replace its files (Windows); otherwise the downloaded file is opened for the
    user (a disk image to drag into Applications, an archive to unpack)."""
    if platform.system() == "Windows":
        # the setup closes this app if still running and starts it again when done
        flags = ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"]
        subprocess.Popen([str(path), *flags], close_fds=True)  # noqa: S603
        return True
    opener = "open" if platform.system() == "Darwin" else "xdg-open"
    target = path if platform.system() == "Darwin" else path.parent
    subprocess.Popen([opener, str(target)], close_fds=True)  # noqa: S603
    return False


# --- contact ----------------------------------------------------------------------------


def issue_url(title: str = "", body: str = "") -> str:
    """New-issue page with the title and body filled in (kept well under URL limits)."""
    query = urllib.parse.urlencode({"title": title, "body": body[:4000]})
    return f"{links()['issues']}?{query}"


def environment_text() -> str:
    from kstruct import build_info

    return (
        f"- KStruct: {build_info()}\n"
        f"- OS: {platform.platform()}\n"
        f"- Python: {platform.python_version()}"
    )
