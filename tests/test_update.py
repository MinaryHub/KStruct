"""Updates from GitHub Releases: version comparison, asset choice, error messages, the
download checksum and the token not following a redirect to the storage host."""

import hashlib
import http.server
import json
import threading
import urllib.error
from email.message import Message
from io import BytesIO

import pytest

from kstruct import RELEASE_VERSION, update


def test_versions():
    assert update.parse_version("v0.2.0") == (0, 2, 0)
    assert update.parse_version("0.1.0.57") == (0, 1, 0, 57)
    assert update.parse_version("v1.2-rc1") == (1, 2)
    assert update.is_newer("v0.2.0", "0.1.0")
    assert update.is_newer("v0.1.1", "0.1")
    assert not update.is_newer("v0.1.0", "0.1.0")
    assert not update.is_newer("v0.1", "0.1.0")
    assert not update.is_newer("v0.0.9", "0.1.0")
    with pytest.raises(ValueError):
        update.parse_version("latest")


def _release(*names):
    return update.Release(
        "v9.0.0", "9.0.0", "notes", "https://x/r", [update.Asset(n, f"api/{n}") for n in names]
    )


def test_installer_per_platform():
    names = (
        "KStruct-Setup-9.0.0.12.exe",
        "KStruct-9.0.0.12-macOS.dmg",
        "KStruct-9.0.0.12-linux-x86_64.tar.gz",
        "SHA256SUMS.txt",
    )
    r = _release(*names)
    assert update.pick_installer(r, "Windows").name == names[0]
    assert update.pick_installer(r, "Darwin").name == names[1]
    assert update.pick_installer(r, "Linux", "x86_64").name == names[2]
    assert update.pick_installer(r, "Linux", "aarch64") is None


class _Response(BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Length"] = str(len(data))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _http_error(code):
    return urllib.error.HTTPError("https://api.github.com/x", code, "err", Message(), None)


def test_check_and_errors(monkeypatch):
    seen = {}

    def fake(url, headers):
        seen["url"], seen["headers"] = url, headers
        return _Response(
            json.dumps({"tag_name": "v99.0.0", "body": "새 기능", "assets": []}).encode()
        )

    monkeypatch.setattr(update, "_request", fake)
    release = update.check("tok")
    assert release.version == "99.0.0" and release.notes == "새 기능"
    assert seen["url"].endswith("/repos/knoxxr/KStruct/releases/latest")
    assert seen["headers"]["Authorization"] == "Bearer tok"

    monkeypatch.setattr(
        update,
        "_request",
        lambda url, headers: _Response(json.dumps({"tag_name": f"v{RELEASE_VERSION}"}).encode()),
    )
    assert update.check() is None  # same version: nothing to do

    def missing(url, headers):
        raise _http_error(404)

    monkeypatch.setattr(update, "_request", missing)
    with pytest.raises(update.UpdateError, match="토큰이 필요"):
        update.check(None)
    with pytest.raises(update.UpdateError, match="권한"):
        update.check("tok")

    def offline(url, headers):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(update, "_request", offline)
    with pytest.raises(update.UpdateError, match="연결할 수 없습니다"):
        update.check("tok")


def test_download_checks_the_checksum(monkeypatch, tmp_path):
    payload = b"installer bytes" * 1000
    good = hashlib.sha256(payload).hexdigest()
    sums = {"value": f"{good}  KStruct-Setup-9.0.0.1.exe\n{'0' * 64}  other.dmg\n"}

    def fake(url, headers):
        assert headers["Accept"] == "application/octet-stream"
        return _Response(sums["value"].encode() if "SHA256" in url else payload)

    monkeypatch.setattr(update, "_request", fake)
    monkeypatch.setattr(update.platform, "system", lambda: "Windows")
    release = _release("KStruct-Setup-9.0.0.1.exe", "SHA256SUMS.txt")
    seen = []
    path = update.download(release, tmp_path, "tok", progress=lambda d, t: seen.append((d, t)))
    assert path.read_bytes() == payload and seen[-1] == (len(payload), len(payload))
    assert not list(tmp_path.glob("*.part"))

    sums["value"] = f"{'f' * 64} *KStruct-Setup-9.0.0.1.exe\n"
    with pytest.raises(update.UpdateError, match="체크섬"):
        update.download(release, tmp_path, "tok")
    assert not (tmp_path / "KStruct-Setup-9.0.0.1.exe").exists()  # a bad file is removed

    with pytest.raises(update.Cancelled):
        update.download(release, tmp_path, "tok", cancelled=lambda: True)

    with pytest.raises(update.UpdateError, match="설치 파일이 없습니다"):
        update.download(_release("KStruct-9.0.0.1-macOS.dmg"), tmp_path)


def test_token_does_not_follow_the_redirect_to_storage(monkeypatch):
    """GitHub answers an asset request with a redirect to a signed storage URL; the
    storage must not receive the token (it would reject two credentials)."""
    for name in ("no_proxy", "NO_PROXY"):  # the local test server, never via a proxy
        monkeypatch.setenv(name, "localhost,127.0.0.1")
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append((self.headers["Host"], self.path, self.headers.get("Authorization")))
            if self.path == "/asset":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{port}/signed?x=1")
                self.end_headers()
                return
            body = b"data"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        headers = update._headers("secret", "application/octet-stream")
        with update._request(f"http://localhost:{port}/asset", headers) as response:
            assert response.read() == b"data"
    finally:
        server.shutdown()
    assert seen[0][2] == "Bearer secret"  # the API request carries the token
    assert seen[1][1] == "/signed?x=1" and seen[1][2] is None  # the storage does not


def test_issue_link_and_sha_lookup():
    url = update.issue_url("제목", "본문 " * 3000)
    assert url.startswith("https://github.com/knoxxr/KStruct/issues/new?")
    assert len(url) < 40_000
    assert "KStruct" in update.environment_text()
    assert update.expected_sha256("ab  a.exe\ncd *b.dmg\n", "b.dmg") == "cd"
    assert update.expected_sha256("ab  a.exe\n", "c.exe") is None
    assert update.links()["sponsor"] == "https://github.com/sponsors/minarhub"
