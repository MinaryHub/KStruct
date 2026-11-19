"""Update check / download / install UI, and the contact and sponsor links.

Network work runs on a plain Python thread and reports back through Qt signals (queued
to the GUI thread), so the window stays responsive and nothing blocks on a slow proxy.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from kstruct import __version__, update
from kstruct.paths import data_dir
from kstruct.ui import prefs
from kstruct.ui.help import with_help

log = logging.getLogger(__name__)


class _Task(QObject):
    """Runs fn(report, cancelled) on a thread; report(done, total) drives `progress`."""

    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, fn: Callable, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="kstruct-update").start()

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self) -> None:
        try:
            value = self._fn(self.progress.emit, self._cancel.is_set)
        except update.Cancelled:
            self.failed.emit("")
        except update.UpdateError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 (reported, never crashes the app)
            log.exception("update task failed")
            self.failed.emit(f"예상하지 못한 오류: {e}")
        else:
            self.done.emit(value)


def open_url(url: str) -> bool:
    return QDesktopServices.openUrl(QUrl(url))


def open_contact(parent: QWidget | None = None, detail: str = "") -> None:
    """New GitHub issue with the version and environment filled in."""
    home = str(Path.home())
    if detail and len(home) > 3:
        detail = detail.replace(home, "~")  # paths would show the user's account name
    body = (
        "### 무엇을 하다가 어떤 일이 있었나요?\n\n\n"
        "### 기대한 동작\n\n\n"
        "### 환경\n"
        f"{update.environment_text()}\n"
        + (f"\n### 오류 내용\n```\n{detail[-1500:]}\n```\n" if detail else "")
        + "\n(로그와 프로젝트가 필요하면 '문제 보고'로 만든 zip 파일을 끌어다 붙여 주세요.)\n"
    )
    if not open_url(update.issue_url("", body)):
        QMessageBox.information(
            parent,
            "문의하기",
            f"브라우저를 열 수 없습니다. 아래 주소로 문의해 주세요.\n\n{update.links()['issues']}",
        )


def open_sponsor(parent: QWidget | None = None) -> None:
    url = update.links().get("sponsor", "")
    if not url or not open_url(url):
        QMessageBox.information(parent, "후원하기", f"브라우저를 열 수 없습니다.\n\n{url}")


class UpdateDialog(QDialog):
    """Check for a newer release, show its notes, download and install it."""

    def __init__(self, parent: QWidget | None = None, release=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("업데이트")
        self.resize(620, 520)
        self.release = release
        self._task: _Task | None = None
        layout = QVBoxLayout(self)
        self.current = QLabel(f"지금 버전: <b>{__version__}</b>")
        layout.addWidget(self.current)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        layout.addWidget(self.notes, 1)

        self.token = QLineEdit(prefs.github_token())
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("github_pat_… (저장소 읽기 권한)")
        if update.TOKEN_ENV in os.environ:
            self.token.setPlaceholderText(f"환경 변수 {update.TOKEN_ENV} 사용 중")
        row = QHBoxLayout()
        row.addWidget(QLabel("GitHub 토큰"))
        row.addWidget(self.token, 1)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(with_help(holder, "help.update_token"))
        self.auto = QCheckBox("시작할 때 하루 한 번 자동으로 확인")
        self.auto.setChecked(prefs.auto_update())
        self.auto.toggled.connect(prefs.set_auto_update)
        layout.addWidget(self.auto)

        buttons = QHBoxLayout()
        self.check_button = QPushButton("다시 확인")
        self.check_button.clicked.connect(self.check)
        self.page_button = QPushButton("릴리스 페이지")
        self.page_button.clicked.connect(self.open_page)
        self.install_button = QPushButton("받아서 설치")
        self.install_button.setProperty("primary", "true")
        self.install_button.clicked.connect(self.install)
        close = QPushButton("닫기")
        close.clicked.connect(self.reject)
        for b in (self.check_button, self.page_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.install_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        if release is not None:
            self._show_release(release)
        else:
            self.check()

    # --- check ----------------------------------------------------------------------

    def _token(self) -> str | None:
        value = self.token.text().strip()
        if value != prefs.github_token():
            prefs.set_github_token(value)
        return update.token(value)

    def check(self) -> None:
        tok = self._token()
        self.install_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.status.setText("새 버전을 확인하는 중…")
        self.notes.clear()
        self._task = _Task(lambda report, cancelled: update.latest_release(tok), self)
        self._task.done.connect(self._checked)
        self._task.failed.connect(self._check_failed)
        self._task.start()

    def _checked(self, release) -> None:
        self.check_button.setEnabled(True)
        prefs.mark_update_checked()
        try:
            newer = update.is_newer(release.tag)
        except ValueError:
            newer = False
        if newer:
            self._show_release(release)
        else:
            self.release = None
            self.status.setText(
                f"최신 버전을 쓰고 있습니다. (최신 릴리스: {release.tag or '없음'})"
            )
            self.notes.setMarkdown(release.notes or "")

    def _check_failed(self, message: str) -> None:
        self.check_button.setEnabled(True)
        self.status.setText(f"<span>확인하지 못했습니다: {message}</span>")

    def _show_release(self, release) -> None:
        self.release = release
        asset = update.pick_installer(release)
        self.status.setText(
            f"새 버전 <b>{release.version}</b>이 있습니다."
            + ("" if asset else " 다만 이 운영체제용 설치 파일이 릴리스에 없습니다.")
        )
        self.notes.setMarkdown(release.notes or "(릴리스 설명 없음)")
        self.install_button.setEnabled(asset is not None)
        self.install_button.setText("받아서 설치" if update.frozen() else "설치 파일 받기")

    def open_page(self) -> None:
        open_url(self.release.page if self.release else update.links()["releases"])

    # --- download & install ---------------------------------------------------------

    def install(self) -> None:
        release, tok = self.release, self._token()
        if release is None:
            return
        dialog = QProgressDialog("설치 파일을 받는 중…", "취소", 0, 100, self)
        dialog.setWindowTitle("업데이트")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        folder = data_dir() / "updates"
        task = _Task(
            lambda report, cancelled: update.download(release, folder, tok, report, cancelled),
            self,
        )
        self._task = task

        def progress(done: int, total: int) -> None:
            if total:
                dialog.setMaximum(100)
                dialog.setValue(int(100 * done / total))
                dialog.setLabelText(
                    f"설치 파일을 받는 중… {done / 1e6:,.1f} / {total / 1e6:,.1f} MB"
                )
            else:
                dialog.setMaximum(0)

        def finished(path) -> None:
            dialog.close()
            self._downloaded(path)

        def failed(message: str) -> None:
            dialog.close()
            if message:
                QMessageBox.warning(self, "업데이트", message)

        task.progress.connect(progress)
        task.done.connect(finished)
        task.failed.connect(failed)
        dialog.canceled.connect(task.cancel)
        task.start()
        dialog.show()

    def _downloaded(self, path) -> None:
        if not update.frozen():
            QMessageBox.information(
                self,
                "업데이트",
                f"소스에서 실행 중이라 설치하지 않고 받아만 두었습니다.\n\n{path}",
            )
            update.launch_installer(path)  # opens the folder / disk image
            return
        if _will_quit():
            answer = QMessageBox.question(
                self,
                "업데이트",
                "설치하려면 KStruct를 닫습니다. 저장하지 않은 작업이 있으면 먼저 저장할지 "
                "묻습니다. 설치가 끝나면 KStruct가 다시 열립니다.\n\n지금 설치할까요?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            window = self.parent()
            self.accept()
            if window is not None and not window.close():
                return  # the user cancelled closing: do not replace the running app
            update.launch_installer(path)
            QApplication.instance().quit()
            return
        update.launch_installer(path)
        QMessageBox.information(
            self,
            "업데이트",
            "설치 파일을 열었습니다. 안내에 따라 설치한 뒤 KStruct를 다시 시작하세요.",
        )


def _will_quit() -> bool:
    """Windows installs in place (the app must close); elsewhere the user installs."""
    return platform.system() == "Windows"


class UpdateChecker(QObject):
    """Quiet start-up check: emits `available(release)` when a newer release exists.
    Failures (offline, no token for the private repository) are only logged."""

    available = Signal(object)

    def start(self, force: bool = False) -> bool:
        if not force and not (
            prefs.auto_update() and prefs.update_check_due(update.CHECK_INTERVAL_HOURS)
        ):
            return False
        tok = update.token(prefs.github_token())
        task = _Task(lambda report, cancelled: update.check(tok), self)
        task.done.connect(self._done)
        task.failed.connect(lambda message: log.info("update check: %s", message))
        self._task = task
        task.start()
        return True

    def _done(self, release) -> None:
        prefs.mark_update_checked()
        if release is not None:
            self.available.emit(release)
