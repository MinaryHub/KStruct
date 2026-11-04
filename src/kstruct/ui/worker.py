"""Run long jobs (meshing, solving) off the UI thread."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class Cancelled(Exception):
    """The user cancelled the job (raised at the job's next progress report)."""


class _Signals(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(object)  # the exception


class JobHandle:
    """Lets the UI cancel a running job. Cancelling is cooperative: the job stops at its
    next progress report (or the next poll while meshing), not in the middle of a
    factorization."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled()


class _Job(QRunnable):
    def __init__(self, fn: Callable, signals: _Signals, handle: JobHandle) -> None:
        super().__init__()
        self.fn = fn
        self.signals = signals
        self.handle = handle

    def run(self) -> None:
        def progress(frac: float, msg: str) -> None:
            self.handle.check()
            self.signals.progress.emit(frac, msg)

        progress.check = self.handle.check  # for loops that poll without reporting

        try:
            self.handle.check()
            result = self.fn(progress)
            self.handle.check()  # cancelled during the last stage: drop the result
        except Cancelled as e:
            logging.getLogger("kstruct.worker").info("background job cancelled")
            self.signals.failed.emit(e)
        except Exception as e:  # report everything to the UI; never kill the worker silently
            logging.getLogger("kstruct.worker").exception("background job failed")
            self.signals.failed.emit(e)
        else:
            self.signals.done.emit(result)


_alive: set[_Signals] = set()


def run_job(
    fn: Callable[[Callable[[float, str], None]], object],
    on_done: Callable[[object], None],
    on_failed: Callable[[Exception], None],
    on_progress: Callable[[float, str], None] | None = None,
) -> JobHandle:
    """fn receives a progress callback; callbacks run on the UI thread. The progress
    callback raises Cancelled once the returned handle is cancelled; on_failed then
    receives that Cancelled."""
    signals = _Signals()
    handle = JobHandle()
    _alive.add(signals)

    def finish(handler, value):
        _alive.discard(signals)
        handler(value)

    signals.done.connect(lambda r: finish(on_done, r))
    signals.failed.connect(lambda e: finish(on_failed, e))
    if on_progress:
        signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(_Job(fn, signals, handle))
    return handle
