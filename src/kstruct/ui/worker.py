"""Run long jobs (meshing, solving) off the UI thread."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(object)  # the exception


class _Job(QRunnable):
    def __init__(self, fn: Callable, signals: _Signals) -> None:
        super().__init__()
        self.fn = fn
        self.signals = signals

    def run(self) -> None:
        try:
            result = self.fn(lambda frac, msg: self.signals.progress.emit(frac, msg))
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
) -> None:
    """fn receives a progress callback; callbacks run on the UI thread."""
    signals = _Signals()
    _alive.add(signals)

    def finish(handler, value):
        _alive.discard(signals)
        handler(value)

    signals.done.connect(lambda r: finish(on_done, r))
    signals.failed.connect(lambda e: finish(on_failed, e))
    if on_progress:
        signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(_Job(fn, signals))
