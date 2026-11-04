"""Cancelling a background mesh / solve job."""

import threading

import pytest

from kstruct.fem import generate_mesh
from kstruct.geometry.samples import l_bracket
from kstruct.model import Part, Project
from kstruct.ui.worker import Cancelled, run_job


def test_cancelled_job_reports_cancelled_and_drops_its_result(qtbot):
    started, release = threading.Event(), threading.Event()
    outcome = {}

    def job(progress):
        progress(0.1, "first stage")
        started.set()
        release.wait(5)
        progress(0.5, "second stage")  # raises: the job was cancelled meanwhile
        return "finished"

    handle = run_job(job, lambda r: outcome.update(done=r), lambda e: outcome.update(failed=e))
    assert started.wait(5)
    handle.cancel()
    release.set()
    qtbot.waitUntil(lambda: bool(outcome), timeout=5000)
    assert "done" not in outcome
    assert isinstance(outcome["failed"], Cancelled)


def test_meshing_stops_when_its_check_raises():
    """The parts are meshed in worker processes; a cancel must not wait for Netgen and
    must come out as the caller's own exception, not as a meshing failure."""
    project = Project(parts=[Part(0, "bracket", l_bracket(), "s235")])
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 1:
            raise Cancelled()

    with pytest.raises(Cancelled):
        generate_mesh(project, "very_fine", check=check)
