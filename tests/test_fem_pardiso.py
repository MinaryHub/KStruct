"""PARDISO (MKL) direct solver: same answers as SuperLU, singular systems still reported,
bit-reproducible, and everything works the same when MKL is not available."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kstruct.fem import SolverError, assembly, generate_mesh, pardiso, solve_static
from kstruct.geometry.face_info import face_at
from kstruct.geometry.samples import _box, l_bracket
from kstruct.model import (
    Connection,
    ConnectionKind,
    FaceRef,
    Fixture,
    FixtureKind,
    Load,
    LoadKind,
    Material,
    Part,
    Project,
)

needs_mkl = pytest.mark.skipif(not pardiso.available(), reason="MKL/PARDISO not available")

STEEL = Material(
    id="steel",
    name="steel",
    category="steel",
    youngs_modulus=210000.0,
    poisson_ratio=0.3,
    density=7.85e-9,
    yield_strength=235.0,
    tensile_strength=360.0,
)


def _laplacian(n: int, shift: float) -> sp.csr_matrix:
    A = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).tolil()
    A[0, 0] = A[-1, -1] = 1.0 + shift  # free ends: singular without the shift
    return A.tocsr()


@needs_mkl
def test_pardiso_matches_superlu_and_is_reproducible():
    rng = np.random.default_rng(1)
    n = 3000
    B = sp.random(n, n, density=4 / n, random_state=2)
    K = (B @ B.T + sp.eye(n) * 0.1 + _laplacian(n, 1.0)).tocsr()
    b = rng.standard_normal((n, 3))
    reference = spla.splu(K.tocsc()).solve(b)
    factor = pardiso.factorize(K)
    x1, x2 = factor.solve(b), factor.solve(b)
    again = pardiso.factorize(K).solve(b)
    assert np.abs(x1 - reference).max() < 1e-9 * np.abs(reference).max()
    assert np.array_equal(x1, x2) and np.array_equal(x1, again)
    assert factor.nnz_factor > 0


@needs_mkl
def test_singular_and_indefinite_matrices_are_left_to_superlu(monkeypatch):
    monkeypatch.setattr(assembly, "PARDISO_MIN_DOF", 10)
    singular = _laplacian(3000, 0.0)  # constant vector in the null space: a free body
    assert assembly._pardiso_solver(singular) is None
    indefinite = sp.diags(np.r_[np.ones(2999), -1.0]).tocsr()
    assert assembly._pardiso_solver(indefinite) is None
    held = _laplacian(3000, 1e-9)  # like the contact stabilising springs: still accepted
    assert assembly._pardiso_solver(held) is not None


def _bracket(load=-2000.0) -> Project:
    shape = l_bracket()
    project = Project(parts=[Part(0, "bracket", shape, "steel")])
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(shape, (0, 30, 40)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(0, face_at(shape, (90, 30, 8)))], vector=(0, 0, load))
    ]
    return project


def _stack(kind) -> Project:
    lower, upper = _box(0, 0, 0, 40, 20, 10), _box(0, 0, 10, 40, 20, 20)
    project = Project(parts=[Part(0, "a", lower, "steel"), Part(1, "b", upper, "steel")])
    project.connections = [
        Connection(
            kind,
            [FaceRef(0, face_at(lower, (20, 10, 10)))],
            [FaceRef(1, face_at(upper, (20, 10, 10)))],
        )
    ]
    project.fixtures = [Fixture(FixtureKind.FIXED, [FaceRef(0, face_at(lower, (20, 10, 0)))])]
    project.loads = [
        Load(LoadKind.FORCE, [FaceRef(1, face_at(upper, (20, 10, 20)))], vector=(0, 0, -800.0))
    ]
    return project


@needs_mkl
@pytest.mark.parametrize("make", [_bracket, lambda: _stack(ConnectionKind.CONTACT)])
def test_analyses_agree_with_and_without_pardiso(monkeypatch, make):
    project = make()
    mesh = generate_mesh(project, "coarse")
    materials = {p.id: STEEL for p in project.parts}
    monkeypatch.setattr(assembly, "PARDISO_MIN_DOF", 0)
    fast = solve_static(project, mesh, materials)
    monkeypatch.setattr(pardiso, "available", lambda: False)
    slow = solve_static(project, mesh, materials)
    scale = np.abs(slow.displacement).max()
    assert np.abs(fast.displacement - slow.displacement).max() < 1e-6 * scale
    assert np.abs(fast.von_mises - slow.von_mises).max() < 1e-5 * slow.von_mises.max()
    for a, b in zip(fast.reactions, slow.reactions, strict=True):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-6)


@needs_mkl
def test_unheld_part_is_still_reported(monkeypatch):
    """Lower block fixed, upper block resting on it with no connection: nothing holds the
    upper block, PARDISO declines the singular matrix and the SuperLU path reports it."""
    project = _stack(ConnectionKind.BONDED)
    project.connections = []
    mesh = generate_mesh(project, "coarse")
    monkeypatch.setattr(assembly, "PARDISO_MIN_DOF", 0)
    with pytest.raises(SolverError, match="unstable"):
        solve_static(project, mesh, {0: STEEL, 1: STEEL})
