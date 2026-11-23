"""Intel oneMKL PARDISO for symmetric positive definite systems, through ctypes.

SciPy's SuperLU factorizes on one core with a minimum-degree ordering that fills badly
on 3D meshes: a 85k-DOF bracket took 16 s and a 73k-DOF bolted joint 52 s to factorize.
PARDISO (METIS nested dissection, supernodal, multithreaded) does both in under a second
with the same answers (1e-11). MKL ships as the `mkl` wheel on Windows and Linux x86-64;
elsewhere, or if the library cannot be loaded, `available()` is False and callers keep
using SuperLU.

Only the SPD path (mtype 2) is used. A matrix that is not positive definite (a part not
held, an indefinite shift in buckling) makes the factorization fail with error -4, and
the caller falls back to SuperLU, which then diagnoses it as before.

Results must not depend on thread timing (friction is path dependent and once swung by
an order of magnitude with round-off): conditional numerical reproducibility is switched
on before the first call.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import logging
import os
import sys
import weakref
from collections import OrderedDict
from functools import cache

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

SPD = 2
_PHASE_ANALYZE_FACTOR = 12
_PHASE_SOLVE = 33
_PHASE_RELEASE = -1
MKL_CBWR_COMPATIBLE = 2  # bit-reproducible across runs and thread counts


class PardisoError(RuntimeError):
    def __init__(self, code: int):
        super().__init__(f"PARDISO error {code}")
        self.code = code


def _candidates() -> list[str]:
    env = os.environ.get("KSTRUCT_MKL_RT")
    if env:
        return [env]
    if sys.platform == "win32":
        names = ["mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll"]
    else:
        names = ["libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so"]
    roots = [sys.prefix, getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)]
    found: list[str] = []
    for root in filter(None, roots):
        for sub in ("", "Library/bin", "lib", "_internal", "_internal/Library/bin"):
            for name in names:
                found += glob.glob(os.path.join(root, sub, name))
    return found + names  # bare names: let the loader search its own path


@cache
def _library():
    if sys.platform == "darwin" or os.environ.get("KSTRUCT_NO_PARDISO"):
        return None
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
            lib.pardiso.restype = None
        except (OSError, AttributeError):
            continue
        try:
            lib.MKL_CBWR_Set.restype = ctypes.c_int
            lib.MKL_CBWR_Set(MKL_CBWR_COMPATIBLE)
        except AttributeError:
            log.warning("MKL without CBWR control: results may vary in the last digits")
        log.info("PARDISO from %s", path)
        return lib
    return None


def available() -> bool:
    return _library() is not None


# METIS ordering is ~75 % of a factorization (85k DOF: 0.8 s of 1.05 s) and depends only
# on the sparsity pattern, which repeats: every transient heat step, contact iterations
# whose active set comes back, load cases. Remember the last few orderings by pattern.
_ORDERINGS: OrderedDict[bytes, np.ndarray] = OrderedDict()
_ORDERINGS_KEPT = 4


def _pattern_key(upper: sp.csr_matrix) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(upper.shape[0]).tobytes())
    h.update(upper.indptr.tobytes())
    h.update(upper.indices.tobytes())
    return h.digest()


class _Factor:
    """One factorization; free() releases MKL's memory (also on garbage collection)."""

    def __init__(self, K: sp.spmatrix):
        lib = _library()
        if lib is None:
            raise PardisoError(-999)
        upper = sp.triu(K, format="csr")
        upper.sort_indices()
        n = upper.shape[0]
        # PARDISO needs every diagonal entry present, even when zero
        if np.any(upper.indices[upper.indptr[:-1]] != np.arange(n)):
            upper = (upper + sp.eye(n, format="csr") * 0.0).tocsr()
            upper.sort_indices()
        self.n = n
        self.a = np.ascontiguousarray(upper.data, dtype=np.float64)
        self.ia = np.ascontiguousarray(upper.indptr, dtype=np.int32)
        self.ja = np.ascontiguousarray(upper.indices, dtype=np.int32)
        self.pt = np.zeros(64, dtype=np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        ip = self.iparm
        ip[0] = 1  # supply all values below
        ip[1] = 3  # METIS nested dissection, parallel (0.76 -> 0.45 s at 85k, same order)
        key = _pattern_key(upper)
        known = _ORDERINGS.get(key)
        if known is not None:
            _ORDERINGS.move_to_end(key)
            self.perm = known.copy()
            ip[4] = 1  # use this ordering
        else:
            self.perm = np.zeros(n, dtype=np.int32)
            ip[4] = 2  # return the computed ordering in perm
        ip[7] = 0  # refine only if pivots were perturbed (never for SPD): 1e-12 as is
        ip[9] = 13  # pivot perturbation 1e-13 (not used for SPD)
        ip[17] = -1  # report nonzeros in the factor
        ip[20] = 1  # Bunch-Kaufman (not used for SPD)
        ip[26] = 0  # no matrix checker (indices are sorted above)
        ip[34] = 1  # zero-based indices
        ip[55] = 1  # keep the factor's diagonal (pivots) for the singularity check
        self._lib = lib
        self._finalizer = weakref.finalize(self, _release, lib, self.pt, n)
        self._call(_PHASE_ANALYZE_FACTOR, None)
        self.nnz_factor = int(ip[17])
        if known is None:
            _ORDERINGS[key] = self.perm.copy()
            while len(_ORDERINGS) > _ORDERINGS_KEPT:
                _ORDERINGS.popitem(last=False)

    def pivots(self) -> tuple[np.ndarray, np.ndarray]:
        """(diagonal of the factor, diagonal of the matrix), as pardiso_getdiag gives them."""
        df = np.zeros(self.n)
        da = np.zeros(self.n)
        error = ctypes.c_int32(0)
        self._lib.pardiso_getdiag(
            self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            df.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            da.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int32(1)),
            ctypes.byref(error),
        )
        if error.value != 0:
            raise PardisoError(error.value)
        return df, da

    def _call(self, phase: int, b: np.ndarray | None) -> np.ndarray | None:
        nrhs = 1 if b is None or b.ndim == 1 else b.shape[1]
        if b is None:
            bb = np.zeros(1)
            x = np.zeros(1)
        else:
            bb = np.asfortranarray(b, dtype=np.float64)
            x = np.zeros_like(bb)
        error = ctypes.c_int32(0)
        i32 = ctypes.POINTER(ctypes.c_int32)
        f64 = ctypes.POINTER(ctypes.c_double)
        self._lib.pardiso(
            self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(ctypes.c_int32(1)),  # maxfct
            ctypes.byref(ctypes.c_int32(1)),  # mnum
            ctypes.byref(ctypes.c_int32(SPD)),
            ctypes.byref(ctypes.c_int32(phase)),
            ctypes.byref(ctypes.c_int32(self.n)),
            self.a.ctypes.data_as(f64),
            self.ia.ctypes.data_as(i32),
            self.ja.ctypes.data_as(i32),
            self.perm.ctypes.data_as(i32),
            ctypes.byref(ctypes.c_int32(nrhs)),
            self.iparm.ctypes.data_as(i32),
            ctypes.byref(ctypes.c_int32(0)),  # msglvl
            bb.ctypes.data_as(f64),
            x.ctypes.data_as(f64),
            ctypes.byref(error),
        )
        if error.value != 0:
            self.free()
            raise PardisoError(error.value)
        return None if b is None else np.ascontiguousarray(x)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self._call(_PHASE_SOLVE, np.asarray(b, dtype=np.float64))

    def free(self) -> None:
        self._finalizer()


def _release(lib, pt: np.ndarray, n: int) -> None:
    if not pt.any():
        return
    error = ctypes.c_int32(0)
    dummy = np.zeros(1)
    idx = np.zeros(1, dtype=np.int32)
    iparm = np.zeros(64, dtype=np.int32)
    i32 = ctypes.POINTER(ctypes.c_int32)
    f64 = ctypes.POINTER(ctypes.c_double)
    lib.pardiso(
        pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(ctypes.c_int32(1)),
        ctypes.byref(ctypes.c_int32(1)),
        ctypes.byref(ctypes.c_int32(SPD)),
        ctypes.byref(ctypes.c_int32(_PHASE_RELEASE)),
        ctypes.byref(ctypes.c_int32(n)),
        dummy.ctypes.data_as(f64),
        idx.ctypes.data_as(i32),
        idx.ctypes.data_as(i32),
        None,
        ctypes.byref(ctypes.c_int32(1)),
        iparm.ctypes.data_as(i32),
        ctypes.byref(ctypes.c_int32(0)),
        dummy.ctypes.data_as(f64),
        dummy.ctypes.data_as(f64),
        ctypes.byref(error),
    )
    pt[:] = 0


def factorize(K: sp.spmatrix) -> _Factor:
    """Cholesky-type factorization of the SPD matrix K (full or upper storage)."""
    return _Factor(K)
