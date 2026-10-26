"""Frictionless contact between parts (node-to-surface, penalty, active set).

Each node of the contact faces_b is paired with the closest point on faces_a. With master
outward normal n, the gap is g = g0 + n·(u_slave - Σ N_i u_master_i). Active pairs are
held at g = 0 by a stiff spring; a pair is released when its spring would pull (tension)
and re-engaged when it penetrates. Parts may slide freely along the contact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from kstruct.fem import tet10
from kstruct.fem.constraints import (
    ConstraintError,
    _project_to_triangles,
    _tri6_shape,
)
from kstruct.fem.mesh import FEMesh
from kstruct.model import Connection

MAX_ITERATIONS = 40
MAX_FLIPS = 3  # state changes allowed per pair within one solve (anti-chatter)
# Contact uses a softer penalty than ties (constraints.PENALTY_FACTOR = 1e6): penetration
# stays ~1e-3 of the contact displacement, and releasing a pair can then be done as a
# low-rank update of the factorization (with 1e6 the subtraction cancels catastrophically)
CONTACT_PENALTY_FACTOR = 1e3
log = logging.getLogger(__name__)


@dataclass
class ContactSet:
    dofs: np.ndarray  # (P, 21) slave xyz + 6 master nodes xyz
    coefs: np.ndarray  # (P, 21) gap gradient
    gap0: np.ndarray  # (P,) initial gap (mm), positive = open
    active: np.ndarray  # (P,) bool
    tol: float = 0.0  # penetration deeper than this engages a pair (mm, ~1e-6 edge)

    def gap(self, u: np.ndarray) -> np.ndarray:
        return self.gap0 + np.einsum("pi,pi->p", self.coefs, u[self.dofs])

    def penalty(self, n_dof: int, alpha: float) -> tuple[sp.csr_matrix, np.ndarray]:
        dofs, c = self.dofs[self.active], self.coefs[self.active]
        rows = np.repeat(dofs, 21, axis=1).ravel()
        cols = np.tile(dofs, (1, 21)).ravel()
        vals = alpha * np.einsum("pi,pj->pij", c, c).ravel()
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        f = np.zeros(n_dof)
        np.add.at(f, dofs.ravel(), (alpha * c * -self.gap0[self.active][:, None]).ravel())
        return K, f


def build_contact(mesh: FEMesh, connection: Connection) -> ContactSet:
    master = mesh.tris[mesh.face_tris(connection.faces_a)]
    slave_nodes = mesh.face_nodes(connection.faces_b)
    if len(master) == 0 or len(slave_nodes) == 0:
        raise ConstraintError("접촉할 면이 메시에 없습니다.")
    X = mesh.nodes
    A, B, C = X[master[:, 0]], X[master[:, 1]], X[master[:, 2]]
    _, normals, _ = tet10.tri6_load_weights(X[master])
    edge = np.linalg.norm(B - A, axis=1).mean()
    search = 0.5 * edge

    tree = cKDTree((A + B + C) / 3)
    k = min(8, len(master))
    _, cand = tree.query(X[slave_nodes], k=k)
    cand = cand.reshape(len(slave_nodes), k)

    dofs, coefs, gaps = [], [], []
    for node, tris in zip(slave_nodes, cand, strict=True):
        p = np.repeat(X[node][None], len(tris), axis=0)
        u, v, w, dist = _project_to_triangles(p, A[tris], B[tris], C[tris])
        best = int(np.argmin(dist))
        if dist[best] > search:
            continue
        t = tris[best]
        n = normals[t]
        N = _tri6_shape(u[best], v[best], w[best])
        point = u[best] * A[t] + v[best] * B[t] + w[best] * C[t]
        dofs.append(
            np.concatenate(
                [node * 3 + np.arange(3), (master[t][:, None] * 3 + np.arange(3)).ravel()]
            )
        )
        coefs.append(np.concatenate([n, (-N[:, None] * n).ravel()]))
        gaps.append((X[node] - point) @ n)
    if not dofs:
        raise ConstraintError("접촉면이 서로 너무 멀리 떨어져 있습니다.")
    gap0 = np.array(gaps)
    return ContactSet(np.array(dofs), np.array(coefs), gap0, gap0 <= 1e-6 * edge, 1e-6 * edge)


MAX_UPDATE_RANK = 400  # more changed pairs than this: factorize again


class ContactSolver:
    """Active-set loop that factorizes once and updates the factorization.

    Changing the active set adds or removes rank-one penalty terms, so the matrix of the
    next iteration is the factorized one plus a low-rank update K_c = K_f + Cᵀ D C (one
    row of C per changed pair, D = ±alpha). The Woodbury identity then solves it with k
    back-substitutions of the old factorization instead of a new factorization, which
    dominated the run time before (a bolted-plate example spent three minutes
    refactorizing an 82k-DOF system nine times). This only works with a moderate
    penalty: with alpha = 1e6 × stiffness the release update cancels catastrophically
    and preconditioned CG stalls, hence CONTACT_PENALTY_FACTOR.

    `factorize(K_reduced) -> solve(rhs)` supplies the exact solver (direct or AMG); the
    solver may accept a (n, k) right-hand side (SuperLU does) and is used column by
    column otherwise.
    """

    def __init__(
        self,
        K: sp.csr_matrix,
        free: np.ndarray,
        contacts: list[ContactSet],
        factorize,
        scale: float,
    ):
        self.n_dof = K.shape[0]
        self.K = K  # without contact terms: K u - f on a slave node is its contact force
        self.free = free
        self.contacts = contacts
        self.factorize = factorize
        self.alpha = CONTACT_PENALTY_FACTOR * scale
        self.position = np.full(self.n_dof, -1)
        self.position[free] = np.arange(len(free))
        # tiny springs everywhere keep parts that are only held by contact from being
        # singular while the active set settles; they carry a negligible share of the load
        self.K_ff = (
            K[free][:, free] + sp.identity(len(free), format="csr") * (1e-9 * scale)
        ).tocsr()
        self._fact_solve = None
        self._fact_active: list[np.ndarray] | None = None
        self._update = None  # (signature, C, Y, S) of the last Woodbury update
        self._t_factorize = np.inf  # measured cost of one factorization (s)
        self._t_solve = 0.0  # measured cost of one back-substitution (s)
        self._t_column = None  # measured cost per column of a batched back-substitution
        self.factorizations = 0
        self.updates = 0

    # --- assembly --------------------------------------------------------------------

    def _rows(self, cs: ContactSet, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reduced DOF numbers and coefficients of the pairs selected by mask (fixed DOFs
        get a zero coefficient)."""
        dofs, c = self.position[cs.dofs[mask]], cs.coefs[mask]
        c = np.where(dofs >= 0, c, 0.0)
        dofs = np.where(dofs >= 0, dofs, 0)
        return dofs, c

    def _penalty(self) -> tuple[sp.csr_matrix, np.ndarray]:
        rows, cols, vals = [], [], []
        f_c = np.zeros(len(self.free))
        for cs in self.contacts:
            dofs, c = self._rows(cs, cs.active)
            rows.append(np.repeat(dofs, 21, axis=1).ravel())
            cols.append(np.tile(dofs, (1, 21)).ravel())
            vals.append(self.alpha * np.einsum("pi,pj->pij", c, c).ravel())
            np.add.at(f_c, dofs.ravel(), (self.alpha * c * -cs.gap0[cs.active][:, None]).ravel())
        K_pen = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=self.K_ff.shape,
        )
        return K_pen, f_c

    def matrix(self) -> sp.csr_matrix:
        """Reduced matrix of the current active set."""
        return self.K_ff + self._penalty()[0]

    # --- linear solves ---------------------------------------------------------------

    def _change(self) -> tuple[sp.csr_matrix, np.ndarray] | None:
        """(C, D) with K_c = K_f + Cᵀ diag(D) C: one row per pair whose state differs from
        the factorized one, D = +alpha (engaged) or -alpha (released). None before the
        first factorization."""
        if self._fact_active is None:
            return None
        rows, cols, vals, signs = [], [], [], []
        n = len(self.free)
        k = 0
        for cs, old in zip(self.contacts, self._fact_active, strict=True):
            for mask, sign in ((cs.active & ~old, 1.0), (~cs.active & old, -1.0)):
                if not mask.any():
                    continue
                dofs, c = self._rows(cs, mask)
                m = len(dofs)
                rows.append(np.repeat(np.arange(k, k + m), 21))
                cols.append(dofs.ravel())
                vals.append(c.ravel())
                signs.append(np.full(m, sign))
                k += m
        if k == 0:
            return sp.csr_matrix((0, n)), np.zeros(0)
        C = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(k, n)
        )
        return C, self.alpha * np.concatenate(signs)

    def reduced_solve(self, K_c: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        import time

        change = self._change()
        k = change[0].shape[0] if change is not None else MAX_UPDATE_RANK + 1
        if k == 0:
            return self._timed_solve(rhs)
        # the update costs about k back-substitutions; refactorize when that would take
        # longer than a fresh factorization (small systems factorize fast)
        per_column = self._t_column if self._t_column is not None else 5 * self._t_solve
        log.debug(
            "contact solve: %d changed pairs, update ~%.1fs vs factorize %.1fs",
            k,
            k * per_column,
            self._t_factorize,
        )
        if k <= MAX_UPDATE_RANK and k * per_column < self._t_factorize:
            x = self._woodbury(K_c, rhs, *change)
            if x is not None:
                return x
        t0 = time.perf_counter()
        self._fact_solve = self.factorize(K_c)
        self._t_factorize = time.perf_counter() - t0
        self._fact_active = [cs.active.copy() for cs in self.contacts]
        self._update = None
        self.factorizations += 1
        x = self._timed_solve(rhs)
        if self._t_column is None:  # calibrate the batched cost on two columns
            self._solve_columns(np.column_stack([rhs, rhs]))
        return x

    def _timed_solve(self, rhs: np.ndarray) -> np.ndarray:
        import time

        t0 = time.perf_counter()
        x = self._fact_solve(rhs)
        self._t_solve = max(time.perf_counter() - t0, 1e-6)
        return x

    def _solve_columns(self, B: np.ndarray) -> np.ndarray:
        import time

        t0 = time.perf_counter()
        try:
            Y = self._fact_solve(np.asfortranarray(B))
            if Y.shape != B.shape:
                raise ValueError
        except Exception:  # solver only takes vectors
            Y = np.column_stack([self._fact_solve(B[:, j]) for j in range(B.shape[1])])
        # SuperLU solves a block of right-hand sides column by column; measure the real
        # per-column cost (several times a single warm solve) for the update/refactorize
        # decision
        self._t_column = (time.perf_counter() - t0) / max(B.shape[1], 1)
        return np.asarray(Y)

    def _woodbury(self, K_c, rhs: np.ndarray, C: sp.csr_matrix, D: np.ndarray):
        """x = K_c⁻¹ rhs with K_c = K_f + Cᵀ diag(D) C and K_f factorized."""
        signature = tuple(cs.active.tobytes() for cs in self.contacts)
        if self._update is None or self._update[0] != signature:
            Y = self._solve_columns(C.T.toarray())  # (n, k)
            S = np.diag(1.0 / D) + C @ Y  # (k, k)
            try:
                if np.linalg.cond(S) > 1e12:
                    return None
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return None
            self._update = (signature, C, Y, S_inv)
        _, C, Y, S_inv = self._update
        y = self._fact_solve(rhs)
        x = y - Y @ (S_inv @ (C @ y))
        # Measured on a lap joint: the update matches a fresh factorization to 2e-7
        # (the factorization's own accuracy). The residual of K_c cannot be used as a
        # check and iterative refinement diverges, because the penalty rows (alpha ~ 1e13)
        # amplify round-off far beyond what the displacements need; so the guard is the
        # conditioning of the small system plus finiteness.
        if not np.all(np.isfinite(x)):
            return None
        self.updates += 1
        return x

    # --- active-set loop -------------------------------------------------------------

    def solve(self, f: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
        """Settle the active set under the full-size load f. Returns the displacement and
        the reduced matrix of the final contact state."""
        f_f = f[self.free]
        load_scale = max(np.abs(f).max(), 1e-30)
        u = np.zeros(self.n_dof)
        K_c = self.K_ff
        changed = True
        seen = {tuple(cs.active.tobytes() for cs in self.contacts)}
        flips = [np.zeros(len(cs.active), dtype=int) for cs in self.contacts]
        for _ in range(MAX_ITERATIONS):
            K_pen, f_c = self._penalty()
            K_c = self.K_ff + K_pen
            u = np.zeros(self.n_dof)
            u[self.free] = self.reduced_solve(K_c, f_f + f_c)
            # Decide by the spring force, not the gap: a stiff penalty keeps even a
            # pulling pair's gap at round-off size. The force is taken from equilibrium
            # (K u - f on the slave node equals the spring force) rather than alpha * gap,
            # which would amplify solver round-off. Release a pair when it pulls with more
            # than a negligible share of the contact forces; engage an open pair when it
            # penetrates noticeably on the scale of the displacements. Pairs carrying
            # next to nothing would otherwise chatter between the two states for many
            # iterations, so each pair may change state only a few times per solve.
            residual = self.K @ u - f
            geom_tol = 1e-6 * max(np.abs(u).max(), 1e-30)
            changed = False
            for cs, count in zip(self.contacts, flips, strict=True):
                g = cs.gap(u)
                # K u - f on the slave node is -alpha*g*n (the spring pushing it back)
                spring = -np.einsum("pi,pi->p", cs.coefs[:, :3], residual[cs.dofs[:, :3]])
                # release threshold: 0.1 % of the largest contact pressure. Pairs far from
                # the loaded area carry round-off-sized forces of either sign; releasing
                # them on a tiny tension only to re-engage them one iteration later used
                # to take ten extra factorizations on a bolted plate pair.
                pressure = -spring[cs.active & (spring < 0)]
                force_tol = max(1e-9 * load_scale, 1e-3 * (pressure.max() if len(pressure) else 0))
                new_active = np.where(cs.active, spring <= force_tol, g < -max(cs.tol, geom_tol))
                frozen = count >= MAX_FLIPS
                new_active = np.where(frozen, cs.active, new_active)
                flipped = new_active != cs.active
                count += flipped
                changed |= bool(np.any(flipped))
                cs.active = new_active
            if not changed:
                break
            signature = tuple(cs.active.tobytes() for cs in self.contacts)
            if signature in seen:  # chattering between two states: keep the current one
                log.info("contact: active set cycles, stopping")
                break
            seen.add(signature)
        log.info(
            "contact: %d iterations, %d factorizations, %d low-rank updates",
            MAX_ITERATIONS if changed else _ + 1,
            self.factorizations,
            self.updates,
        )
        return u, K_c

    def current_solver(self):
        """Exact solver for the current active set on full-size right-hand sides (used for
        the unit responses of the bolt preload)."""
        K_c = self.matrix()

        def run(rhs: np.ndarray) -> np.ndarray:
            out = np.zeros(self.n_dof)
            out[self.free] = self.reduced_solve(K_c, rhs[self.free])
            return out

        return run


def solve_with_contact(K, f, free, contacts, solve, scale):
    """Compatibility wrapper around ContactSolver with a plain `solve(K_free, f_free)`."""
    solver = ContactSolver(K, free, contacts, lambda A: lambda b: solve(A, b), scale)
    return solver.solve(f)
