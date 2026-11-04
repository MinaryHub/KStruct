"""Contact between parts (node-to-surface, penalty, active set, Coulomb friction).

Each node of the contact faces_b is paired with the closest point on faces_a. With master
outward normal n, the gap is g = g0 + n·(u_slave - Σ N_i u_master_i). Active pairs are
held at g = 0 by a stiff spring; a pair is released when its spring would pull (tension)
and re-engaged when it penetrates.

With a friction coefficient μ > 0 every closed pair also starts *sticking*: tangential
springs of the same stiffness hold the relative sliding s = t·(u_slave − Σ N_i u_master_i)
at zero. When the tangential force of a sticking pair exceeds μ·N (N = normal force) the
pair *slips*: its tangential springs are removed and a constant friction force μ·N
opposing the sliding it resisted is applied instead (updated with N each iteration); it
sticks again when it moves back past the point where it stuck. Without friction parts
slide freely along the contact.
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
MAX_FLIPS = 3  # open/closed changes allowed per pair within one solve (anti-chatter)
MAX_FRICTION_FLIPS = 8  # stick/slip changes allowed per pair within one solve
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
    friction: float = 0.0  # Coulomb coefficient μ (0: frictionless)
    tcoefs: np.ndarray | None = None  # (P, 2, 21) sliding gradients along two tangents
    stick: np.ndarray | None = None  # (P,) bool, meaningful for active pairs
    slip_force: np.ndarray | None = None  # (P, 2) friction force on the slave (N) when slipping
    anchor: np.ndarray | None = None  # (P, 2) sliding at which the tangential springs are relaxed
    slip_dir: np.ndarray | None = None  # (P, 2) unit direction of the friction force once slipped

    def __post_init__(self) -> None:
        n = len(self.active)
        if self.stick is None:
            self.stick = np.ones(n, dtype=bool)
        if self.slip_force is None:
            self.slip_force = np.zeros((n, 2))
        if self.anchor is None:
            self.anchor = np.zeros((n, 2))
        if self.slip_dir is None:
            self.slip_dir = np.zeros((n, 2))

    @property
    def has_friction(self) -> bool:
        return self.friction > 0 and self.tcoefs is not None

    @property
    def sticking(self) -> np.ndarray:
        """Pairs that carry tangential springs."""
        if not self.has_friction:
            return np.zeros(len(self.active), dtype=bool)
        return self.active & self.stick

    @property
    def slipping(self) -> np.ndarray:
        if not self.has_friction:
            return np.zeros(len(self.active), dtype=bool)
        return self.active & ~self.stick

    def reset_friction(self) -> None:
        """Every pair sticks again with relaxed springs (a new load is a new solution, not
        an increment: the stick/slip pattern and the slip directions found under another
        load would only mislead the first iterations)."""
        self.stick[:] = True
        self.slip_force[:] = 0.0
        self.anchor[:] = 0.0
        self.slip_dir[:] = 0.0

    def gap(self, u: np.ndarray) -> np.ndarray:
        return self.gap0 + np.einsum("pi,pi->p", self.coefs, u[self.dofs])

    def sliding(self, u: np.ndarray) -> np.ndarray:
        """(P, 2) relative tangential displacement of the slave against the master."""
        return np.einsum("pki,pi->pk", self.tcoefs, u[self.dofs])

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

    dofs, coefs, tcoefs, gaps = [], [], [], []
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
        tcoefs.append([np.concatenate([t, (-N[:, None] * t).ravel()]) for t in _tangents(n)])
        gaps.append((X[node] - point) @ n)
    if not dofs:
        raise ConstraintError("접촉면이 서로 너무 멀리 떨어져 있습니다.")
    gap0 = np.array(gaps)
    return ContactSet(
        np.array(dofs),
        np.array(coefs),
        gap0,
        gap0 <= 1e-6 * edge,
        1e-6 * edge,
        friction=max(float(connection.friction), 0.0),
        tcoefs=np.array(tcoefs),
    )


def _tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to n."""
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(n)))] = 1.0
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


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
        self._fact_state: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._update = None  # (signature, C, Y, S) of the last Woodbury update
        # how many update columns (back-substitutions) one refactorization is worth; set
        # from the factorization's size, never from wall-clock time (that made the result
        # of a contact solve change from run to run)
        self._breakeven = 0.0
        self.factorizations = 0
        self.updates = 0
        # False once any solve stopped at MAX_ITERATIONS with the state still changing:
        # the result then belongs to an arbitrary intermediate contact/friction state
        self.converged = True

    # --- assembly --------------------------------------------------------------------

    def _rows(
        self, cs: ContactSet, mask: np.ndarray, coefs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduced DOF numbers and coefficients of the pairs selected by mask (fixed DOFs
        get a zero coefficient). `coefs` defaults to the gap gradients."""
        dofs = self.position[cs.dofs[mask]]
        c = (cs.coefs if coefs is None else coefs)[mask]
        c = np.where(dofs >= 0, c, 0.0)
        dofs = np.where(dofs >= 0, dofs, 0)
        return dofs, c

    def _spring_rows(self, cs: ContactSet) -> list[tuple[np.ndarray, np.ndarray]]:
        """(dofs, coefs) of every penalty spring of the current state: the normal spring of
        each closed pair and the two tangential springs of each sticking pair."""
        rows = [self._rows(cs, cs.active)]
        if cs.has_friction:
            rows += [self._rows(cs, cs.sticking, cs.tcoefs[:, k]) for k in range(2)]
        return rows

    def _penalty(self) -> tuple[sp.csr_matrix, np.ndarray]:
        rows, cols, vals = [], [], []
        f_c = np.zeros(len(self.free))
        for cs in self.contacts:
            for dofs, c in self._spring_rows(cs):
                rows.append(np.repeat(dofs, 21, axis=1).ravel())
                cols.append(np.tile(dofs, (1, 21)).ravel())
                vals.append(self.alpha * np.einsum("pi,pj->pij", c, c).ravel())
            dofs, c = self._rows(cs, cs.active)
            np.add.at(f_c, dofs.ravel(), (self.alpha * c * -cs.gap0[cs.active][:, None]).ravel())
            if cs.has_friction:
                for k in range(2):
                    # sticking: spring relaxed at the anchor, f = -alpha (s - anchor)
                    dofs, c = self._rows(cs, cs.sticking, cs.tcoefs[:, k])
                    offset = self.alpha * cs.anchor[cs.sticking, k]
                    np.add.at(f_c, dofs.ravel(), (c * offset[:, None]).ravel())
                    # slipping: Coulomb force on the slave, reaction on the master
                    dofs, c = self._rows(cs, cs.slipping, cs.tcoefs[:, k])
                    force = cs.slip_force[cs.slipping, k]
                    np.add.at(f_c, dofs.ravel(), (c * force[:, None]).ravel())
        K_pen = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=self.K_ff.shape,
        )
        return K_pen, f_c

    def _state(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(cs.active.copy(), cs.sticking.copy()) for cs in self.contacts]

    def _signature(self) -> tuple:
        return tuple(cs.active.tobytes() + cs.sticking.tobytes() for cs in self.contacts)

    def matrix(self) -> sp.csr_matrix:
        """Reduced matrix of the current active set."""
        return self.K_ff + self._penalty()[0]

    # --- linear solves ---------------------------------------------------------------

    def _change(self) -> tuple[sp.csr_matrix, np.ndarray] | None:
        """(C, D) with K_c = K_f + Cᵀ diag(D) C: one row per pair whose state differs from
        the factorized one, D = +alpha (engaged) or -alpha (released). None before the
        first factorization."""
        if self._fact_state is None:
            return None
        rows, cols, vals, signs = [], [], [], []
        n = len(self.free)
        k = 0
        for cs, (old_active, old_stick) in zip(self.contacts, self._fact_state, strict=True):
            springs = [(cs.active, old_active, None)]
            if cs.has_friction:
                springs += [(cs.sticking, old_stick, cs.tcoefs[:, j]) for j in range(2)]
            for now, old, coefs in springs:
                for mask, sign in ((now & ~old, 1.0), (~now & old, -1.0)):
                    if not mask.any():
                        continue
                    dofs, c = self._rows(cs, mask, coefs)
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
        change = self._change()
        k = change[0].shape[0] if change is not None else MAX_UPDATE_RANK + 1
        if k == 0:
            return self._fact_solve(rhs)
        # the update costs about k back-substitutions; refactorize when that is more than
        # a fresh factorization is worth (small systems factorize fast)
        log.debug("contact solve: %d changed pairs, break-even %.0f", k, self._breakeven)
        if k <= MAX_UPDATE_RANK and k < self._breakeven:
            x = self._woodbury(K_c, rhs, *change)
            if x is not None:
                return x
        self._fact_solve = self.factorize(K_c)
        self._breakeven = float(getattr(self._fact_solve, "breakeven", 0.0))
        self._fact_state = self._state()
        self._update = None
        self.factorizations += 1
        return self._fact_solve(rhs)

    def _solve_columns(self, B: np.ndarray) -> np.ndarray:
        try:
            Y = self._fact_solve(np.asfortranarray(B))
            if Y.shape != B.shape:
                raise ValueError
        except Exception:  # solver only takes vectors
            Y = np.column_stack([self._fact_solve(B[:, j]) for j in range(B.shape[1])])
        return np.asarray(Y)

    def _woodbury(self, K_c, rhs: np.ndarray, C: sp.csr_matrix, D: np.ndarray):
        """x = K_c⁻¹ rhs with K_c = K_f + Cᵀ diag(D) C and K_f factorized."""
        signature = self._signature()
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
        """Settle the active set (and the stick/slip state) under the full-size load f.
        Returns the displacement and the reduced matrix of the final contact state."""
        f_f = f[self.free]
        load_scale = max(np.abs(f).max(), 1e-30)
        u = np.zeros(self.n_dof)
        K_c = self.K_ff
        changed = True
        for cs in self.contacts:  # the open/closed state carries over, friction starts afresh
            cs.reset_friction()
        seen = {self._signature()}
        flips = [np.zeros(len(cs.active), dtype=int) for cs in self.contacts]
        tflips = [np.zeros(len(cs.active), dtype=int) for cs in self.contacts]
        exhausted = True
        for _ in range(MAX_ITERATIONS):
            K_pen, f_c = self._penalty()
            K_c = self.K_ff + K_pen
            u = np.zeros(self.n_dof)
            u[self.free] = self.reduced_solve(K_c, f_f + f_c)
            # Decide by the spring force, not the gap: a stiff penalty keeps even a
            # pulling pair's gap at round-off size. The force is taken from equilibrium
            # (K u - f on the slave node equals the contact force) rather than alpha * gap,
            # which would amplify solver round-off. Release a pair when it pulls with more
            # than a negligible share of the contact forces; engage an open pair when it
            # penetrates noticeably on the scale of the displacements. Pairs carrying
            # next to nothing would otherwise chatter between the two states for many
            # iterations, so each pair may change state only a few times per solve.
            residual = self.K @ u - f
            geom_tol = 1e-6 * max(np.abs(u).max(), 1e-30)
            changed = False  # a pair changed state (needs new springs)
            forces_changed = False  # only the Coulomb forces moved (same springs)
            for cs, count, tcount in zip(self.contacts, flips, tflips, strict=True):
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
                new_stick = cs.stick.copy()
                if cs.has_friction:
                    new_stick, forces_moved = self._coulomb(
                        cs,
                        new_active,
                        tcount >= MAX_FRICTION_FLIPS,
                        residual,
                        u,
                        -spring,
                        force_tol,
                    )
                    forces_changed |= forces_moved
                flipped = new_active != cs.active
                tflipped = new_stick != cs.stick
                count += flipped
                tcount += tflipped
                changed |= bool(np.any(flipped | tflipped))
                cs.active, cs.stick = new_active, new_stick
            if not changed and not forces_changed:
                exhausted = False
                break
            if changed:
                signature = self._signature()
                if signature in seen:  # chattering between two states: keep the current one
                    log.info("contact: active set cycles, stopping")
                    exhausted = False
                    break
                seen.add(signature)
        if exhausted:
            self.converged = False
            log.warning("contact: no settled state after %d iterations", MAX_ITERATIONS)
        log.info(
            "contact: %d iterations, %d factorizations, %d low-rank updates",
            MAX_ITERATIONS if changed else _ + 1,
            self.factorizations,
            self.updates,
        )
        return u, K_c

    def _coulomb(self, cs, new_active, frozen, residual, u, normal, force_tol):
        """Stick/slip decision of one contact set. `normal` is the compressive normal force
        of each pair (N); returns the new stick mask and whether the friction forces of
        slipping pairs changed by more than 0.1 %.

        A sticking pair slips when its spring force -alpha (s - anchor) exceeds mu N; the
        friction force then points along that spring force (against the sliding the pair
        was resisting) and keeps that direction: the direction a slipping pair's node moves
        afterwards is dominated by the elastic response of its neighbours to the force
        changes around it, so it would just flip from iteration to iteration. A slipping
        pair sticks again when it has moved back past its anchor along the friction force
        (the anchor is then reset to the current sliding), or when it opens. A pair that
        keeps alternating is frozen in the slipping state, whose force is bounded by mu N;
        frozen in the sticking state it could hold any force.
        """
        sliding = cs.sliding(u)
        limit = cs.friction * np.maximum(normal, 0.0)
        trial = -self.alpha * (sliding - cs.anchor)
        magnitude = np.linalg.norm(trial, axis=1)
        was_slipping = cs.active & ~cs.stick
        start = new_active & ~was_slipping & (magnitude > limit * (1 + 1e-6) + force_tol)
        # The sticking pairs shed load to each other one iteration at a time; when the
        # whole interface is pushed beyond its grip mu * sum(N) in one direction, it slides
        # as a whole: let every sticking pair slip at once instead of cascading.
        holding = new_active & ~was_slipping
        whole = None  # direction of the sliding when the interface slides as a whole
        if holding.any():
            slip_norm = np.linalg.norm(cs.slip_force, axis=1).sum()
            resultant = trial[holding].sum(axis=0) + cs.slip_force.sum(axis=0)
            total = magnitude[holding].sum() + slip_norm
            aligned = np.linalg.norm(resultant) > 0.9 * total
            if aligned and total > limit[new_active].sum() * (1 + 1e-3) + force_tol:
                start |= holding & (magnitude > force_tol)
                whole = resultant / np.linalg.norm(resultant)
        returned = np.einsum("pk,pk->p", sliding - cs.anchor, cs.slip_dir) > 0
        keep = was_slipping & new_active & (~returned | frozen)
        slip = start | keep
        first = start & ~np.any(cs.slip_dir != 0, axis=1)
        slip_dir = np.where(
            first[:, None], trial / np.maximum(magnitude, 1e-300)[:, None], cs.slip_dir
        )
        if whole is not None:
            # the local (Poisson-type) shear that set the first directions is dwarfed by
            # the sliding of the whole interface: every slipping pair resists that sliding
            slip_dir[slip] = whole
        new_force = np.where(slip[:, None], limit[:, None] * slip_dir, 0.0)
        anchor = cs.anchor.copy()
        reset = ~new_active | (was_slipping & ~slip)
        anchor[reset] = sliding[reset]  # re-engaging / sticking again is force-free
        delta = np.abs(new_force - cs.slip_force).max() if slip.any() else 0.0
        scale = max(np.abs(new_force).max() if slip.any() else 0.0, force_tol)
        cs.slip_force, cs.anchor, cs.slip_dir = new_force, anchor, slip_dir
        return ~slip, bool(delta > 1e-3 * scale)

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
