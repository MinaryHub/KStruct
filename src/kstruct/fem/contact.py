"""Contact between parts (node-to-surface, penalty, active set, Coulomb friction).

Each node of the contact faces_b is paired with the closest point on faces_a. With master
outward normal n, the gap is g = g0 + n·(u_slave - Σ N_i u_master_i). Active pairs are
held at g = 0 by a stiff spring; a pair is released when its spring would pull (tension)
and re-engaged when it penetrates.

With a friction coefficient μ > 0 every closed pair also carries two tangential springs
(TANGENT_PENALTY_FACTOR × the normal penalty) on the relative sliding
s = t·(u_slave − Σ N_i u_master_i), relaxed at an anchor. Coulomb's law is enforced by
return mapping on the anchors: the elastic trial force −α_t (s − base), measured from the
anchor committed at the start of the load increment, is cut back onto the friction cone
|F| ≤ μ N and the spring is re-anchored so that it carries exactly that force. The
matrix does not change while the anchors move, so each sweep is one back-substitution
(Aitken-accelerated), every pair keeps tangential support, and friction can only resist.
When nearly every pair is on the cone and their forces add up to the grip μ ΣN, the
interface slides as a whole: its springs leave the matrix and each pair carries μ N
against the resultant. Friction depends on the load path, so solve_path applies the
load in increments on top of the bolt tightening. Without friction parts slide freely
along the contact.
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
from kstruct.i18n import tr
from kstruct.model import Connection

MAX_ITERATIONS = 40
MAX_FLIPS = 3  # open/closed changes allowed per pair within one solve (anti-chatter)
# Friction depends on the load path: the external load is applied in this many equal
# increments on top of the bolt tightening, each starting from the friction state the
# previous one committed
FRICTION_INCREMENTS = 4
# Tangential springs are softer than the normal ones (and than ties): their stiffness sets
# how fast the anchor iteration settles (each sweep removes about the share k / (k +
# alpha_t) of the error, k = the parts' own stiffness against sliding). Elastic sliding
# within the grip stays ~1e-3 of the displacements.
TANGENT_PENALTY_FACTOR = 1e-3
FRICTION_TOL = 1e-3  # anchor movement (as force) relative to the friction forces: settled
# a gross slide: this share of the closed pairs on the friction cone, and the friction
# forces adding up to this share of the grip (they cannot exceed it)
GROSS_SLIP_SHARE = 0.9
GROSS_GRIP_SHARE = 0.98
MAX_FRICTION_SWEEPS = 400  # back-substitutions allowed for the anchor iteration of a solve
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
    stick: np.ndarray | None = None  # (P,) bool: within the grip (meaningful when closed)
    anchor: np.ndarray | None = None  # (P, 2) sliding at which the tangential springs relax
    base: np.ndarray | None = None  # (P, 2) anchor committed at the start of the increment
    gross: bool = False  # the interface slides as a whole: no springs, mu N along `slide`
    slide: np.ndarray | None = None  # (2,) unit direction of the friction force when gross
    slip_force: np.ndarray | None = None  # (P, 2) friction force on the slave when gross
    aitken: tuple | None = None  # (last anchor step, its relaxation, closed pairs) of a solve
    normal: np.ndarray | None = None  # (P,) compressive normal force of the last iteration (N)

    def __post_init__(self) -> None:
        n = len(self.active)
        if self.stick is None:
            self.stick = np.ones(n, dtype=bool)
        if self.slip_force is None:
            self.slip_force = np.zeros((n, 2))
        if self.anchor is None:
            self.anchor = np.zeros((n, 2))
        if self.base is None:
            self.base = np.zeros((n, 2))

    @property
    def has_friction(self) -> bool:
        return self.friction > 0 and self.tcoefs is not None

    @property
    def sprung(self) -> np.ndarray:
        """Pairs that carry tangential springs: every closed pair unless the interface
        slides as a whole."""
        if not self.has_friction or self.gross:
            return np.zeros(len(self.active), dtype=bool)
        return self.active.copy()

    @property
    def slipping(self) -> np.ndarray:
        if not self.has_friction:
            return np.zeros(len(self.active), dtype=bool)
        return self.active & (~self.stick | self.gross)

    def slip_share(self) -> float:
        """Share of the clamping (normal) force that sits on slipping pairs. Counting pairs
        would mislead: the corner nodes of quadratic triangles carry next to no normal
        force under a uniform pressure, so they "slip" anywhere while transmitting
        nothing."""
        if not self.has_friction or self.normal is None:
            return 0.0
        n = np.maximum(self.normal, 0.0)
        total = n[self.active].sum()
        return float(n[self.slipping].sum() / total) if total > 0 else 0.0

    def commit_friction(self, u: np.ndarray, alpha_t: float) -> None:
        """End of a load increment: the springs keep the friction forces they carry (for a
        gross slide the anchors move onto the friction cone) and become the reference
        the next increment measures its elastic trial forces from. A gross slide carries
        over: the load of the next increment grows in the same direction."""
        if not self.has_friction:
            return
        if self.gross:
            # the load grows in proportion: a slide as a whole goes on in the next
            # increment (finding it again from sticking springs takes hundreds of sweeps)
            self.anchor = self.sliding(u) + self.slip_force / alpha_t
        else:
            self.stick[:] = True
        self.base = self.anchor.copy()

    def reset_friction(self) -> None:
        """Every pair sticks again with relaxed springs (a new load is a new solution, not
        an increment of the last one)."""
        self.stick[:] = True
        self.slip_force[:] = 0.0
        self.anchor[:] = 0.0
        self.base[:] = 0.0
        self.gross = False

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
        raise ConstraintError(tr("접촉할 면이 메시에 없습니다."))
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
        raise ConstraintError(tr("접촉면이 서로 너무 멀리 떨어져 있습니다."))
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
        self.alpha_t = TANGENT_PENALTY_FACTOR * self.alpha
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
        self._offset = np.zeros(self.n_dof)  # imposed displacement of the current solve

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

    def _springs(self, cs: ContactSet, active, sprung) -> list:
        """(mask, coefs, stiffness) of every penalty spring family for a given state: the
        normal spring of each closed pair and the two tangential springs of each sprung
        pair."""
        out = [(active, None, self.alpha)]
        if cs.has_friction:
            out += [(sprung, cs.tcoefs[:, k], self.alpha_t) for k in range(2)]
        return out

    def _penalty(self) -> tuple[sp.csr_matrix, np.ndarray]:
        rows, cols, vals = [], [], []
        f_c = np.zeros(len(self.free))
        for cs in self.contacts:
            for mask, coefs, stiffness in self._springs(cs, cs.active, cs.sprung):
                dofs, c = self._rows(cs, mask, coefs)
                rows.append(np.repeat(dofs, 21, axis=1).ravel())
                cols.append(np.tile(dofs, (1, 21)).ravel())
                vals.append(stiffness * np.einsum("pi,pj->pij", c, c).ravel())
            dofs, c = self._rows(cs, cs.active)
            # a moved fixed node shifts the gap; its coefficient is not in the matrix
            gap0 = cs.gap(self._offset)[cs.active]
            np.add.at(f_c, dofs.ravel(), (self.alpha * c * -gap0[:, None]).ravel())
            if cs.has_friction:
                sprung, gross = cs.sprung, cs.active & cs.gross
                for k in range(2):
                    # springs relaxed at the anchor: f = -alpha_t (s - anchor)
                    dofs, c = self._rows(cs, sprung, cs.tcoefs[:, k])
                    moved = cs.sliding(self._offset)[sprung, k]
                    offset = self.alpha_t * (cs.anchor[sprung, k] - moved)
                    np.add.at(f_c, dofs.ravel(), (c * offset[:, None]).ravel())
                    # gross slide: Coulomb force on the slave, reaction on the master
                    dofs, c = self._rows(cs, gross, cs.tcoefs[:, k])
                    force = cs.slip_force[gross, k]
                    np.add.at(f_c, dofs.ravel(), (c * force[:, None]).ravel())
        K_pen = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=self.K_ff.shape,
        )
        return K_pen, f_c

    def _state(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(cs.active.copy(), cs.sprung) for cs in self.contacts]

    def _signature(self) -> tuple:
        return tuple(cs.active.tobytes() + cs.sprung.tobytes() for cs in self.contacts)

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
            springs = [
                (now, old, coefs, stiffness)
                for (now, coefs, stiffness), (old, _, _) in zip(
                    self._springs(cs, cs.active, cs.sprung),
                    self._springs(cs, old_active, old_stick),
                    strict=True,
                )
            ]
            for now, old, coefs, stiffness in springs:
                for mask, sign in ((now & ~old, stiffness), (~now & old, -stiffness)):
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
        return C, np.concatenate(signs)

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

        def apply(b: np.ndarray) -> np.ndarray:
            y = self._fact_solve(b)
            return y - Y @ (S_inv @ (C @ y))

        x = apply(rhs)
        if not np.all(np.isfinite(x)):
            return None
        # The update alone is only as accurate as cond(S)·eps allows: on a lap joint
        # cond(S) ~ 1e6 left 8e-7 of the peak displacement (a fresh factorization agrees
        # with another ordering to 6e-10), and on Windows CI it crossed 1e-6. One step of
        # iterative refinement with the same operator brings it to 1.5e-9. The residual
        # itself is no accuracy measure (the penalty rows amplify round-off), so the step
        # is kept only when it is a small correction; otherwise the plain update stands.
        dx = apply(rhs - K_c @ x)
        if np.all(np.isfinite(dx)) and np.abs(dx).max() <= 1e-3 * np.abs(x).max():
            x = x + dx
        self.updates += 1
        return x

    # --- active-set loop -------------------------------------------------------------

    @property
    def has_friction(self) -> bool:
        return any(cs.has_friction for cs in self.contacts)

    def solve_path(
        self, f_start: np.ndarray, f: np.ndarray, offset: np.ndarray | None = None
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        """Settle the contact under f_start (the bolt tightening, or nothing), then apply
        the rest of the load in FRICTION_INCREMENTS steps, each from the friction state the
        previous one committed. Without friction the answer does not depend on the path:
        a single solve under f."""
        if not self.has_friction:
            return self.solve(f, offset=offset)
        u = np.zeros(self.n_dof)
        if np.any(f_start):
            u = self.solve(f_start)[0]
        else:
            for cs in self.contacts:
                cs.reset_friction()
        for k in range(1, FRICTION_INCREMENTS + 1):
            for cs in self.contacts:
                cs.commit_friction(u, self.alpha_t)
            share = k / FRICTION_INCREMENTS  # imposed displacements grow with the load
            u, K_c = self.solve(
                f_start + (f - f_start) * share,
                True,
                None if offset is None else offset * share,
            )
        return u, K_c  # the last increment's stick/slip state is the reported one

    def solve(
        self, f: np.ndarray, keep_friction: bool = False, offset: np.ndarray | None = None
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        """Settle the active set (and the stick/slip state) under the full-size load f.
        Returns the displacement and the reduced matrix of the final contact state.
        keep_friction starts from the committed friction state (an increment of
        solve_path) instead of relaxing every tangential spring at zero sliding.
        offset: displacement imposed on the fixed DOFs (prescribed movements)."""
        self._offset = np.zeros(self.n_dof) if offset is None else offset
        f_f = (f - self.K @ self._offset)[self.free]
        load_scale = max(np.abs(f).max(), np.abs(f_f).max(), 1e-30)
        u = self._offset.copy()
        K_c = self.K_ff
        changed = True
        for cs in self.contacts:  # the open/closed state carries over, friction starts afresh
            cs.aitken = None
            if not keep_friction:
                cs.reset_friction()
        seen = {self._signature()}
        flips = [np.zeros(len(cs.active), dtype=int) for cs in self.contacts]
        exhausted = True
        changes = 0  # iterations that changed the springs (each needs a new solve matrix)
        sweeps = 0  # iterations that only moved anchors / friction forces (same matrix)
        while True:
            K_pen, f_c = self._penalty()
            K_c = self.K_ff + K_pen
            u = self._offset.copy()
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
            changed = False  # the springs changed (pairs opened / closed, gross slide)
            forces_changed = False  # only anchors or Coulomb forces moved (same matrix)
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
                if cs.has_friction:
                    switched, moved = self._coulomb(cs, new_active, u, -spring, force_tol)
                    changed |= switched
                    forces_changed |= moved
                cs.active = new_active
            log.debug(
                "contact iteration: %d open/closed changes so far, springs %s, forces %s",
                sum(int(x.sum()) for x in flips),
                "changed" if changed else "same",
                "moving" if forces_changed else "settled",
            )
            if not changed and not forces_changed:
                exhausted = False
                break
            if changed:
                changes += 1
                signature = self._signature()
                if signature in seen:  # chattering between two states: keep the current one
                    log.info("contact: active set cycles, stopping")
                    exhausted = False
                    break
                seen.add(signature)
                if changes >= MAX_ITERATIONS:
                    break
            else:
                sweeps += 1
                if sweeps >= MAX_FRICTION_SWEEPS:
                    break
        if exhausted:
            self.converged = False
            log.warning(
                "contact: no settled state after %d state changes and %d friction sweeps",
                changes,
                sweeps,
            )
        log.info(
            "contact: %d iterations (%d state changes), %d factorizations, %d low-rank updates",
            changes + sweeps + 1,
            changes,
            self.factorizations,
            self.updates,
        )
        return u, K_c

    def _coulomb(self, cs, new_active, u, normal, force_tol):
        """Friction update of one contact set after a solve; `normal` is the compressive
        normal force of each pair (N). Returns (springs changed, forces still moving).

        Partial slip (return mapping on fixed springs): the elastic trial force
        -alpha_t (s - base) is measured from the anchors committed at the start of the
        increment. Within mu N the pair sticks and its spring stays at the base;
        beyond, the force is cut back onto the friction cone, mu N along the trial force,
        and the spring is re-anchored to carry exactly that force at the current sliding.
        The springs never leave the matrix, so each sweep is one back-substitution, no
        pair can lose its tangential support (a fully slipping bolted plate used to spin
        about the bolt), and friction can only resist: a pair pushed back by its own
        friction force sticks again.

        Gross slide: when the tangential load on the whole interface exceeds mu sum(N),
        no anchor can hold it; the springs leave and every pair carries mu N against the
        resultant (the part is then held only by whatever else holds it, or slides away).
        """
        sliding = cs.sliding(u)
        cs.normal = normal
        limit = cs.friction * np.maximum(normal, 0.0)
        # pairs that are open (or have just closed) are force-free where they are
        fresh = ~new_active | ~cs.active
        cs.base[fresh] = sliding[fresh]
        cs.anchor[fresh] = sliding[fresh]
        trial = -self.alpha_t * (sliding - cs.base)
        magnitude = np.linalg.norm(trial, axis=1)
        grip = limit[new_active].sum()
        # round-off-sized tangential forces on nearly unloaded pairs are not slip
        over = new_active & (magnitude > limit * (1 + 1e-6) + force_tol)
        # A gross slide needs both: more tangential load than the grip, and no sticking
        # core left. Part of the load can be self-equilibrated (a pressed block's Poisson
        # expansion held back by a side support) and relaxes by partial slip instead.
        force = trial.copy()
        force[over] *= (limit[over] / magnitude[over])[:, None]
        if not cs.gross and over.sum() >= GROSS_SLIP_SHARE * new_active.sum() > 0:
            carried = force[new_active].sum(axis=0)
            demand = np.linalg.norm(carried)
            if demand >= GROSS_GRIP_SHARE * grip:
                cs.gross = True
                cs.slide = carried / demand
                log.info(
                    "contact: interface slides as a whole (%.4g N of grip %.4g N)", demand, grip
                )
                cs.slip_force = np.where(new_active[:, None], limit[:, None] * cs.slide, 0.0)
                cs.stick[:] = False
                return True, True
        if cs.gross:
            new_force = np.where(new_active[:, None], limit[:, None] * cs.slide, 0.0)
            change = np.abs(new_force - cs.slip_force).sum()
            cs.slip_force = new_force
            cs.stick[:] = False
            return False, bool(change > FRICTION_TOL * max(np.abs(new_force).sum(), force_tol))
        anchor = np.where(new_active[:, None], sliding + force / self.alpha_t, cs.anchor)
        movement = self.alpha_t * np.abs(anchor - cs.anchor)[new_active].sum()
        # Aitken (Irons-Tuck) acceleration of the anchor fixed point: each plain sweep only
        # removes a small share of the error when the springs are stiff against the parts
        step = (anchor - cs.anchor)[new_active].ravel()
        prev = cs.aitken
        omega = 1.0
        if prev is not None and prev[0].shape == step.shape and np.array_equal(prev[2], new_active):
            d = step - prev[0]
            dd = float(d @ d)
            if dd > 0:
                omega = float(np.clip(-prev[1] * (prev[0] @ d) / dd, 0.1, 20.0))
        cs.aitken = (step, omega, new_active.copy())
        if omega != 1.0:
            anchor[new_active] = cs.anchor[new_active] + omega * (anchor - cs.anchor)[new_active]
        total = np.abs(force[new_active]).sum()
        log.debug(
            "friction: closed %d, slipping %d, anchor movement %.3g N of %.3g N",
            int(new_active.sum()),
            int(over.sum()),
            movement,
            total,
        )
        cs.anchor = anchor
        cs.stick = ~over
        return False, bool(movement > FRICTION_TOL * max(total, force_tol))

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
