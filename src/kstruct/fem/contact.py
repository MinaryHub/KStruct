"""Frictionless contact between parts (node-to-surface, penalty, active set).

Each node of the contact faces_b is paired with the closest point on faces_a. With master
outward normal n, the gap is g = g0 + n·(u_slave - Σ N_i u_master_i). Active pairs are
held at g = 0 by a stiff spring; a pair is released when its spring would pull (tension)
and re-engaged when it penetrates. Parts may slide freely along the contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from kstruct.fem import tet10
from kstruct.fem.constraints import (
    PENALTY_FACTOR,
    ConstraintError,
    _project_to_triangles,
    _tri6_shape,
)
from kstruct.fem.mesh import FEMesh
from kstruct.model import Connection

MAX_ITERATIONS = 40


@dataclass
class ContactSet:
    dofs: np.ndarray  # (P, 21) slave xyz + 6 master nodes xyz
    coefs: np.ndarray  # (P, 21) gap gradient
    gap0: np.ndarray  # (P,) initial gap (mm), positive = open
    active: np.ndarray  # (P,) bool

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
    return ContactSet(np.array(dofs), np.array(coefs), gap0, gap0 <= 1e-6 * edge)


def solve_with_contact(
    K: sp.csr_matrix,
    f: np.ndarray,
    free: np.ndarray,
    contacts: list[ContactSet],
    solve,
    scale: float,
) -> tuple[np.ndarray, int]:
    """Active-set loop. `solve(K_free, f_free)` solves the reduced system.
    Returns the displacement and the number of iterations used."""
    n_dof = K.shape[0]
    alpha = PENALTY_FACTOR * scale
    # tiny springs everywhere keep parts that are only held by contact from being singular
    # while the active set settles; they carry a negligible share of the load
    stabilise = sp.identity(n_dof, format="csr") * (1e-9 * scale)
    u = np.zeros(n_dof)
    for iteration in range(1, MAX_ITERATIONS + 1):
        Kc, fc = K + stabilise, f.copy()
        for cs in contacts:
            Kp, fp = cs.penalty(n_dof, alpha)
            Kc = Kc + Kp
            fc += fp
        u = np.zeros(n_dof)
        u[free] = solve(Kc[free][:, free].tocsr(), fc[free])
        changed = False
        # decide by the spring force, not the gap: a stiff penalty keeps even a pulling
        # pair's gap at round-off size. Release a pair when it pulls with more than a
        # negligible share of the applied load; engage an open pair when it penetrates.
        force_tol = 1e-9 * max(np.abs(f).max(), 1e-30)
        for cs in contacts:
            g = cs.gap(u)
            geom_tol = 1e-9 * max(np.abs(cs.gap0).max(), 1.0)
            new_active = np.where(cs.active, alpha * g <= force_tol, g < -geom_tol)
            changed |= bool(np.any(new_active != cs.active))
            cs.active = new_active
        if not changed:
            return u, iteration
    return u, MAX_ITERATIONS
