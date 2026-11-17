"""Quadratic tetrahedron (TET10) and quadratic triangle (TRI6) element math, vectorized
over all elements with NumPy.

Natural coordinates (r, s, t); barycentric L0 = 1-r-s-t, L1 = r, L2 = s, L3 = t.
Voigt order for stress/strain: xx, yy, zz, xy, yz, zx (engineering shear strains).
"""

from __future__ import annotations

import numpy as np

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# 4-point Gauss rule, exact for quadratic integrands; weights sum to tet volume 1/6
_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS_POINTS = np.array([[_B, _B, _B], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A]])
GAUSS_WEIGHTS = np.full(4, 1.0 / 24.0)

# natural coordinates of the 10 nodes (for stress recovery at nodes)
NODE_POINTS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    + [
        [0.5 * (a + b) for a, b in zip(p, q, strict=True)]
        for p, q in (
            ([0, 0, 0], [1, 0, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ([0, 0, 0], [0, 0, 1]),
            ([1, 0, 0], [0, 1, 0]),
            ([1, 0, 0], [0, 0, 1]),
            ([0, 1, 0], [0, 0, 1]),
        )
    ],
    dtype=float,
)

# d L_i / d(r, s, t)
_DL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def _barycentric(p: np.ndarray) -> np.ndarray:
    r, s, t = p
    return np.array([1 - r - s - t, r, s, t])


def shape_functions(p: np.ndarray) -> np.ndarray:
    L = _barycentric(p)
    corner = L * (2 * L - 1)
    edge = [4 * L[i] * L[j] for i, j in EDGES]
    return np.concatenate([corner, edge])


def shape_derivatives(p: np.ndarray) -> np.ndarray:
    """dN/d(r,s,t), shape (10, 3)."""
    L = _barycentric(p)
    dN = np.zeros((10, 3))
    for i in range(4):
        dN[i] = (4 * L[i] - 1) * _DL[i]
    for k, (i, j) in enumerate(EDGES):
        dN[4 + k] = 4 * (L[i] * _DL[j] + L[j] * _DL[i])
    return dN


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def gradients(X: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical shape-function gradients at natural point p.

    X: (E, 10, 3) element node coordinates.
    Returns dN/dx (E, 10, 3) and |det J| (E,).
    """
    dN = shape_derivatives(p)
    J = np.einsum("eni,nj->eij", X, dN)  # J[e, i, j] = d x_i / d r_j
    det = np.linalg.det(J)
    if np.any(np.abs(det) < 1e-14 * np.abs(det).max()):
        raise ValueError("degenerate element in mesh")
    invJ = np.linalg.inv(J)
    dNdx = np.einsum("nj,eji->eni", dN, invJ)
    return dNdx, np.abs(det)


def strain_matrix(dNdx: np.ndarray) -> np.ndarray:
    """B matrices (E, 6, 30) mapping element displacements [u0x u0y u0z u1x ...] to strain."""
    n_el = dNdx.shape[0]
    B = np.zeros((n_el, 6, 30))
    dx, dy, dz = dNdx[..., 0], dNdx[..., 1], dNdx[..., 2]
    B[:, 0, 0::3] = dx
    B[:, 1, 1::3] = dy
    B[:, 2, 2::3] = dz
    B[:, 3, 0::3] = dy
    B[:, 3, 1::3] = dx
    B[:, 4, 1::3] = dz
    B[:, 4, 2::3] = dy
    B[:, 5, 0::3] = dz
    B[:, 5, 2::3] = dx
    return B


def stiffness(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Element stiffness matrices (E, 30, 30) for per-element D (E, 6, 6)."""
    K = np.zeros((X.shape[0], 30, 30))
    for p, w in zip(GAUSS_POINTS, GAUSS_WEIGHTS, strict=True):
        dNdx, det = gradients(X, p)
        B = strain_matrix(dNdx)
        # batched matmul (BLAS) is several times faster than the equivalent einsum
        K += (B.transpose(0, 2, 1) @ (D @ B)) * (det * w)[:, None, None]
    return K


def mass_lumped_volume(X: np.ndarray) -> np.ndarray:
    """Consistent nodal volume weights (E, 10): integral of N_i over each element."""
    out = np.zeros((X.shape[0], 10))
    for p, w in zip(GAUSS_POINTS, GAUSS_WEIGHTS, strict=True):
        _, det = gradients(X, p)
        out += np.outer(det * w, shape_functions(p))
    return out


def consistent_mass(X: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Consistent mass matrices (E, 30, 30) for per-element density rho (E,).

    N_i N_j is degree 4, so this uses the 11-point Keast rule.
    """
    M = np.zeros((X.shape[0], 30, 30))
    eye = np.eye(3)
    for p, w in zip(_KEAST_POINTS, _KEAST_WEIGHTS, strict=True):
        _, det = gradients(X, p)
        N = shape_functions(p)
        NN = np.kron(np.outer(N, N), eye)
        M += np.einsum("e,ij->eij", rho * det * w, NN)
    return M


# Keast 11-point rule (degree 4) for the reference tetrahedron, weights sum to 1/6
_K1, _K2, _K3 = 0.0714285714285714, 0.785714285714286, 0.399403576166799
_K4 = 0.100596423833201
_KEAST_POINTS = np.array(
    [
        [0.25, 0.25, 0.25],
        [_K1, _K1, _K1],
        [_K2, _K1, _K1],
        [_K1, _K2, _K1],
        [_K1, _K1, _K2],
        [_K3, _K3, _K4],
        [_K3, _K4, _K3],
        [_K4, _K3, _K3],
        [_K4, _K4, _K3],
        [_K4, _K3, _K4],
        [_K3, _K4, _K4],
    ]
)
_KEAST_WEIGHTS = np.array(
    [-0.0131555555555556] + [0.00762222222222222] * 4 + [0.0248888888888889] * 6
)


def tri6_load_weights(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For boundary TRI6 faces (S, 6, 3): nodal weights of a uniform traction, the
    area-weighted unit normal (pointing out of the solid), and the area.

    For a flat TRI6, a uniform traction q gives nodal forces 0 at corners and q*A/3 at
    midside nodes.
    """
    v1 = X[:, 1] - X[:, 0]
    v2 = X[:, 2] - X[:, 0]
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / (2 * area[:, None])
    weights = np.zeros((len(X), 6))
    weights[:, 3:] = area[:, None] / 3.0
    return weights, normal, area
