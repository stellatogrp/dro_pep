"""Reduced TV-inpainting LP: only the unknown pixels are variables.

The known pixels p_K are substituted, and TV is summed only over edges that
touch an unknown pixel (edges between two known pixels are constants):

    min  sum_e t_e
    s.t. t_e - (D_U p_U)_e >=  (D_K p_K)_e
         t_e + (D_U p_U)_e >= -(D_K p_K)_e
         0 <= p_U <= u,  0 <= t <= u

In the standard form of ``tv_inpainting_test`` (min c^T x, A x = b, G x >= h,
l <= x <= u) with x = [p_U; t]:

    G = [[-D_U, I], [D_U, I]],  h = [D_K p_K; -D_K p_K],  no equality block.

The cross terms of G^T G cancel, so ||G||^2 = 2 max(||D_U||^2, 1): the operator
norm depends on how the unknown pixels cluster (isolated pixel -> ||D_U||^2 ~ 4,
large hole -> ~8), unlike the full LP whose norm is fixed by D.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from learning.tv_inpainting_test import (
    LP_UPPER,
    TVInpaintingMatrices,
    _horizontal_diff_matrix,
    _vertical_diff_matrix,
)


def diff_matrix(M: int, N: int) -> sp.csr_matrix:
    """Stacked [D_v; D_h], shape (K_v + K_h, M*N), row-major pixels."""
    return sp.vstack([_vertical_diff_matrix(M, N), _horizontal_diff_matrix(M, N)], format="csr")


def reduced_edges(known: np.ndarray, M: int, N: int):
    """(D_U, D_K) restricted to edges touching an unknown pixel.

    ``known`` is a flat boolean mask. Columns of D_U follow the ascending
    unknown-pixel order, columns of D_K the ascending known-pixel order.
    """
    D = diff_matrix(M, N).tocsc()
    unk, kn = np.flatnonzero(~known), np.flatnonzero(known)
    touch = np.flatnonzero(np.asarray(abs(D[:, unk]).sum(axis=1)).ravel() > 0)
    D = D[touch]
    return D[:, unk].tocsr(), D[:, kn].tocsr()


def build_reduced_lp(pix: np.ndarray, known: np.ndarray, M: int, N: int,
                     lp_upper: float = LP_UPPER) -> TVInpaintingMatrices:
    """Reduced LP of one channel. ``pix`` is the flat image, ``known`` the flat mask."""
    known = np.asarray(known, bool).reshape(-1)
    D_U, D_K = reduced_edges(known, M, N)
    E, U = D_U.shape
    dk = D_K @ np.asarray(pix, float).reshape(-1)[known]
    I = sp.eye(E, format="csr")
    G = sp.bmat([[-D_U, I], [D_U, I]], format="csr")
    n = U + E
    return TVInpaintingMatrices(
        c=np.concatenate([np.zeros(U), np.ones(E)]),
        A=sp.csr_matrix((0, n)), b=np.zeros(0),
        G=G, h=np.concatenate([dk, -dk]),
        l=np.zeros(n), u=lp_upper * np.ones(n),
    )


def known_known_tv(pix: np.ndarray, known: np.ndarray, M: int, N: int) -> float:
    """TV over edges whose two pixels are both known (the constant the reduced LP drops)."""
    known = np.asarray(known, bool).reshape(-1)
    D = diff_matrix(M, N)
    both = np.asarray(abs(D[:, np.flatnonzero(~known)]).sum(axis=1)).ravel() == 0
    return float(np.abs(D[both] @ np.asarray(pix, float).reshape(-1)).sum())


def solve_reduced(mats: TVInpaintingMatrices, tol: float = 1e-5) -> dict:
    """Clarabel solve of a reduced LP; raw_y = duals of G x >= h (y >= 0, PDHG convention)."""
    import cvxpy as cp

    x = cp.Variable(mats.c.size)
    cons = [mats.G @ x >= mats.h, x >= mats.l, x <= mats.u]
    prob = cp.Problem(cp.Minimize(mats.c @ x), cons)
    for opts in ({}, {"max_iter": 5000}):
        prob.solve(solver=cp.CLARABEL, verbose=False, **opts)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            continue
        raw_x = np.asarray(x.value)
        viol = np.maximum(mats.h - mats.G @ raw_x, 0.0)
        box = np.maximum(mats.l - raw_x, 0) + np.maximum(raw_x - mats.u, 0)
        res = float(np.sqrt(viol @ viol + box @ box))
        if res <= tol:
            return {"objective_value": float(prob.value), "raw_x": raw_x,
                    "raw_y": np.asarray(cons[0].dual_value, float)}
    raise RuntimeError(f"reduced LP solve failed: status={prob.status}")


def reduced_opnorm(mats: TVInpaintingMatrices) -> float:
    return float(sla.svds(mats.G.tocsc(), k=1, return_singular_vectors=False)[0])


def stack_channels(mats_list):
    """Block-diagonal color LP from per-channel reduced LPs (shared mask)."""
    return TVInpaintingMatrices(
        c=np.concatenate([m.c for m in mats_list]),
        A=sp.csr_matrix((0, sum(m.c.size for m in mats_list))), b=np.zeros(0),
        G=sp.block_diag([m.G for m in mats_list], format="csr"),
        h=np.concatenate([m.h for m in mats_list]),
        l=np.concatenate([m.l for m in mats_list]),
        u=np.concatenate([m.u for m in mats_list]),
    )
