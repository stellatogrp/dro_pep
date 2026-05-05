"""ALISTA W matrix computation and persistence.

Implements the offline convex problem from Liu, Chen, Wang, Yin
(ICLR 2019, https://openreview.net/pdf?id=B1lnzn0ctQ), eq. 16:

    W~ = argmin_{W in R^{m x n}}  ||W^T A||_F^2
         subject to  (W^T A)_{i,i} = 1   for i = 1, ..., n

Frobenius-norm minimization with linear equality constraints; decouples
per column of W. We solve n small QPs with CVXPY (CLARABEL), matching the
solver choice used elsewhere for Lasso.
"""

import logging
import os
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from tqdm import trange

log = logging.getLogger(__name__)


def compute_alista_W(A_np: np.ndarray) -> np.ndarray:
    """Solve the ALISTA W problem (paper eq. 16) by per-column decomposition.

    For each column index i in [0, n):
        w_i = argmin_w  ||A^T w||_2^2   s.t.  w^T A[:, i] = 1
    Stack as W[:, i] = w_i, so W has shape (m, n) and (W^T A)_{i,i} = 1.

    Args:
        A_np: (m, n) measurement matrix.

    Returns:
        W: (m, n) numpy array such that diag(W^T A) ~= 1 and off-diagonals
        of W^T A are minimized in Frobenius norm.
    """
    m, n = A_np.shape
    W = np.zeros((m, n))

    log.info(f"Computing ALISTA W via per-column CVXPY (m={m}, n={n}, {n} QPs)...")
    for i in trange(n, desc="ALISTA W columns"):
        w = cp.Variable(m)
        residual = A_np.T @ w  # (n,)
        obj = cp.Minimize(cp.sum_squares(residual))
        constraints = [residual[i] == 1.0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            raise RuntimeError(
                f"CVXPY failed to compute ALISTA W column {i} (status={prob.status})"
            )
        W[:, i] = w.value

    M = W.T @ A_np
    diag_err = float(np.max(np.abs(np.diag(M) - 1.0)))
    off_diag = M - np.diag(np.diag(M))
    fro = float(np.linalg.norm(M, ord='fro'))
    max_off = float(np.max(np.abs(off_diag)))
    log.info(
        f"ALISTA W done: ||diag(W^T A) - 1||_inf = {diag_err:.3e}, "
        f"||W^T A||_F = {fro:.3e}, max |off-diag| = {max_off:.3e}"
    )
    return W


def load_or_compute_W(
    data_source_dir: Optional[str], A_np: np.ndarray
) -> Tuple[np.ndarray, str]:
    """Load W_alista.npz from data_source_dir if present (and shape-compatible),
    else compute it from A_np.

    Args:
        data_source_dir: Directory containing W_alista.npz, or None.
        A_np: (m, n) measurement matrix used to validate the loaded W and
            (if needed) to compute W from scratch.

    Returns:
        (W_np, source) where source is 'loaded' or 'computed'.
    """
    if data_source_dir is not None:
        W_path = os.path.join(data_source_dir, 'W_alista.npz')
        if os.path.isfile(W_path):
            d = np.load(W_path)
            W_np = d['W']
            if W_np.shape == A_np.shape:
                log.info(f"Loaded ALISTA W from {W_path}, shape: {W_np.shape}")
                return W_np, 'loaded'
            log.warning(
                f"{W_path} has shape {W_np.shape}, expected {A_np.shape}; "
                f"recomputing W from A."
            )

    W_np = compute_alista_W(A_np)
    return W_np, 'computed'
