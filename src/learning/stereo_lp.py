"""Linearized TV-L1 stereo-disparity LP on a rectified image pair.

For left/right views I_L, I_R (C channels, M x N) and a disparity map d
(pixels, one per pixel, shared by the channels), the linearized data residual
of channel c is  rho_c(d) = I_R,c - I_L,c - g_c * d  with g_c the horizontal
gradient of I_L,c. The LP is

    min  sum_c sum_i a s_{c,i} + sum_e t_e
    s.t. s_{c,i} >= +-(w (g_c d - r_c))_i,   t_e >= +-(D d)_e,   box bounds,

with r_c = I_R,c - I_L,c. The data weight lambda enters either the cost
('cost': a = lambda, w = 1) or the constraints ('inside': a = 1, w = lambda).
In the standard form of ``tv_inpainting_test`` (min c^T x, G x >= h,
l <= x <= u, no equality block), with x = [d; s_1..s_C; t]:

    G = [[-W g_1, I, .., 0], [W g_1, I, .., 0], ..., [-D, 0, I], [D, 0, I]].

G's d-block Gram is 2 (w^2 diag(sum_c g_c^2) + D^T D), so with lambda inside
||G|| follows the image's strongest horizontal edges and the channel count.
"""
import numpy as np
import scipy.sparse as sp

from learning.tv_inpainting_reduced import diff_matrix
from learning.tv_inpainting_test import TVInpaintingMatrices

DMAX = 1.0  # max |true disparity| (pixels); the linearization is for small shifts


def true_disparity(M: int, N: int, rng: np.random.Generator, dmax: float = DMAX) -> np.ndarray:
    """Synthetic depth: a smooth Gaussian bump plus a vertical layer step, max |d| = dmax."""
    yy, xx = np.mgrid[0:M, 0:N] / M
    cy, cx = rng.uniform(0.3, 0.7, 2)
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(0.08, 0.2) ** 2))
    layer = (xx > rng.uniform(0.2, 0.8)).astype(float) * rng.uniform(-0.5, 0.5)
    d = 0.6 * bump + layer
    return dmax * d / max(np.abs(d).max(), 1e-9)


def warp(img: np.ndarray, d: np.ndarray) -> np.ndarray:
    """out(y, x) = img(y, x + d(y, x)), horizontal linear interpolation, edge-clamped."""
    xs = np.arange(img.shape[1])
    return np.stack([np.interp(xs + d[r], xs, img[r]) for r in range(img.shape[0])])


def grad_x(img: np.ndarray) -> np.ndarray:
    """Horizontal central difference (one-sided at the borders)."""
    g = np.zeros_like(img)
    g[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    g[:, 0] = img[:, 1] - img[:, 0]
    g[:, -1] = img[:, -1] - img[:, -2]
    return g


def make_pair(left: np.ndarray, d: np.ndarray):
    """(left channels, right channels) for an (M, N, C) image in [0, 1] and disparity d."""
    L = [left[:, :, c] for c in range(left.shape[2])]
    return L, [warp(c, d) for c in L]


def stereo_x0(l: np.ndarray, u: np.ndarray, K_pix: int, slack_init=None,
              feasible_h: np.ndarray | None = None, n_channels: int = 1,
              delta: float = 0.01) -> np.ndarray:
    """PDHG primal init with d = 0 (the box center); strictly interior in every case.

      * feasible_h given: the feasible start at d = 0 -- data slacks
        s_c = |w r_c| + delta (the channel's first K rows of h are -w r_c) and
        TV slacks t = delta, so c^T x0 - f* measures how far d = 0 is from optimal;
      * slack_init given: every slack at slack_init;
      * otherwise: the box center (arbitrary large slacks; R ~ 1e3).
    Layout: x = [d (K); s_1..s_C (K each); t], rows of h = [ch1 -+; ...; chC -+; TV -+].
    """
    x0 = np.asarray(0.5 * (l + u), dtype=float).copy()
    if feasible_h is not None:
        x0[K_pix:] = delta
        for c in range(n_channels):
            x0[(c + 1) * K_pix:(c + 2) * K_pix] = np.abs(feasible_h[2 * c * K_pix:(2 * c + 1) * K_pix]) + delta
    elif slack_init is not None:
        x0[K_pix:] = float(slack_init)
    return x0


def stereo_y0(h: np.ndarray, K_pix: int, n_channels: int = 1, a: float = 1.0,
              delta: float = 0.01) -> np.ndarray:
    """Warm dual start consistent with the primal start d = 0.

    Each constraint pair  s >= +-e  (rows [top; bottom]) gets the subgradient of
    |e| at d = 0, scaled to the slack's cost and kept strictly positive: the
    active row (top when h_top > 0, i.e. e > 0) gets (1 - delta) of the cost,
    the other delta; e = 0 gives an even split. Data pairs have cost ``a``,
    TV pairs cost 1 and residual 0 at d = 0 (even split).
    Layout of h: [ch1 top (K); ch1 bottom (K); ...; TV top (E); TV bottom (E)].
    """
    y0 = np.empty_like(h, dtype=float)
    for c in range(n_channels):
        top = h[2 * c * K_pix:(2 * c + 1) * K_pix]
        w_top = np.where(top > 0, 1 - delta, np.where(top < 0, delta, 0.5)) * a
        y0[2 * c * K_pix:(2 * c + 1) * K_pix] = w_top
        y0[(2 * c + 1) * K_pix:(2 * c + 2) * K_pix] = a - w_top
    y0[2 * n_channels * K_pix:] = 0.5
    return y0


def stereo_lp(chans_L, chans_R, lam: float, place: str = "inside",
              dmax: float = DMAX) -> TVInpaintingMatrices:
    """Stereo LP of one pair; x = [d (K); s (C*K); t (E)]."""
    if place not in ("inside", "cost"):
        raise ValueError(f"place must be 'inside' or 'cost', got {place!r}")
    M, N = chans_L[0].shape
    D = diff_matrix(M, N)
    K, E, C = M * N, D.shape[0], len(chans_L)
    w, a = (lam, 1.0) if place == "inside" else (1.0, lam)
    rows, h = [], []
    Z = sp.csr_matrix((K, E))
    for c in range(C):
        g = grad_x(chans_L[c]).ravel()
        r = (chans_R[c] - chans_L[c]).ravel()
        Wg = sp.diags(w * g)
        S = sp.csr_matrix((np.ones(K), (np.arange(K), c * K + np.arange(K))), shape=(K, C * K))
        rows += [sp.hstack([-Wg, S, Z]), sp.hstack([Wg, S, Z])]
        h += [-w * r, w * r]
    I = sp.eye(E)
    Zs = sp.csr_matrix((E, C * K))
    rows += [sp.hstack([-D, Zs, I]), sp.hstack([D, Zs, I])]
    h += [np.zeros(E), np.zeros(E)]
    G = sp.vstack(rows, format="csr")
    # |w (g d - r)| <= w (2 dmax max|g| + max|r|) on the d-box [-2 dmax, 2 dmax]
    s_up = w * (2 * 2 * dmax * max(np.abs(grad_x(cl)).max() for cl in chans_L)
                + max(np.abs(cr - cl).max() for cl, cr in zip(chans_L, chans_R))) + 0.1
    n = K + C * K + E
    return TVInpaintingMatrices(
        c=np.r_[np.zeros(K), a * np.ones(C * K), np.ones(E)],
        A=sp.csr_matrix((0, n)), b=np.zeros(0), G=G, h=np.concatenate(h),
        l=np.r_[-2 * dmax * np.ones(K), np.zeros(C * K + E)],
        u=np.r_[2 * dmax * np.ones(K), s_up * np.ones(C * K), 4 * dmax * np.ones(E)],
    )
