"""PDLP problem module: TV inpainting on Olivetti faces.

Replaces the previous facility-location formulation with anisotropic L1 total
variation (TV) inpainting — see ``src/learning/tv_inpainting_test.py`` for the
underlying LP construction and ``fc_pdlp.py`` for the prior facility-location
implementation kept for reference.

The TV inpainting LP (from ``tv_inpainting_test.py``) is:

    min  c^T x   s.t.   l <= x <= u,  A x = b,  G x >= h

with x = [p; v; w], c = [0_K; 1_{K_v}; 1_{K_h}], h = 0. Translating to the
fc_pdlp / Chambolle–Pock convention used by the verified trajectory and PEP:

    K_mat = [-A_ineq; A_eq] = [G; A]      (G is mask-independent;
                                           A varies row-by-row with the mask)
    q     = [-b_ineq; b_eq] = [0; b]      (b = pixel values at known indices)

Sample-creation caches ``(image_index, mask, x_opt, y_opt, f_opt, M_val_batch,
R_val_batch)`` per instance; the training-time module reconstructs (c, K_mat,
q) on the fly from ``(image_index, mask)`` and reuses the cached optima.
``M_val`` and ``R_val`` come straight from ``out_of_sample_metadata.npz``.
"""

import diffcp_patch  # noqa: F401  # COO -> CSC fix for diffcp (used by DRO SDP)
import logging
import os
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.datasets import fetch_olivetti_faces

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from learning.problem_module import (
    GroundTruth,
    ParameterNames,
    ProblemData,
    ProblemModule,
    Stepsizes,
)
from learning.pep_constructions import construct_chambolle_pock_pep_data
from learning.trajectories import problem_data_to_cp_lp_trajectories
from learning.unified_trainer import UnifiedTrainer
from learning.tv_inpainting_test import (
    _horizontal_diff_matrix,
    _vertical_diff_matrix,
    extract_constraint_matrices,
    solve_lp,
)

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# Module-level helpers
# =============================================================================

def sample_corrupted_indices(
    M: int, N: int, missing_fraction: float, seed: int
) -> np.ndarray:
    """Deterministic-count corruption.

    Picks exactly ``round(missing_fraction * M * N)`` pixel indices uniformly
    without replacement. Fixed count keeps S = K - n_corrupted constant across
    instances so x_opt / y_opt arrays stack cleanly.
    """
    rng = np.random.default_rng(seed)
    K = M * N
    n_corrupted = int(round(missing_fraction * K))
    return np.sort(rng.permutation(K)[:n_corrupted])


def split_persons_by_subject(
    person_split_seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_total: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute subject indices and split. Output arrays are sorted ascending."""
    if n_train + n_val + n_test != n_total:
        raise ValueError(
            f"split sizes {n_train}+{n_val}+{n_test} != n_total {n_total}"
        )
    rng = np.random.default_rng(person_split_seed)
    perm = rng.permutation(n_total)
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train : n_train + n_val])
    test = np.sort(perm[n_train + n_val :])
    return train, val, test


def image_pool_for_persons(persons: np.ndarray, images_per_subject: int) -> np.ndarray:
    """Each subject p contributes images [p*S, p*S+1, ..., p*S + S-1] for S images_per_subject."""
    return np.concatenate(
        [np.arange(p * images_per_subject, (p + 1) * images_per_subject) for p in sorted(persons)]
    )


def build_strict_interior_init(
    n_vars: int, m1: int, S: int, lp_upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strict-interior PDHG init (numpy form, used by sample creation).

    Returns ``(x0, y0)`` matching the trainer's ``_make_cp_tv_traj_fn``
    exactly:
        x0 = 0.5 * lp_upper * ones(n_vars)        (interior of [0, lp_upper])
        y0 = [0.1 * ones(m1) ; zeros(S)]          (interior of nonneg cone × R^S)

    The PEP basis encodes the initial subgradients as ``gf1_0 = c`` and
    ``gh_0 = -q``. These are *only* valid subgradients of
        f1(v) = c^T v + ind_{[l,u]}(v),
        h(y)  = -q^T y + ind_{R^{m1}_+ × R^{m2}}(y)
    when (x0, y0) is in the *strict* interior of the box and the nonneg cone.
    Boundary points (e.g., x0 = 0) make the implied subgradient ill-posed and
    can break the PEP interpolation inequalities downstream.
    """
    x0 = 0.5 * lp_upper * np.ones(n_vars, dtype=np.float64)
    y0 = np.concatenate([0.1 * np.ones(m1), np.zeros(S)])
    return x0, y0


def _warm_start_primal_single(
    pix: np.ndarray, mask: np.ndarray, M: int, N: int, delta: float, lp_upper: float,
) -> np.ndarray:
    """Warm-start primal block ``[p0; v0; w0]`` for one grayscale channel.

    ``p0`` is the corrupted image (missing pixels = 0) clipped into
    ``[delta, 1 - delta] * lp_upper`` and the slacks are the TV differences of
    ``p0`` plus ``delta * lp_upper``. Since ``|p_a - p_b| <= (1 - 2 delta) *
    lp_upper``, every coordinate lies strictly inside ``(0, lp_upper)`` and
    every ``G x0 >= 0`` row holds strictly, so the interior argument of
    ``build_strict_interior_init`` still applies.
    """
    lo, hi = delta * lp_upper, (1.0 - delta) * lp_upper
    P = np.clip(np.where(mask, pix, 0.0), lo, hi).reshape(M, N)
    v0 = np.abs(P[1:, :] - P[:-1, :]).reshape(-1) + lo   # matches _vertical_diff_matrix
    w0 = np.abs(P[:, 1:] - P[:, :-1]).reshape(-1) + lo   # matches _horizontal_diff_matrix
    return np.concatenate([P.reshape(-1), v0, w0])


def _consistent_init_single(xp, pix, mask, M: int, N: int, delta: float, lp_upper: float, S: int,
                            hole_value: float | None = None):
    """Consistent primal-dual start for one grayscale channel (numpy or JAX via ``xp``).

    Primal: known pixels at their observed values, each missing pixel at the mean
    of its known 4-neighbours (the known-pixel mean if it has none) -- or, when
    ``hole_value`` is given, at hole_value * lp_upper (the corrupted image) -- clipped into
    [delta, 1 - delta] * lp_upper; TV slacks |D p0| + delta * lp_upper (strictly
    feasible). Dual: on each TV pair (t >= D p, t >= -D p) the subgradient of |D p0|
    (1 - delta on the active row, delta on the other, even split at 0; each pair
    sums to its cost 1), and on the known pixels the equality duals that make the
    known pixels stationary, y_A = (D_v^T (2 t_v - 1) + D_h^T (2 t_h - 1))[known].
    Layout matches extract_constraint_matrices: x = [p; v; w],
    y = [top_v; bot_v; top_h; bot_h; y_A (known pixels ascending)].
    """
    lo, hi = delta * lp_upper, (1.0 - delta) * lp_upper
    known = mask.reshape(M, N)
    img = xp.where(known, pix.reshape(M, N), 0.0)
    kf = known.astype(img.dtype)
    P, Kp = xp.pad(img, 1), xp.pad(kf, 1)
    nsum = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]
    ncnt = Kp[:-2, 1:-1] + Kp[2:, 1:-1] + Kp[1:-1, :-2] + Kp[1:-1, 2:]
    fill = xp.where(ncnt > 0, nsum / xp.maximum(ncnt, 1.0), img.sum() / kf.sum())
    if hole_value is not None:
        fill = xp.full_like(img, hole_value * lp_upper)
    p = xp.clip(xp.where(known, img, fill), lo, hi)
    dv, dh = p[1:, :] - p[:-1, :], p[:, 1:] - p[:, :-1]
    x0 = xp.concatenate([p.reshape(-1), (xp.abs(dv) + lo).reshape(-1), (xp.abs(dh) + lo).reshape(-1)])
    tv = xp.where(dv > 0, 1.0 - delta, xp.where(dv < 0, delta, 0.5))
    th = xp.where(dh > 0, 1.0 - delta, xp.where(dh < 0, delta, 0.5))
    av, ah = 2.0 * tv - 1.0, 2.0 * th - 1.0
    zero = xp.zeros((1, N)), xp.zeros((M, 1))
    force = (xp.concatenate([zero[0], av], 0) - xp.concatenate([av, zero[0]], 0)
             + xp.concatenate([zero[1], ah], 1) - xp.concatenate([ah, zero[1]], 1))   # D_v^T av + D_h^T ah
    if xp is np:
        idx = np.flatnonzero(mask.reshape(-1))
    else:
        idx = jnp.where(mask.reshape(-1), size=S)[0]
    yG = xp.concatenate([tv.reshape(-1), (1.0 - tv).reshape(-1), th.reshape(-1), (1.0 - th).reshape(-1)])
    return x0, yG, force.reshape(-1)[idx]


def build_pdhg_init(
    init_type: str,
    n_vars: int,
    m1: int,
    S: int,
    lp_upper: float,
    pix_channels=None,
    mask: np.ndarray | None = None,
    M: int | None = None,
    N: int | None = None,
    delta: float = 0.01,
    dual_ineq_init: float = 0.1,
    dual_eq_init: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PDHG init for ``init_type`` in {'center', 'warm'} (numpy form).

    'center' is ``build_strict_interior_init``; 'warm' starts the primal at the
    corrupted image of each channel in ``pix_channels`` (one flat (M*N,) array
    per channel, laid out as in ``build_color_lp``). ``dual_ineq_init`` and
    ``dual_eq_init`` set the inequality- and equality-dual entries of ``y0``
    (0.1 / 0 reproduce the original init; the paper states y0 = 1 for both).
    """
    x0, y0 = build_strict_interior_init(n_vars, m1, S, lp_upper)
    y0[:m1] = dual_ineq_init
    y0[m1:] = dual_eq_init
    if init_type == 'center':
        return x0, y0
    if init_type in ('consistent', 'consistent_holes'):
        # Both ignore dual_ineq_init / dual_eq_init (the duals are constructed).
        # 'consistent_holes' starts the missing pixels at 0.5 (the corrupted image)
        # instead of the neighbour fill.
        S_c = int(np.count_nonzero(mask))
        hv = 0.5 if init_type == 'consistent_holes' else None
        parts = [_consistent_init_single(np, np.asarray(pch, float), np.asarray(mask, bool), M, N,
                                         delta, lp_upper, S_c, hole_value=hv) for pch in pix_channels]
        x0 = np.concatenate([q[0] for q in parts])
        y0 = np.concatenate([q[1] for q in parts] + [q[2] for q in parts])
        if x0.size != n_vars or y0.size != m1 + S:
            raise ValueError(f"consistent init sizes x0 {x0.size} / y0 {y0.size} != {n_vars} / {m1 + S}")
        return x0, y0
    if init_type != 'warm':
        raise ValueError(f"Unknown init_type {init_type!r}; expected 'center', 'warm', 'consistent' "
                         f"or 'consistent_holes'.")
    x0 = np.concatenate([
        _warm_start_primal_single(p, mask, M, N, delta, lp_upper) for p in pix_channels
    ])
    if x0.size != n_vars:
        raise ValueError(f"warm-start x0 has size {x0.size}, expected n_vars={n_vars}")
    return x0, y0


def gaussian_blur_matrix(M: int, N: int, sigma: float) -> sp.csr_matrix:
    """(M*N, M*N) separable Gaussian blur on a row-major image; identity if ``sigma == 0``.

    The 1-D kernel is truncated at ``ceil(3 sigma)`` and each row renormalized
    to sum to 1 (so borders are not darkened). Rows sum to 1, and the border
    renormalization makes ``||H||_2`` exceed 1 only slightly (<= ~1.004).
    """
    if sigma == 0:
        return sp.eye(M * N, format="csr")

    def _blur_1d(n):
        r = int(np.ceil(3 * sigma))
        offs = np.arange(-r, r + 1)
        k = np.exp(-offs ** 2 / (2 * sigma ** 2))
        B = sp.diags([k[i] * np.ones(n - abs(o)) for i, o in enumerate(offs) if abs(o) < n],
                     [o for o in offs if abs(o) < n], shape=(n, n)).tocsr()
        return (sp.diags(1.0 / np.asarray(B.sum(axis=1)).ravel()) @ B).tocsr()

    return sp.kron(_blur_1d(M), _blur_1d(N), format="csr")


def blurred_data_block(
    known_indices: np.ndarray, pix_channels, H_list, n_block_vars: int,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Equality data ``(A, b)`` for observing the blurred image at the known pixels.

    Channel c contributes ``A_c = [E H_c, 0]`` (E selects ``known_indices``) on
    its own variable block of size ``n_block_vars`` (layout of
    ``build_color_lp``), and ``b_c = (H_c p_c)[known]``. One channel with
    ``H = I`` recovers the plain inpainting constraint ``p[known] = values``.
    """
    blocks, rhs = [], []
    for pix, H in zip(pix_channels, H_list):
        EH = H[known_indices]
        blocks.append(sp.hstack([EH, sp.csr_matrix((EH.shape[0], n_block_vars - EH.shape[1]))]))
        rhs.append((H @ np.asarray(pix, dtype=np.float64))[known_indices])
    return sp.block_diag(blocks, format="csr"), np.concatenate(rhs)


def lp_primal_residual(mats, x: np.ndarray) -> float:
    """Max violation of ``A x = b``, ``G x >= h`` and ``l <= x <= u``."""
    return float(max(
        np.abs(mats.A @ x - mats.b).max(initial=0.0),
        np.maximum(mats.h - mats.G @ x, 0).max(initial=0.0),
        np.maximum(mats.l - x, 0).max(initial=0.0),
        np.maximum(x - mats.u, 0).max(initial=0.0),
    ))


def solve_lp_checked(mats, tol: float = 1e-5) -> dict:
    """``solve_lp`` that rejects inaccurate solutions.

    Clarabel can stop at its iteration cap with an OPTIMAL_INACCURATE status
    that cvxpy accepts; with blur the returned point can then be far from
    feasible. Retry once with a higher cap, then raise.
    """
    for max_iter in (None, 5000):
        sol = solve_lp(mats, max_iter=max_iter)
        res = lp_primal_residual(mats, np.ravel(sol["raw_x"]))
        if res <= tol:
            return sol
    raise RuntimeError(f"LP reference solve inaccurate: primal residual {res:.2e} > {tol:.0e}")


def init_settings_from_cfg(cfg) -> Tuple[str, float, float, float]:
    """``(init_type, warm_delta, dual_ineq_init, dual_eq_init)`` from a config.

    ``dual_init`` (when not null) sets every dual entry, inequality and equality;
    otherwise the inequality duals are 0.5 if ``warm_dual`` else 0.1 and the
    equality duals 0 (the original init).
    """
    dual_init = cfg.get('dual_init', None)
    if dual_init is not None:
        if bool(cfg.get('warm_dual', False)):
            raise ValueError("set either dual_init or warm_dual, not both")
        d_ineq = d_eq = float(dual_init)
    else:
        d_ineq, d_eq = (0.5 if bool(cfg.get('warm_dual', False)) else 0.1), 0.0
    return (str(cfg.get('init_type', 'center')), float(cfg.get('warm_delta', 0.01)), d_ineq, d_eq)


def _summarize(name: str, arr: np.ndarray) -> Dict[str, float]:
    return {
        f"{name}_min": float(arr.min()),
        f"{name}_median": float(np.median(arr)),
        f"{name}_max": float(arr.max()),
    }


# =============================================================================
# Sample creation
# =============================================================================

def pdlp_sample_creation_run(cfg):
    """Generate TV-inpainting LP instances for the four problem splits.

    Persons are split 28/4/8 (train/val/test) under ``cfg.person_split_seed`` so
    all 10 angles of any Olivetti subject stay in the same split (no leakage).
    OOD reuses the full 400-image pool with ``missing_fraction_out_of_dist``.

    Saves per-split ``.npz`` (image_index, boolean mask, x_opt, y_opt, f_opt,
    M_val, R_val) plus an ``out_of_sample_metadata.npz`` carrying the pooled
    operator-norm and IC-radius bounds (training-pool max × safety factor)
    consumed downstream by PEP interpolation conditions.
    """
    log.info("=" * 60)
    log.info("Generating TV-inpainting (PDLP) sample-creation problem sets")
    log.info("=" * 60)
    log.info(cfg)

    # ---- Config ----
    M = int(cfg.image_M)
    N = int(cfg.image_N)
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = K + K_v + K_h
    lp_upper = 1.0 if bool(cfg.scaled_lp_01) else 255.0

    n_subjects_total = int(cfg.n_subjects_total)
    images_per_subject = int(cfg.images_per_subject)
    n_train = int(cfg.n_subjects_train)
    n_val = int(cfg.n_subjects_val)
    n_test = int(cfg.n_subjects_test)

    miss_in = float(cfg.missing_fraction_in_dist)
    miss_ood = float(cfg.missing_fraction_out_of_dist)
    # NOTE: sample-creation R must be computed with the same PDHG init the
    # trainer's _make_cp_tv_traj_fn uses, or the cached pool R is meaningless.
    # The init settings are saved in the metadata and checked by the trainer.
    init_type, warm_delta, dual_ineq_init, dual_eq_init = init_settings_from_cfg(cfg)
    # Optional Gaussian blur before masking: the LP then observes the blurred
    # image at the known pixels, A = [E H, 0] and b = (H p_orig)[known].
    blur_sigma = float(cfg.get('blur_sigma', 0.0))
    if blur_sigma > 0 and init_type != 'center':
        raise ValueError("blur_sigma > 0 is only supported with init_type='center'.")
    H_blur = gaussian_blur_matrix(M, N, blur_sigma)

    person_split_seed = int(cfg.person_split_seed)
    train_seed_base = int(cfg.training_mask_seed_base)
    val_seed_base = int(cfg.val_mask_seed_base)
    test_seed_base = int(cfg.test_mask_seed_base)
    ood_seed_base = int(cfg.ood_mask_seed_base)

    m_safety = float(cfg.m_safety_factor)
    r_safety = float(cfg.r_safety_factor)

    # ---- Olivetti dataset (fetched once) ----
    log.info("Fetching Olivetti faces dataset…")
    faces = fetch_olivetti_faces()
    images = faces.images.astype(np.float64) * lp_upper  # (400, 64, 64), scaled
    log.info(f"Olivetti images shape={images.shape}, dtype={images.dtype}, range=[0, {lp_upper}]")

    # ---- Person-stratified subject split ----
    train_persons, val_persons, test_persons = split_persons_by_subject(
        person_split_seed, n_train, n_val, n_test, n_total=n_subjects_total
    )
    train_pool = image_pool_for_persons(train_persons, images_per_subject)
    val_pool = image_pool_for_persons(val_persons, images_per_subject)
    test_pool = image_pool_for_persons(test_persons, images_per_subject)
    ood_pool = np.arange(n_subjects_total * images_per_subject)

    # No-leakage check (cheap, defensive)
    assert np.intersect1d(train_persons, val_persons).size == 0
    assert np.intersect1d(train_persons, test_persons).size == 0
    assert np.intersect1d(val_persons, test_persons).size == 0
    log.info(
        f"Subject split: train={len(train_persons)} val={len(val_persons)} test={len(test_persons)} "
        f"(image counts: {len(train_pool)}/{len(val_pool)}/{len(test_pool)}); ood pool={len(ood_pool)}"
    )

    # ---- Per-split builder ----
    def _build_split(name: str, image_pool: np.ndarray, missing_fraction: float, seed_base: int):
        n_corrupted = int(round(missing_fraction * K))
        S = K - n_corrupted
        log.info(
            f"[{name}] building {len(image_pool)} instances "
            f"(missing_fraction={missing_fraction}, n_corrupted={n_corrupted}, S={S})"
        )

        N_split = len(image_pool)
        image_index_batch = np.asarray(image_pool, dtype=np.int32)
        mask_batch = np.ones((N_split, K), dtype=bool)
        x_opt_batch = np.zeros((N_split, n_vars), dtype=np.float64)
        y_opt_batch = np.zeros((N_split, 2 * K_v + 2 * K_h + S), dtype=np.float64)
        f_opt_batch = np.zeros(N_split, dtype=np.float64)
        M_val_batch = np.zeros(N_split, dtype=np.float64)
        R_val_batch = np.zeros(N_split, dtype=np.float64)

        for i, image_index in enumerate(image_pool):
            seed_i = int(seed_base + image_index)
            corrupted = sample_corrupted_indices(M, N, missing_fraction, seed_i)
            mask_flat = np.ones(K, dtype=bool)
            mask_flat[corrupted] = False
            known_indices = np.flatnonzero(mask_flat)
            assert known_indices.size == S, (known_indices.size, S)

            image = images[image_index]
            known_values = image.reshape(-1)[known_indices].copy()

            matrices = extract_constraint_matrices(known_indices, known_values, M, N)
            if blur_sigma > 0:
                A_blur, b_blur = blurred_data_block(
                    known_indices, [image.reshape(-1)], [H_blur], n_vars)
                matrices = matrices._replace(A=A_blur, b=b_blur)
            sol = solve_lp_checked(matrices)

            # Spectral norm of K_mat = vstack([G, A]) — operator norm bound.
            K_mat = sp.vstack([matrices.G, matrices.A], format="csr")
            _, sing, _ = spla.svds(K_mat, k=1, which="LM")
            M_val_i = float(sing[0])

            m1 = 2 * K_v + 2 * K_h
            x0, y0 = build_pdhg_init(
                init_type, n_vars, m1, S, lp_upper,
                pix_channels=[image.reshape(-1)], mask=mask_flat, M=M, N=N,
                delta=warm_delta, dual_ineq_init=dual_ineq_init, dual_eq_init=dual_eq_init,
            )
            R_val_i = float(
                np.linalg.norm(np.concatenate([x0 - sol["raw_x"], y0 - sol["raw_y"]]))
            )

            mask_batch[i] = mask_flat
            x_opt_batch[i] = sol["raw_x"]
            y_opt_batch[i] = sol["raw_y"]
            f_opt_batch[i] = float(sol["objective_value"])
            M_val_batch[i] = M_val_i
            R_val_batch[i] = R_val_i

            if (i + 1) % 10 == 0 or i == N_split - 1:
                log.info(
                    f"[{name}] {i+1}/{N_split} solved | "
                    f"f_opt={f_opt_batch[i]:.4f} M_val={M_val_i:.4f} R_val={R_val_i:.4f}"
                )

        out_path = f"{name}_set.npz"
        np.savez_compressed(
            out_path,
            image_index_batch=image_index_batch,
            mask_batch=mask_batch,
            x_opt_batch=x_opt_batch,
            y_opt_batch=y_opt_batch,
            f_opt_batch=f_opt_batch,
            M_val_batch=M_val_batch,
            R_val_batch=R_val_batch,
            missing_fraction=np.float64(missing_fraction),
        )
        log.info(f"[{name}] wrote {out_path}")
        return {
            "image_index_batch": image_index_batch,
            "mask_batch": mask_batch,
            "x_opt_batch": x_opt_batch,
            "y_opt_batch": y_opt_batch,
            "f_opt_batch": f_opt_batch,
            "M_val_batch": M_val_batch,
            "R_val_batch": R_val_batch,
            "S": S,
        }

    train_set = _build_split("training", train_pool, miss_in, train_seed_base)
    val_set = _build_split("validation", val_pool, miss_in, val_seed_base)
    test_set = _build_split("test", test_pool, miss_in, test_seed_base)
    ood_set = _build_split("ood", ood_pool, miss_ood, ood_seed_base)

    # ---- Convenience split-test files (parallel to lasso's b_test_samples.npz etc.) ----
    np.savez_compressed(
        "image_index_test_samples.npz", image_index=test_set["image_index_batch"]
    )
    np.savez_compressed("mask_test_samples.npz", mask=test_set["mask_batch"])
    np.savez_compressed("x_opt_test_samples.npz", x_opt=test_set["x_opt_batch"])
    np.savez_compressed("y_opt_test_samples.npz", y_opt=test_set["y_opt_batch"])
    np.savez_compressed("f_opt_test_samples.npz", f_opt=test_set["f_opt_batch"])
    np.savez_compressed(
        "image_index_out_of_dist_samples.npz", image_index=ood_set["image_index_batch"]
    )
    np.savez_compressed("mask_out_of_dist_samples.npz", mask=ood_set["mask_batch"])
    np.savez_compressed("x_opt_out_of_dist_samples.npz", x_opt=ood_set["x_opt_batch"])
    np.savez_compressed("y_opt_out_of_dist_samples.npz", y_opt=ood_set["y_opt_batch"])
    np.savez_compressed("f_opt_out_of_dist_samples.npz", f_opt=ood_set["f_opt_batch"])

    # ---- Pooled bounds from training pool (for PEP IC) ----
    M_val_pool = m_safety * float(train_set["M_val_batch"].max())
    R_val_pool = r_safety * float(train_set["R_val_batch"].max())
    log.info("=" * 60)
    log.info(
        f"[POOL] M_val = {M_val_pool:.4f} "
        f"(train max={train_set['M_val_batch'].max():.4f}, safety={m_safety})"
    )
    log.info(
        f"[POOL] R_val = {R_val_pool:.4f} "
        f"(train max={train_set['R_val_batch'].max():.4f}, safety={r_safety})"
    )
    for split_name, split_data in [
        ("train", train_set), ("val", val_set), ("test", test_set), ("ood", ood_set)
    ]:
        m_summ = _summarize("M_val", split_data["M_val_batch"])
        r_summ = _summarize("R_val", split_data["R_val_batch"])
        log.info(f"[{split_name}] {m_summ} | {r_summ}")
    log.info("=" * 60)

    # ---- Metadata ----
    metadata: Dict[str, Any] = {
        # Image / problem shape
        "M": int(M),
        "N": int(N),
        "K": int(K),
        "K_v": int(K_v),
        "K_h": int(K_h),
        "n_vars": int(n_vars),
        "S_in_dist": int(train_set["S"]),  # constant across train/val/test under deterministic count
        "S_out_of_dist": int(ood_set["S"]),
        "lp_upper": float(lp_upper),
        "scaled_lp_01": bool(cfg.scaled_lp_01),
        # Subject split
        "train_persons": train_persons.astype(np.int32),
        "val_persons": val_persons.astype(np.int32),
        "test_persons": test_persons.astype(np.int32),
        "person_split_seed": int(person_split_seed),
        # Sample sizes (== pool sizes under one-mask-per-image)
        "training_sample_N": int(len(train_pool)),
        "out_of_sample_val_N": int(len(val_pool)),
        "out_of_sample_test_N": int(len(test_pool)),
        "out_of_dist_N": int(len(ood_pool)),
        # Mask seeds
        "training_mask_seed_base": int(train_seed_base),
        "val_mask_seed_base": int(val_seed_base),
        "test_mask_seed_base": int(test_seed_base),
        "ood_mask_seed_base": int(ood_seed_base),
        # Corruption
        "missing_fraction_in_dist": float(miss_in),
        "missing_fraction_out_of_dist": float(miss_ood),
        # Pooled PEP bounds
        "M_val": float(M_val_pool),
        "R_val": float(R_val_pool),
        "m_safety_factor": float(m_safety),
        "r_safety_factor": float(r_safety),
        # PDHG init used for R (the trainer must match it)
        "init_type": np.str_(init_type),
        "warm_delta": float(warm_delta),
        "dual_ineq_init": float(dual_ineq_init),
        "dual_eq_init": float(dual_eq_init),
        "blur_sigma": float(blur_sigma),
    }
    # Per-split diagnostic summaries
    for split_name, split_data in [
        ("train", train_set), ("val", val_set), ("test", test_set), ("ood", ood_set)
    ]:
        for k, v in _summarize(f"M_val_{split_name}", split_data["M_val_batch"]).items():
            metadata[k] = float(v)
        for k, v in _summarize(f"R_val_{split_name}", split_data["R_val_batch"]).items():
            metadata[k] = float(v)

    np.savez_compressed("out_of_sample_metadata.npz", **metadata)
    log.info("Saved out_of_sample_metadata.npz")
    log.info("=== TV-inpainting (PDLP) sample-creation complete ===")


# =============================================================================
# Training-time wiring
# =============================================================================

def _build_G_sparse(M: int, N: int) -> sp.csr_matrix:
    """Build the mask-independent inequality block G as a scipy sparse matrix.

    G enforces the four blocks  v >= +/- D_v p,  w >= +/- D_h p  via
    ``G x >= 0`` with x = [p; v; w]. Reused across every instance — only the
    A_mask block changes per mask. Sparsity is ~2 nnz per row (one ±1 in
    pixel, one +1 in slack); densifying at 64×64 wastes ~1.5 GB.
    """
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    D_v = _vertical_diff_matrix(M, N)
    D_h = _horizontal_diff_matrix(M, N)
    I_v = sp.eye(K_v, format="csr")
    I_h = sp.eye(K_h, format="csr")
    Z_vh = sp.csr_matrix((K_v, K_h))
    Z_hv = sp.csr_matrix((K_h, K_v))
    return sp.bmat(
        [
            [-D_v, I_v, Z_vh],
            [D_v, I_v, Z_vh],
            [-D_h, Z_hv, I_h],
            [D_h, Z_hv, I_h],
        ],
        format="csr",
    ).astype(np.float64)


def _jax_warm_start_primal(
    pix: jnp.ndarray, mask: jnp.ndarray, M: int, N: int, delta: float, lp_upper: float,
) -> jnp.ndarray:
    """JAX twin of ``_warm_start_primal_single`` (jit/vmap traceable)."""
    lo, hi = delta * lp_upper, (1.0 - delta) * lp_upper
    P = jnp.clip(jnp.where(mask, pix, 0.0), lo, hi).reshape(M, N)
    v0 = jnp.abs(P[1:, :] - P[:-1, :]).reshape(-1) + lo
    w0 = jnp.abs(P[:, 1:] - P[:, :-1]).reshape(-1) + lo
    return jnp.concatenate([P.reshape(-1), v0, w0])


def _make_cp_tv_traj_fn(
    l: jnp.ndarray,
    u: jnp.ndarray,
    m1: int,
    S: int,
    reconstruct_single: Callable,
    x0_fn: Callable | None = None,
    dual_ineq_init: float = 0.1,
    dual_eq_init: float = 0.0,
    init_fn: Callable | None = None,
) -> Callable:
    """Trajectory wrapper that takes light kwargs and reconstructs K_mat/c/q on the fly.

    The unified trainer kwargs-unpacks ``**full_data`` into traj_fn, so this
    function's parameter names (after ``stepsizes``) must match the keys
    declared by ``get_batched_parameters``. We declare
    ``('image_index', 'mask', 'x_opt', 'y_opt')``; K_mat/c/q are reconstructed
    *inside* this function so they only ever materialize for the single
    instance currently traced (under vmap, one per minibatch slot).

    ``x0_fn(image_index, mask)`` overrides the primal init (warm start); when
    None the strict-interior center ``0.5 * (l + u)`` is used.
    """
    def wrapped_traj_fn(stepsizes, image_index, mask, x_opt, y_opt,
                        K_max, return_Gram_representation=True):
        K_mat, c, q = reconstruct_single(image_index, mask, S)
        m_total = K_mat.shape[0]
        m2 = m_total - m1
        x0 = 0.5 * (l + u) if x0_fn is None else x0_fn(image_index, mask)
        y0 = jnp.concatenate([dual_ineq_init * jnp.ones(m1), dual_eq_init * jnp.ones(m2)])
        if init_fn is not None:   # full (x0, y0), e.g. the consistent start
            x0, y0 = init_fn(image_index, mask)
        return problem_data_to_cp_lp_trajectories(
            stepsizes, c, K_mat, q, l, u, x_opt, y_opt, x0, y0,
            K_max, m1,
            return_Gram_representation=return_Gram_representation,
        )

    return wrapped_traj_fn


CP_METRICS = {'obj_val': 'gap', 'opt_dist_sq_norm': 'dist', 'opt_primal_dist_sq_norm': 'pdist'}


def pep_data_fn_cp(stepsizes, mu, L, R, K_max, pep_obj,
                   composition_type='final', decay_rate=0.9):
    """Adapter for the UnifiedTrainer pep_data_fn signature.

    For CP, ``L`` is repurposed as the operator-norm bound M = ||K||_op (a
    strictly upper-bounding scalar) and ``mu`` is unused. ``pep_obj`` selects
    the metric: 'obj_val' = duality gap, 'opt_dist_sq_norm' = squared distance
    ||x_k - x*||^2 + ||y_k - y*||^2 to the saddle point. ``R`` is the Euclidean
    initial radius.
    """
    if pep_obj not in CP_METRICS:
        raise ValueError(f"CP supports pep_obj in {sorted(CP_METRICS)}, got {pep_obj!r}")
    tau, sigma, theta = stepsizes
    return construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L, R=R, K_max=K_max,
        composition_type=composition_type, decay_rate=decay_rate,
        metric=CP_METRICS[pep_obj],
    )


class PDLPProblemModule(ProblemModule):
    """TV-inpainting LP wrapped for the unified Chambolle–Pock learning loop.

    Loads the cached sample-creation bundles under ``cfg.data_source_dir``,
    reconstructs (c, K_mat, q) per instance from ``(image_index, mask)``, and
    pulls ``M_val``/``R_val`` straight from ``out_of_sample_metadata.npz``
    (no fresh estimation pool — those bounds were computed and stored at
    sample-creation time).
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)

        data_source_dir = cfg.get('data_source_dir', None)
        if data_source_dir is None:
            raise ValueError(
                "PDLP TV-inpainting requires cfg.data_source_dir pointing at a "
                "sample-creation output directory; got None."
            )
        self.data_source_dir = data_source_dir

        meta_path = os.path.join(data_source_dir, 'out_of_sample_metadata.npz')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Missing {meta_path}; rerun python run_sample_creation.py PDLP local."
            )
        meta = np.load(meta_path, allow_pickle=False)

        # Image / problem shape
        self.M_img = int(meta['M'])
        self.N_img = int(meta['N'])
        self.K = int(meta['K'])
        self.K_v = int(meta['K_v'])
        self.K_h = int(meta['K_h'])
        self.n_vars = int(meta['n_vars'])
        self.S_in_dist = int(meta['S_in_dist'])
        self.S_out_of_dist = int(meta['S_out_of_dist'])
        self.lp_upper = float(meta['lp_upper'])

        # In our convention K_mat = [G; A_mask]: m1 is the inequality (G) block.
        self.m1 = 2 * self.K_v + 2 * self.K_h

        # Cached PEP bounds
        self.M_val = float(meta['M_val'])
        self.R_val = float(meta['R_val'])
        # Optional enlarged PEP initial-condition radius (e.g. to cover OOD
        # instances); may only grow R_val, so training samples stay inside it.
        R_val_override = cfg.get('R_val_override', None)
        if R_val_override is not None:
            if float(R_val_override) < self.R_val:
                raise ValueError(
                    f"R_val_override={R_val_override} is below the sample-creation "
                    f"R_val={self.R_val}; training samples would violate the PEP ball."
                )
            log.info(f"R_val override: {self.R_val:.4f} -> {float(R_val_override):.4f}")
            self.R_val = float(R_val_override)

        # PDHG init: R_val is only valid for the init it was computed with.
        # Metadata predating the init_type key used the original center init.
        (self.init_type, self.warm_delta, self.dual_ineq_init,
         self.dual_eq_init) = init_settings_from_cfg(cfg)
        meta_init = (
            str(meta['init_type']) if 'init_type' in meta.files else 'center',
            float(meta['warm_delta']) if 'warm_delta' in meta.files else self.warm_delta,
            float(meta['dual_ineq_init']) if 'dual_ineq_init' in meta.files else 0.1,
            float(meta['dual_eq_init']) if 'dual_eq_init' in meta.files else 0.0,
        )
        cfg_init = (self.init_type, self.warm_delta, self.dual_ineq_init, self.dual_eq_init)
        if meta_init[0] != cfg_init[0] or not np.allclose(meta_init[1:], cfg_init[1:]):
            raise ValueError(
                f"cfg PDHG init (init_type, warm_delta, dual_ineq_init, dual_eq_init)={cfg_init} does not "
                f"match the sample-creation init {meta_init} in {meta_path}."
            )
        # Blur of the forward model: the cached x_opt / M_val / R_val belong to it.
        self.blur_sigma = float(cfg.get('blur_sigma', 0.0))
        meta_blur = float(meta['blur_sigma']) if 'blur_sigma' in meta.files else 0.0
        if not np.isclose(meta_blur, self.blur_sigma):
            raise ValueError(
                f"cfg blur_sigma={self.blur_sigma} does not match sample-creation "
                f"blur_sigma={meta_blur} in {meta_path}."
            )
        log.info(
            f"PDLP TV-inpainting: M={self.M_img} N={self.N_img} n_vars={self.n_vars} "
            f"m1={self.m1} S_in={self.S_in_dist} S_ood={self.S_out_of_dist} "
            f"lp_upper={self.lp_upper}"
        )
        log.info(
            f"Loaded cached M_val={self.M_val:.6f} R_val={self.R_val:.6f} "
            f"from {meta_path}"
        )

        # Box bounds and (fixed) cost vector — same for every instance.
        self.l = jnp.zeros(self.n_vars)
        self.u = self.lp_upper * jnp.ones(self.n_vars)
        self._c_np = np.concatenate([
            np.zeros(self.K),
            np.ones(self.K_v),
            np.ones(self.K_h),
        ]).astype(np.float64)

        # Olivetti images cached locally; pre-scaled to [0, lp_upper] so we
        # don't redo the multiply per instance.
        log.info("Fetching Olivetti faces dataset…")
        self.images_np = (
            fetch_olivetti_faces().images.astype(np.float64) * self.lp_upper
        )

        # JAX-resident fixtures used by the lazy K_mat reconstruction inside
        # the trajectory function. G is mask-independent and sparse (~2 nnz
        # per row); BCOO storage is ~1 MB instead of 1.57 GB dense at 64×64.
        # c_jnp is tiny; images_jnp is ~78 MB. All allocated once.
        log.info("Building mask-independent G inequality block (sparse) on JAX device…")
        G_sp = _build_G_sparse(self.M_img, self.N_img)
        self.G_bcoo = jsparse.BCOO.from_scipy_sparse(G_sp).astype(jnp.float64)
        log.info(
            f"G_bcoo shape={self.G_bcoo.shape} nnz={int(self.G_bcoo.nse)} "
            f"~{(self.G_bcoo.data.nbytes + self.G_bcoo.indices.nbytes) / 1e6:.2f} MB"
        )
        self.c_jnp = jnp.asarray(self._c_np)
        self.images_jnp = jnp.asarray(self.images_np)

        # Blur rows as fixed-width (column, weight) arrays so A = E H can be
        # gathered per instance under jit/vmap. Border rows have fewer
        # nonzeros; they are padded with weight-0 entries at column 0.
        H = gaussian_blur_matrix(self.M_img, self.N_img, self.blur_sigma)
        width = int(np.diff(H.indptr).max())
        cols = np.zeros((self.K, width), dtype=np.int64)
        vals = np.zeros((self.K, width), dtype=np.float64)
        for r in range(self.K):
            lo, hi = H.indptr[r], H.indptr[r + 1]
            cols[r, :hi - lo] = H.indices[lo:hi]
            vals[r, :hi - lo] = H.data[lo:hi]
        self.H_cols = jnp.asarray(cols)
        self.H_vals = jnp.asarray(vals)

    # ------------------------------------------------------------------------
    # Pool: light per-instance descriptors only. K_mat is rebuilt lazily.
    # ------------------------------------------------------------------------

    def _reconstruct_from_arrays(
        self,
        image_index_b: np.ndarray,
        mask_b: np.ndarray,
        x_opt_b: np.ndarray,
        y_opt_b: np.ndarray,
        S: int,
    ) -> Tuple[ProblemData, GroundTruth]:
        """Return only the lightweight descriptors needed to reconstruct K_mat.

        Pool size for 280 instances: ~73 MB total. The full K_mat is rebuilt
        per-instance inside the trajectory function via ``_reconstruct_single``
        once per SGD step (wrapped in ``vmap``), then garbage-collected.
        """
        # Cheap per-batch validation that masks are well-formed (S True entries).
        for i in range(image_index_b.shape[0]):
            n_known = int(np.count_nonzero(mask_b[i]))
            if n_known != S:
                raise RuntimeError(
                    f"instance {i}: mask has {n_known} known pixels, expected S={S}."
                )
        return (
            {
                'image_index_batch': jnp.asarray(image_index_b, dtype=jnp.int32),
                'mask_batch': jnp.asarray(mask_b, dtype=jnp.bool_),
            },
            {
                'x_opt_batch': jnp.asarray(x_opt_b),
                'y_opt_batch': jnp.asarray(y_opt_b),
            },
        )

    def _reconstruct_single(self, image_index, mask, S: int):
        """Build (K_mat, c, q) for a single instance — all jnp ops, jit/vmap traceable.

        K_mat is returned as a BCOO sparse matrix to avoid materializing the
        full (m1+S, n_vars) dense block under vmap; the trajectory function
        uses it only for matvecs (K @ x and K.T @ y) which BCOO handles
        natively.

        Args:
            image_index: scalar int32.
            mask: (K,) bool with exactly S True entries.
            S: static Python int (closed over per split).

        Returns: (K_mat (m1+S, n_vars) as BCOO, c (n_vars,), q (m1+S,)).
        """
        # jnp.where with size= gives a fixed-shape output. Under deterministic-
        # count masks every instance has exactly S True entries so no dummy
        # fill values are populated.
        known_indices = jnp.where(mask, size=S)[0]
        pix = self.images_jnp[image_index].reshape(-1)

        # A = E H as BCOO: row i holds the blur weights of pixel known_indices[i]
        # (a single 1.0 at column known_indices[i] when there is no blur).
        idx_dtype = self.G_bcoo.indices.dtype
        cols = self.H_cols[known_indices]                    # (S, width)
        vals = self.H_vals[known_indices]
        width = cols.shape[1]
        A_mask_idx = jnp.stack(
            [jnp.repeat(jnp.arange(S, dtype=idx_dtype), width),
             cols.reshape(-1).astype(idx_dtype)],
            axis=-1,
        )
        A_mask_data = vals.reshape(-1).astype(self.G_bcoo.data.dtype)
        A_mask_bcoo = jsparse.BCOO(
            (A_mask_data, A_mask_idx), shape=(S, self.n_vars)
        )
        known_values = jnp.sum(vals * pix[cols], axis=1)     # (H p)[known]
        K_mat = jsparse.bcoo_concatenate([self.G_bcoo, A_mask_bcoo], dimension=0)
        q = jnp.concatenate([jnp.zeros(self.m1), known_values])
        return K_mat, self.c_jnp, q

    def _load_split(
        self, split_name: str, S: int, N: int, seed: int,
    ) -> Tuple[ProblemData, GroundTruth]:
        """Load and (seeded) subsample N rows from a saved split, then reconstruct."""
        npz_path = os.path.join(self.data_source_dir, f'{split_name}_set.npz')
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Expected {npz_path}.")
        d = np.load(npz_path)
        total = int(d['image_index_batch'].shape[0])
        if N >= total:
            idx = np.arange(total)
            if N > total:
                log.warning(
                    f"{split_name}: requested N={N} > available {total}; using all {total}."
                )
        else:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(total, size=N, replace=False)
        log.info(f"{split_name}: loaded {len(idx)} / {total} from {npz_path}")
        return self._reconstruct_from_arrays(
            d['image_index_batch'][idx],
            d['mask_batch'][idx],
            d['x_opt_batch'][idx],
            d['y_opt_batch'][idx],
            S,
        )

    # ------------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------------

    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        return self._load_split(
            'training', self.S_in_dist, N,
            int(self.cfg.get('training_mask_seed_base', 40000)),
        )

    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        return self._load_split(
            'validation', self.S_in_dist, N,
            int(self.cfg.get('val_mask_seed_base', 10000)),
        )

    def sample_test_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        return self._load_split(
            'test', self.S_in_dist, N,
            int(self.cfg.get('test_mask_seed_base', 20000)),
        )

    def _sample_ood_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        return self._load_split(
            'ood', self.S_out_of_dist, N,
            int(self.cfg.get('ood_mask_seed_base', 30000)),
        )

    # ------------------------------------------------------------------------
    # Trajectory / PEP wiring
    # ------------------------------------------------------------------------

    def get_trajectory_fn(self, alg: str) -> Callable:
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        # Bake S=S_in_dist into the wrapper. Training and validation are both
        # in-distribution; OOD evaluation (when added) will request a separate
        # traj_fn with S=self.S_out_of_dist.
        x0_fn = None
        if self.init_type == 'warm':
            def x0_fn(image_index, mask):
                return _jax_warm_start_primal(
                    self.images_jnp[image_index].reshape(-1), mask,
                    self.M_img, self.N_img, self.warm_delta, self.lp_upper,
                )
        init_fn = None
        if self.init_type in ('consistent', 'consistent_holes'):
            S = self.S_in_dist
            hv = 0.5 if self.init_type == 'consistent_holes' else None

            def init_fn(image_index, mask):
                x0, yG, yA = _consistent_init_single(
                    jnp, self.images_jnp[image_index].reshape(-1), mask,
                    self.M_img, self.N_img, self.warm_delta, self.lp_upper, S, hole_value=hv)
                return x0, jnp.concatenate([yG, yA])
        return _make_cp_tv_traj_fn(
            self.l, self.u, self.m1, self.S_in_dist, self._reconstruct_single,
            x0_fn=x0_fn, dual_ineq_init=self.dual_ineq_init, dual_eq_init=self.dual_eq_init,
            init_fn=init_fn,
        )

    def get_pep_data_fn(self, alg: str) -> Callable:
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        return pep_data_fn_cp

    # ------------------------------------------------------------------------
    # Problem parameters / stepsizes
    # ------------------------------------------------------------------------

    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        return (self.M_val, 0.0, self.R_val)

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        M = L
        tau_scalar = 0.5 / M
        sigma_scalar = 0.5 / M
        theta_scalar = 1.0
        if self.cfg.stepsize_type == "vector":
            tau = jnp.full(K, tau_scalar)
            sigma = jnp.full(K, sigma_scalar)
            theta = jnp.full(K, theta_scalar)
        else:
            tau = jnp.array(tau_scalar)
            sigma = jnp.array(sigma_scalar)
            theta = jnp.array(theta_scalar)
        return (tau, sigma, theta)

    # ------------------------------------------------------------------------
    # DataFrame formatting
    # ------------------------------------------------------------------------

    def build_stepsizes_dataframe(
        self,
        stepsizes_history: list,
        K_max: int,
        alg: str,
        training_losses=None,
        validation_losses=None,
        times=None,
        raw_grad_norms=None,
        lrs=None,
    ) -> pd.DataFrame:
        tau_sample = stepsizes_history[0][0]
        is_vector = jnp.ndim(tau_sample) > 0
        data: Dict[str, list] = {'iteration': list(range(len(stepsizes_history)))}
        if training_losses is not None:
            data['training_loss'] = [float(x) for x in training_losses]
        if validation_losses is not None:
            data['validation_loss'] = [float(x) for x in validation_losses]
        if times is not None:
            data['iter_time'] = [float(x) for x in times]
        if raw_grad_norms is not None:
            data['raw_grad_norm'] = [float(x) for x in raw_grad_norms]
        if lrs is not None:
            data['lr'] = [float(x) for x in lrs]
        if is_vector:
            for k in range(K_max):
                data[f'tau_{k}'] = [float(ss[0][k]) for ss in stepsizes_history]
                data[f'sigma_{k}'] = [float(ss[1][k]) for ss in stepsizes_history]
                data[f'theta_{k}'] = [float(ss[2][k]) for ss in stepsizes_history]
        else:
            data['tau'] = [float(ss[0]) for ss in stepsizes_history]
            data['sigma'] = [float(ss[1]) for ss in stepsizes_history]
            data['theta'] = [float(ss[2]) for ss in stepsizes_history]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------------
    # Parameter / ground-truth declarations
    # ------------------------------------------------------------------------

    def get_batched_parameters(self) -> ParameterNames:
        # K_mat / c / q are reconstructed lazily inside the traj_fn, so they
        # don't appear in the pool. We declare only the lightweight
        # descriptors plus the cached optima (also needed in the trajectory
        # for the PEP basis).
        return ('image_index', 'mask', 'x_opt', 'y_opt')

    def get_fixed_parameters(self) -> ParameterNames:
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        return ('x_opt', 'y_opt')

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        return (4 * K + 11, 2 * (K + 2))

    # ------------------------------------------------------------------------
    # Batched trajectory computation
    # ------------------------------------------------------------------------

    def compute_batched_trajectories(
        self,
        stepsizes: Stepsizes,
        batched_data: Dict[str, jnp.ndarray],
        fixed_data: Dict[str, jnp.ndarray],
        traj_fn: Callable,
        K_max: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # K_mat is reconstructed inside traj_fn (one per vmap slot), so peak
        # memory is bounded by N * (m_total * n_vars * 8 bytes); transient.
        batch_GF_func = jax.vmap(
            lambda image_index, mask, x_opt, y_opt: traj_fn(
                stepsizes, image_index, mask, x_opt, y_opt, K_max,
                return_Gram_representation=True,
            ),
            in_axes=(0, 0, 0, 0),
        )
        return batch_GF_func(
            batched_data['image_index'],
            batched_data['mask'],
            batched_data['x_opt'],
            batched_data['y_opt'],
        )

    # ------------------------------------------------------------------------
    # Metric: CP duality gap
    # ------------------------------------------------------------------------

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData,
        ground_truth: GroundTruth, pep_obj: str,
    ) -> Callable[[int], float]:
        if pep_obj not in CP_METRICS:
            raise NotImplementedError(
                f"PDLP supports pep_obj in {sorted(CP_METRICS)}; got {pep_obj!r}"
            )

        # Per-instance under vmap: image_index is a scalar, mask is (K,).
        # Rebuild K_mat/c/q for this single instance.
        K_mat, c, q = self._reconstruct_single(
            problem_data['image_index'], problem_data['mask'], self.S_in_dist,
        )
        x_opt = ground_truth['x_opt']
        y_opt = ground_truth['y_opt']

        v_iter = trajectories[0]
        y_iter = trajectories[1]

        def L(vv, yy):
            return c @ vv - yy @ K_mat @ vv + q @ yy

        if pep_obj == 'opt_dist_sq_norm':
            def metric_fn(k):
                dx, dy = v_iter[k] - x_opt, y_iter[k] - y_opt
                return dx @ dx + dy @ dy
        elif pep_obj == 'opt_primal_dist_sq_norm':
            def metric_fn(k):
                dx = v_iter[k] - x_opt
                return dx @ dx
        else:
            def metric_fn(k):
                return L(v_iter[k], y_opt) - L(x_opt, y_iter[k])

        return metric_fn

    # ------------------------------------------------------------------------
    # Out-of-sample
    # ------------------------------------------------------------------------

    def generate_out_of_sample_data(
        self, key: jax.Array,
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        N_val = int(self.cfg.get('out_of_sample_val_N', 40))
        N_test = int(self.cfg.get('out_of_sample_test_N', 80))
        N_ood = int(self.cfg.get('out_of_dist_N', 400))
        key, val_key, test_key, ood_key = jax.random.split(key, 4)
        val = self.sample_validation_batch(val_key, N_val)
        test = self.sample_test_batch(test_key, N_test)
        ood = self._sample_ood_batch(ood_key, N_ood)
        return {'validation': val, 'test': test, 'ood': ood}

    # ------------------------------------------------------------------------
    # Algorithm support / validation
    # ------------------------------------------------------------------------

    def get_supported_algorithms(self) -> list[str]:
        return ['cp']

    def validate_config(self) -> None:
        alg = self.cfg.get('alg', 'cp')
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")


# =============================================================================
# Entry point: training loop over K_max
# =============================================================================

def pdlp_run(cfg):
    """Run PDLP TV-inpainting learning experiment.

    Loops over ``cfg.K_max`` values, runs UnifiedTrainer for each K, saves
    per-K progress CSVs.
    """
    log.info("=" * 60)
    log.info("Starting PDLP (TV inpainting) learning experiment")
    log.info("=" * 60)
    log.info(cfg)

    key = jax.random.PRNGKey(int(cfg.sgd_seed))

    problem_module = PDLPProblemModule(cfg)
    problem_module.validate_config()

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    key, train_key = jax.random.split(key)
    trainer = UnifiedTrainer(problem_module, cfg, train_key)
    trainer.prepare_data(save_dir=output_dir)

    for K in cfg.K_max:
        log.info(f"=== Starting training for K={K} ===")
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")

        result = trainer.train(K, csv_path, K_output_dir)

        tau0 = result.stepsizes[0]
        is_vector = jnp.ndim(tau0) > 0
        tau_str = str(tau0.tolist()) if is_vector else f'{float(tau0):.6f}'
        log.info(f"K={K} complete. Final tau={tau_str}. Saved to {csv_path}")

    log.info("=== PDLP TV-inpainting experiment complete ===")
