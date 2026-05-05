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
    # NOTE: the trainer's _make_cp_tv_traj_fn always uses the strict-interior
    # init x0 = 0.5*lp_upper*1, y0 = [0.1*1; 0]; sample-creation R must be
    # computed with that same init or the cached pool R is meaningless.

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
            sol = solve_lp(matrices)

            # Spectral norm of K_mat = vstack([G, A]) — operator norm bound.
            K_mat = sp.vstack([matrices.G, matrices.A], format="csr")
            _, sing, _ = spla.svds(K_mat, k=1, which="LM")
            M_val_i = float(sing[0])

            m1 = 2 * K_v + 2 * K_h
            x0, y0 = build_strict_interior_init(n_vars, m1, S, lp_upper)
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


def _make_cp_tv_traj_fn(
    l: jnp.ndarray,
    u: jnp.ndarray,
    m1: int,
    S: int,
    reconstruct_single: Callable,
) -> Callable:
    """Trajectory wrapper that takes light kwargs and reconstructs K_mat/c/q on the fly.

    The unified trainer kwargs-unpacks ``**full_data`` into traj_fn, so this
    function's parameter names (after ``stepsizes``) must match the keys
    declared by ``get_batched_parameters``. We declare
    ``('image_index', 'mask', 'x_opt', 'y_opt')``; K_mat/c/q are reconstructed
    *inside* this function so they only ever materialize for the single
    instance currently traced (under vmap, one per minibatch slot).
    """
    def wrapped_traj_fn(stepsizes, image_index, mask, x_opt, y_opt,
                        K_max, return_Gram_representation=True):
        K_mat, c, q = reconstruct_single(image_index, mask, S)
        m_total = K_mat.shape[0]
        m2 = m_total - m1
        x0 = 0.5 * (l + u)
        y0 = jnp.concatenate([0.1 * jnp.ones(m1), jnp.zeros(m2)])
        return problem_data_to_cp_lp_trajectories(
            stepsizes, c, K_mat, q, l, u, x_opt, y_opt, x0, y0,
            K_max, m1,
            return_Gram_representation=return_Gram_representation,
        )

    return wrapped_traj_fn


def pep_data_fn_cp(stepsizes, mu, L, R, K_max, pep_obj,
                   composition_type='final', decay_rate=0.9):
    """Adapter for the UnifiedTrainer pep_data_fn signature.

    For CP, ``L`` is repurposed as the operator-norm bound M = ||K||_op (a
    strictly upper-bounding scalar), while ``mu`` and ``pep_obj`` are unused
    (the CP objective is fixed to the duality gap). ``R`` is the Lyapunov
    radius for the P-norm IC.
    """
    tau, sigma, theta = stepsizes
    return construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L, R=R, K_max=K_max,
        composition_type=composition_type, decay_rate=decay_rate,
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
        known_values = self.images_jnp[image_index].reshape(-1)[known_indices]

        # A_mask as BCOO: row i has a single 1.0 at column known_indices[i].
        idx_dtype = self.G_bcoo.indices.dtype
        A_mask_idx = jnp.stack(
            [jnp.arange(S, dtype=idx_dtype), known_indices.astype(idx_dtype)],
            axis=-1,
        )
        A_mask_data = jnp.ones(S, dtype=self.G_bcoo.data.dtype)
        A_mask_bcoo = jsparse.BCOO(
            (A_mask_data, A_mask_idx), shape=(S, self.n_vars)
        )
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
        return _make_cp_tv_traj_fn(
            self.l, self.u, self.m1, self.S_in_dist, self._reconstruct_single,
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
        if pep_obj != 'obj_val':
            raise NotImplementedError(
                f"PDLP only supports pep_obj='obj_val' (duality gap); got {pep_obj!r}"
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
