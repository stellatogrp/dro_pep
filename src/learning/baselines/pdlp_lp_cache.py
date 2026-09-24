"""Build, solve and cache the PDLP (TV-inpainting) LP instances of the paper.

The paper's PDLP evaluation (iclr_data_outputs/archive/pdlp/create_paper_plots.py)
uses, in distribution, all 400 Olivetti faces at MISSING_FRACTION with mask
seed 42 and, out of distribution, N_IMAGES tiny-imagenet RGB images with the
same missing fraction and MASK_SEED. It re-solves every LP with CVXPY /
Clarabel to get (x*, y*, f*) and caches only the resulting gaps, so a new
method needs the LP solutions again. This module solves them once, in
parallel, into per-instance NPZ files, and assembles a split into the batched
column layout that ``coarse_line_search.pdhg_lp`` consumes.

Because the mask depends only on the seed, every instance of a split shares
``known_indices`` and therefore the operator K = [G; A]; only the equality
right-hand side differs. ``assemble_split`` asserts this.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from learning.tv_inpainting_test import (
    LP_UPPER, MISSING_FRACTION, extract_constraint_matrices, solve_lp,
)

MASK_SEED = 42          # == tv_inpainting_color_averages.MASK_SEED and the Olivetti mask seed
IMAGE_SIZE = 64


def gray_known_indices(M=IMAGE_SIZE, N=IMAGE_SIZE, missing_fraction=MISSING_FRACTION,
                       seed=MASK_SEED):
    """Mask of tv_inpainting_test.generate_tv_inpainting_problem, without the dataset fetch."""
    rng = np.random.default_rng(seed)
    mask = rng.random((M, N)) >= missing_fraction
    return np.flatnonzero(mask)


def load_olivetti_images():
    """(400, 64, 64) float64 in [0, LP_UPPER], as generate_tv_inpainting_problem scales them."""
    from sklearn.datasets import fetch_olivetti_faces
    return fetch_olivetti_faces().images.astype(np.float64) * LP_UPPER


def load_tiny_imagenet_images():
    """(N_IMAGES, 64, 64, 3) uint8, the paper's OOD sample (needs the HF cache)."""
    from learning.tv_inpainting_color_averages import (
        IMAGE_SAMPLE_SEED, N_IMAGES, load_color_images,
    )
    return load_color_images(N_IMAGES, IMAGE_SAMPLE_SEED)


def build_matrices(image, kind, missing_fraction=MISSING_FRACTION, seed=MASK_SEED):
    if kind == 'gray':
        M, N = image.shape
        known = gray_known_indices(M, N, missing_fraction, seed)
        return extract_constraint_matrices(known, image.reshape(-1)[known].copy(), M, N)
    if kind == 'color':
        from learning.tv_inpainting_color_averages import build_color_lp
        return build_color_lp(image, missing_fraction, seed)
    raise ValueError(kind)


def solve_and_cache_one(job):
    """Worker: solve one LP unless its cache file exists. Returns (idx, f_opt)."""
    idx, image, kind, cache_path = job
    cache_path = Path(cache_path)
    if cache_path.exists():
        return idx, float(np.load(cache_path)['f_opt'])
    mats = build_matrices(image, kind)
    sol = solve_lp(mats)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, raw_x=sol['raw_x'], raw_y=sol['raw_y'],
                        f_opt=float(sol['objective_value']), b=mats.b)
    return idx, float(sol['objective_value'])


def cache_path(cache_dir, split, idx):
    return Path(cache_dir) / split / f'{idx:04d}.npz'


def ensure_lp_cache(split, images, kind, cache_dir, workers=1, recompute=False):
    """Solve every LP of a split (in parallel) into ``cache_dir/<split>/<idx>.npz``."""
    jobs = []
    for idx, image in enumerate(images):
        p = cache_path(cache_dir, split, idx)
        if recompute and p.exists():
            p.unlink()
        jobs.append((idx, image, kind, str(p)))
    f_opt = np.empty(len(jobs))
    if workers <= 1:
        for job in jobs:
            idx, f = solve_and_cache_one(job)
            f_opt[idx] = f
        return f_opt
    # Single-threaded BLAS in the workers (inherited at spawn time).
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(var, '1')
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda it, **kw: it  # noqa: E731
    with mp.get_context('spawn').Pool(workers) as pool:
        for idx, f in tqdm(pool.imap_unordered(solve_and_cache_one, jobs),
                           total=len(jobs), desc=f'LP solves ({split})'):
            f_opt[idx] = f
    return f_opt


def assemble_split(split, kind, n_instances, cache_dir):
    """Load a split's cached solutions into the batched layout for ``pdhg_lp``.

    Returns dict(K, c, l, u, m1, Qmat (m, N), X0 (n, N), Y0 (m, N),
    Xstar (n, N), Ystar (m, N), f_opt (N,), normK).
    """
    if kind == 'gray':
        template = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    else:
        template = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    mats = build_matrices(template, kind)      # b is zero; replaced per instance
    K = sp.vstack([mats.G, mats.A], format='csr')
    m1 = mats.G.shape[0]
    S_total = mats.A.shape[0]
    n = K.shape[1]

    B = np.empty((S_total, n_instances))
    Xstar = np.empty((n, n_instances))
    Ystar = np.empty((K.shape[0], n_instances))
    f_opt = np.empty(n_instances)
    for idx in range(n_instances):
        z = np.load(cache_path(cache_dir, split, idx))
        B[:, idx] = z['b']
        Xstar[:, idx] = z['raw_x']
        Ystar[:, idx] = z['raw_y']
        f_opt[idx] = float(z['f_opt'])
    Qmat = np.vstack([np.tile(mats.h[:, None], (1, n_instances)), B])

    # Strict-interior init, as learning_experiment_classes/pdlp.py::build_strict_interior_init
    # (not imported: that module pulls in JAX + diffcp).
    x0 = 0.5 * LP_UPPER * np.ones(n)
    y0 = np.concatenate([0.1 * np.ones(m1), np.zeros(S_total)])
    X0 = np.tile(x0[:, None], (1, n_instances))
    Y0 = np.tile(y0[:, None], (1, n_instances))

    normK = float(spla.svds(K, k=1, which='LM', return_singular_vectors=False)[0])
    return dict(K=K, c=mats.c, l=mats.l, u=mats.u, m1=m1, Qmat=Qmat, X0=X0, Y0=Y0,
                Xstar=Xstar, Ystar=Ystar, f_opt=f_opt, normK=normK)
