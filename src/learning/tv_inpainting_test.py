"""
Anisotropic (1-norm) Total Variation Inpainting LP - Sparse Matrix Extraction

This module sets up the L1 total variation image inpainting problem as a Linear
Program in the standard form:

    min  c^T x
    s.t. l <= x <= u
         A x = b
         G x >= h

Image is the first Olivetti face from sklearn.datasets.fetch_olivetti_faces,
shape (M, N) = (64, 64), pixel range [0, 1] or [0, 255] depending on the
SCALED_LP_01 flag. Variable layout:

    x = [p; v; w]    of length n_vars = K + K_v + K_h
        p in R^K      flattened image pixels         (K = M*N)
        v in R^{K_v}  vertical-difference auxiliaries (K_v = (M-1)*N)
        w in R^{K_h}  horizontal-difference auxs.    (K_h = M*(N-1))

The auxiliaries upper-bound the absolute pixel-to-pixel differences via the four
inequality blocks  v >= +/- D_v p,  w >= +/- D_h p, encoded in G x >= h. The
equality block A x = b pins p to the known pixel values from the mask.

All constraint matrices are scipy.sparse.csr_matrix; vectors are numpy.ndarray.
"""

from functools import partial
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.datasets import fetch_olivetti_faces
from tqdm import trange

np.set_printoptions(suppress=True, precision=5)

# Fraction of pixels marked as unknown in the random mask. Tune to taste.
MISSING_FRACTION = 0.1

# Number of PDHG iterations to run in the __main__ demo. Tune to taste.
K_MAX = 5

# Number of times to repeat the loaded learned stepsize schedule end-to-end.
# Effective horizon for learned PDHG is K_MAX * NUM_REPS. NUM_REPS=1 keeps the
# original behavior (run the schedule once).
NUM_REPS = 1

# If True, the LP operates in [0, 1] (image left at the dataset's native scale,
# u = 1). If False, the LP operates in [0, 255] (image upscaled by 255, u = 255).
# Plotting always rescales to [0, 255] for imshow.
SCALED_LP_01 = True
LP_UPPER = 1.0 if SCALED_LP_01 else 255.0


class TVInpaintingMatrices(NamedTuple):
    """Container for the LP constraint data in the user's standard form."""
    c: np.ndarray
    A: sp.csr_matrix
    b: np.ndarray
    G: sp.csr_matrix
    h: np.ndarray
    l: np.ndarray
    u: np.ndarray


def generate_tv_inpainting_problem(
    missing_fraction: float,
    random_seed: int,
    face_index: int = 0,
) -> dict:
    """Load an Olivetti face (scaled by LP_UPPER) and randomly mask pixels.

    Args:
        missing_fraction: Fraction of pixels to mark as unknown (in [0, 1]).
        random_seed: Seed for the mask RNG.
        face_index: Which face from the Olivetti dataset to use.

    Returns:
        Dict with the original image, the bool mask of known pixels, the flat
        indices and values of those known pixels, and image dimensions.
    """
    faces = fetch_olivetti_faces()
    image = faces.images[face_index].astype(np.float64) * LP_UPPER
    M, N = image.shape

    rng = np.random.default_rng(random_seed)
    mask = rng.random((M, N)) >= missing_fraction
    known_indices = np.flatnonzero(mask)
    known_values = image.reshape(-1)[known_indices].copy()

    return {
        "image": image,
        "mask": mask,
        "known_indices": known_indices,
        "known_values": known_values,
        "M": M,
        "N": N,
    }


def _vertical_diff_matrix(M: int, N: int) -> sp.csr_matrix:
    """D_v p computes p[i+1, j] - p[i, j], shape (K_v, K)."""
    K = M * N
    K_v = (M - 1) * N
    r = np.arange(K_v)
    rows = np.repeat(r, 2)
    cols = np.empty(2 * K_v, dtype=np.int64)
    cols[0::2] = r
    cols[1::2] = r + N
    data = np.empty(2 * K_v, dtype=np.float64)
    data[0::2] = -1.0
    data[1::2] = +1.0
    return sp.coo_matrix((data, (rows, cols)), shape=(K_v, K)).tocsr()


def _horizontal_diff_matrix(M: int, N: int) -> sp.csr_matrix:
    """D_h p computes p[i, j+1] - p[i, j], shape (K_h, K)."""
    K = M * N
    K_h = M * (N - 1)
    r = np.arange(K_h)
    i = r // (N - 1)
    j = r - i * (N - 1)
    col_low = i * N + j
    rows = np.repeat(r, 2)
    cols = np.empty(2 * K_h, dtype=np.int64)
    cols[0::2] = col_low
    cols[1::2] = col_low + 1
    data = np.empty(2 * K_h, dtype=np.float64)
    data[0::2] = -1.0
    data[1::2] = +1.0
    return sp.coo_matrix((data, (rows, cols)), shape=(K_h, K)).tocsr()


def extract_constraint_matrices(
    known_indices: np.ndarray,
    known_values: np.ndarray,
    M: int,
    N: int,
) -> TVInpaintingMatrices:
    """Build (c, A, b, G, h, l, u) for the 1-norm TV inpainting LP.

    Args:
        known_indices: Flat (C-order) pixel indices that are observed, length S.
        known_values: Observed pixel values aligned to known_indices, length S.
        M, N: Image dimensions.

    Returns:
        TVInpaintingMatrices in the standard LP form
            min c^T x   s.t.   l <= x <= u,  A x = b,  G x >= h.
    """
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = K + K_v + K_h
    S = len(known_indices)

    c = np.concatenate([np.zeros(K), np.ones(K_v), np.ones(K_h)])
    l = np.zeros(n_vars)
    u = LP_UPPER * np.ones(n_vars)

    D_v = _vertical_diff_matrix(M, N)
    D_h = _horizontal_diff_matrix(M, N)

    A_mask = sp.coo_matrix(
        (np.ones(S, dtype=np.float64), (np.arange(S), np.asarray(known_indices, dtype=np.int64))),
        shape=(S, K),
    ).tocsr()
    A = sp.hstack(
        [A_mask, sp.csr_matrix((S, K_v)), sp.csr_matrix((S, K_h))],
        format="csr",
    )
    b = np.asarray(known_values, dtype=np.float64)

    I_v = sp.eye(K_v, format="csr")
    I_h = sp.eye(K_h, format="csr")
    Z_vh = sp.csr_matrix((K_v, K_h))
    Z_hv = sp.csr_matrix((K_h, K_v))
    G = sp.bmat(
        [
            [-D_v, I_v, Z_vh],
            [D_v, I_v, Z_vh],
            [-D_h, Z_hv, I_h],
            [D_h, Z_hv, I_h],
        ],
        format="csr",
    )
    h = np.zeros(2 * K_v + 2 * K_h)

    return TVInpaintingMatrices(c=c, A=A, b=b, G=G, h=h, l=l, u=u)


def make_matrix_extractor(known_indices: np.ndarray, M: int, N: int):
    """Closure with fixed mask + dimensions; only known_values varies."""
    return partial(
        extract_constraint_matrices,
        known_indices=known_indices,
        M=M,
        N=N,
    )


def solve_lp(matrices: TVInpaintingMatrices) -> dict:
    """Solve the LP via CVXPY in the user's standard form."""
    import cvxpy as cp

    n_vars = matrices.c.shape[0]
    x = cp.Variable(n_vars)

    constraints = [
        matrices.A @ x == matrices.b,
        matrices.G @ x >= matrices.h,
        x >= matrices.l,
        x <= matrices.u,
    ]
    prob = cp.Problem(cp.Minimize(matrices.c @ x), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")

    K = matrices.A.shape[1] - matrices.G.shape[0] // 2
    raw_x = np.asarray(x.value)

    # Dual recovery (matches PDHG convention y = [y_G; y_A], y_G >= 0):
    #   CVXPY mu_eq for (Ax == b),  mu_ineq for (Gx >= h),  with mu_ineq >= 0.
    #   PDHG sign convention: y_G = mu_ineq,  y_A = -mu_eq.
    mu_eq = np.asarray(constraints[0].dual_value)
    mu_ineq = np.asarray(constraints[1].dual_value)
    raw_y = np.concatenate([mu_ineq, -mu_eq])

    return {
        "objective_value": prob.value,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "p": raw_x[:K],
        "vw": raw_x[K:],
    }


def verify_matrices(problem: dict, matrices: TVInpaintingMatrices) -> bool:
    """Compare the LP-form objective to a direct cp.diff-based formulation."""
    import cvxpy as cp

    M, N = problem["M"], problem["N"]
    image = np.asarray(problem["image"])
    mask = np.asarray(problem["mask"])

    sol_matrices = solve_lp(matrices)

    P = cp.Variable((M, N))
    obj = cp.sum(cp.abs(cp.diff(P, axis=0))) + cp.sum(cp.abs(cp.diff(P, axis=1)))
    cons = [P[mask] == image[mask], P >= 0, P <= LP_UPPER]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    print(f"Matrix formulation objective: {sol_matrices['objective_value']:.6f}")
    print(f"Direct CVXPY objective:       {prob.value:.6f}")
    print(f"Difference:                   {abs(sol_matrices['objective_value'] - prob.value):.2e}")

    return np.isclose(sol_matrices["objective_value"], prob.value, rtol=1e-5)


def run_PDHG(
    c: np.ndarray,
    G: sp.csr_matrix,
    h: np.ndarray,
    A: sp.csr_matrix,
    b: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    raw_xs: np.ndarray,
    raw_ys: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chambolle-Pock PDHG for the standard-form LP, sparse mat-vecs.

    Iterates from the warm start (x0, y0) and returns the final (x, y).
    """
    K_mat = sp.vstack([G, A], format="csr")
    K_T = K_mat.T.tocsr()
    q = np.concatenate([h, b])
    m1 = G.shape[0]

    def satlin(v: np.ndarray) -> np.ndarray:
        return np.minimum(u, np.maximum(v, l))

    def partial_relu(v: np.ndarray) -> np.ndarray:
        out = v.copy()
        out[:m1] = np.maximum(out[:m1], 0.0)
        return out

    def lagrangian(x: np.ndarray, y: np.ndarray) -> float:
        return float(c @ x - y @ (K_mat @ x) + q @ y)

    def lagrangian_gap(xk: np.ndarray, yk: np.ndarray) -> float:
        primal = lagrangian(xk, raw_ys)
        dual = lagrangian(raw_xs, yk)
        return primal - dual

    xk = np.asarray(x0, dtype=np.float64).copy()
    yk = np.asarray(y0, dtype=np.float64).copy()
    R = float(np.linalg.norm(np.concatenate([xk - raw_xs, yk - raw_ys])))
    print("warm start to opt norm:", R)

    # Spectral norm via top singular value (sparse).
    _, s, _ = spla.svds(K_mat, k=1, which="LM")
    M_norm = float(s[0])
    print("M:", M_norm)
    tau = 0.9 / M_norm
    sigma = 0.9 / M_norm
    theta = 1

    print(K_mat.shape, q.shape)

    for _ in trange(K_MAX):
        print("lagrangian gap loss:", lagrangian_gap(xk, yk))

        xkplus1 = satlin(xk - tau * (c - K_T @ yk))
        xbar = xkplus1 + theta * (xkplus1 - xk)
        ykplus1 = partial_relu(yk + sigma * (q - K_mat @ xbar))

        xk = xkplus1
        yk = ykplus1

    print(c @ xk)
    return xk, yk


def run_PDHG_with_stepsizes(
    c: np.ndarray,
    G: sp.csr_matrix,
    h: np.ndarray,
    A: sp.csr_matrix,
    b: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    raw_xs: np.ndarray,
    raw_ys: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    tau_arr: np.ndarray,
    sigma_arr: np.ndarray,
    theta_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chambolle-Pock PDHG with per-iteration stepsizes (K_max = len(tau_arr)).

    Same iteration body as `run_PDHG` (sat-lin primal step, partial-relu dual
    step) but reads (tau_k, sigma_k, theta_k) from the supplied arrays each
    step. Skips the M = ||K||_2 computation (stepsizes are pre-determined) and
    prints the Lagrangian gap each step for direct comparison with the
    fixed-stepsize run.
    """
    K_mat = sp.vstack([G, A], format="csr")
    K_T = K_mat.T.tocsr()
    q = np.concatenate([h, b])
    m1 = G.shape[0]

    def satlin(v: np.ndarray) -> np.ndarray:
        return np.minimum(u, np.maximum(v, l))

    def partial_relu(v: np.ndarray) -> np.ndarray:
        out = v.copy()
        out[:m1] = np.maximum(out[:m1], 0.0)
        return out

    def lagrangian(x: np.ndarray, y: np.ndarray) -> float:
        return float(c @ x - y @ (K_mat @ x) + q @ y)

    def lagrangian_gap(xk: np.ndarray, yk: np.ndarray) -> float:
        return lagrangian(xk, raw_ys) - lagrangian(raw_xs, yk)

    xk = np.asarray(x0, dtype=np.float64).copy()
    yk = np.asarray(y0, dtype=np.float64).copy()
    K_max = len(tau_arr)
    assert len(sigma_arr) == K_max and len(theta_arr) == K_max, (
        f"stepsize array length mismatch: tau={len(tau_arr)}, "
        f"sigma={len(sigma_arr)}, theta={len(theta_arr)}"
    )

    for k in trange(K_max):
        print("lagrangian gap loss:", lagrangian_gap(xk, yk))
        tau_k = float(tau_arr[k])
        sigma_k = float(sigma_arr[k])
        theta_k = float(theta_arr[k])
        xkplus1 = satlin(xk - tau_k * (c - K_T @ yk))
        xbar = xkplus1 + theta_k * (xkplus1 - xk)
        ykplus1 = partial_relu(yk + sigma_k * (q - K_mat @ xbar))
        xk = xkplus1
        yk = ykplus1

    print("learned-stepsize final c@x:", c @ xk)
    return xk, yk


def get_benchmark_solution(
    missing_fraction: float,
    random_seed: int,
    face_index: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a related TV inpainting LP and return its optimal (x, y).

    Uses a different face_index by default so the LP shape matches the main
    problem (same mask seed -> same A/G/c/...) but the data b differs, giving a
    distinct primal-dual optimum to seed PDHG diagnostics.
    """
    problem = generate_tv_inpainting_problem(
        missing_fraction=missing_fraction,
        random_seed=random_seed,
        face_index=face_index,
    )
    extractor = make_matrix_extractor(problem["known_indices"], problem["M"], problem["N"])
    matrices = extractor(known_values=problem["known_values"])
    sol = solve_lp(matrices)
    return sol["raw_x"], sol["raw_y"]


if __name__ == "__main__":
    print("=" * 60)
    print("Generating TV Inpainting Problem (Olivetti face, 1-norm TV)")
    print("=" * 60)

    problem = generate_tv_inpainting_problem(
        missing_fraction=MISSING_FRACTION,
        random_seed=42,
        face_index=29,
    )

    M, N = problem["M"], problem["N"]
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = K + K_v + K_h
    S = len(problem["known_indices"])

    print(f"Image dimensions:    M={M}, N={N}")
    print(f"K (pixels):          {K}")
    print(f"K_v (vert diffs):    {K_v}")
    print(f"K_h (horz diffs):    {K_h}")
    print(f"n_vars:              {n_vars}")
    print(f"S (known pixels):    {S}")
    print(f"Inequality rows:     {2 * K_v + 2 * K_h}")

    print("\n" + "=" * 60)
    print("Extracting Sparse Constraint Matrices")
    print("=" * 60)

    extractor = make_matrix_extractor(problem["known_indices"], M, N)
    matrices = extractor(known_values=problem["known_values"])

    print(f"c shape: {matrices.c.shape}")
    print(f"A shape: {matrices.A.shape}, nnz: {matrices.A.nnz}, type: {type(matrices.A).__name__}")
    print(f"b shape: {matrices.b.shape}")
    print(f"G shape: {matrices.G.shape}, nnz: {matrices.G.nnz}, type: {type(matrices.G).__name__}")
    print(f"h shape: {matrices.h.shape}")
    print(f"l shape: {matrices.l.shape}, u shape: {matrices.u.shape}")

    assert sp.issparse(matrices.A)
    assert sp.issparse(matrices.G)

    print("\n" + "=" * 60)
    print("Verifying Matrix Formulation")
    print("=" * 60)

    is_correct = verify_matrices(problem, matrices)
    print(f"Verification passed: {is_correct}")

    print("\n" + "=" * 60)
    print("LP Solution")
    print("=" * 60)

    solution = solve_lp(matrices)
    print(f"Optimal objective: {solution['objective_value']:.4f}")

    print("\n" + "=" * 60)
    print("Testing PDHG (warm-started at strict-interior init)")
    print("=" * 60)

    # Local import avoids a circular dependency: pdlp.py imports helpers from
    # this module at top level, so build_strict_interior_init can only be
    # pulled in once the diagnostic actually runs. Also prepend src/ to
    # sys.path so direct invocation (`python learning/tv_inpainting_test.py`
    # from src/) resolves the sibling package, not just `python -m ...`.
    import os, sys
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from learning_experiment_classes.pdlp import build_strict_interior_init

    m1 = matrices.G.shape[0]
    x0, y0 = build_strict_interior_init(n_vars, m1, S, LP_UPPER)

    xk_final, _ = run_PDHG(
        matrices.c,
        matrices.G,
        matrices.h,
        matrices.A,
        matrices.b,
        matrices.l,
        matrices.u,
        solution["raw_x"],
        solution["raw_y"],
        x0,
        y0,
    )

    print("\n" + "=" * 60)
    print("Testing PDHG with learned per-iteration stepsizes")
    print("=" * 60)

    stepsize_root = os.path.join(
        _SRC_DIR, "learning_experiment_classes", "pdhg_stepsizes",
    )
    # (label, subfolder) — order determines plot order when multiple exist.
    schedule_specs = [("l2o", "l2o"), ("ldro-pep", "ldro-pep")]
    learned_iterates: list[tuple[str, np.ndarray]] = []
    for label, subfolder in schedule_specs:
        s_path = os.path.join(
            stepsize_root, subfolder, f"learned_pdhg_stepsizes_K{K_MAX}.csv",
        )
        if not os.path.exists(s_path):
            print(f"[{label}] No K={K_MAX} CSV at {s_path}; skipping panel.")
            continue
        arr = np.loadtxt(s_path, delimiter=",", skiprows=1)
        tau_arr = np.tile(arr[:, 0], NUM_REPS)
        sigma_arr = np.tile(arr[:, 1], NUM_REPS)
        theta_arr = np.tile(arr[:, 2], NUM_REPS)
        print(f"[{label}] Loaded {arr.shape[0]} (tau, sigma, theta) triples "
              f"from {s_path}; repeating x{NUM_REPS} -> "
              f"{len(tau_arr)} total iterations.")

        xk_learned, _ = run_PDHG_with_stepsizes(
            matrices.c,
            matrices.G,
            matrices.h,
            matrices.A,
            matrices.b,
            matrices.l,
            matrices.u,
            solution["raw_x"],
            solution["raw_y"],
            x0,
            y0,
            tau_arr,
            sigma_arr,
            theta_arr,
        )
        learned_iterates.append((label, xk_learned[:K].reshape(M, N)))

    print("\n" + "=" * 60)
    print("Plotting original / corrupted / LP reconstruction / learned PDHG")
    print("=" * 60)

    import matplotlib.pyplot as plt

    original = problem["image"]
    corrupted = np.where(problem["mask"], original, 0.0)
    reconstructed = solution["raw_x"][:K].reshape(M, N)

    titles = [
        "Original",
        f"Corrupted ({int(round(MISSING_FRACTION * 100))}% missing)",
        "L1-TV Reconstruction (LP)",
    ]
    panels = [original, corrupted, reconstructed]
    horizon_label = (
        f"K={K_MAX}" if NUM_REPS == 1 else f"K={K_MAX}x{NUM_REPS}"
    )
    for label, img in learned_iterates:
        titles.append(f"learned PDHG ({label}), {horizon_label}")
        panels.append(img)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2))
    plot_scale = 255.0 / LP_UPPER
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(plot_scale * img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()
