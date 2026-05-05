"""
Anisotropic (1-norm) Total Variation Deblurring LP - Sparse Matrix Extraction

This module sets up the L1-TV image deblurring problem as a Linear Program in
the standard form:

    min  c^T x
    s.t. l <= x <= u
         A x = b
         G x >= h

The image is the Olivetti face at FACE_INDEX (sklearn.datasets.fetch_olivetti_faces),
shape (M, N) = (64, 64), pixel range [0, 1] or [0, 255] depending on the
SCALED_LP_01 flag. The forward observation model is b = K_blur @ p_true (+ optional
Gaussian noise), where K_blur is a sparse 2D Gaussian blur with reflect padding.
The LP recovers p from b by minimizing TV(p) plus an L1 data-fit penalty:

    min   λ * ||K_blur p - b||_1  +  TV(p)
    s.t.  0 <= p <= LP_UPPER

Variable layout:

    x = [p; v; w; s]    of length n_vars = 2K + K_v + K_h
        p in R^K        flattened image pixels         (K = M*N)
        v in R^{K_v}    vertical-difference auxiliaries (K_v = (M-1)*N)
        w in R^{K_h}    horizontal-difference auxs.    (K_h = M*(N-1))
        s in R^K        L1 data-fit slacks

The TV epigraph is encoded by v >= +/- D_v p, w >= +/- D_h p; the L1 data fit is
encoded by -s <= K_blur p - b <= s. Both live in the inequality block G x >= h;
there is no equality block (A is 0 x n_vars by design).

Compared to TV inpainting, the data-fit operator K_blur couples many pixels per
constraint (low-pass / ill-conditioned), which makes the dual y* respond strongly
to changes in the blur level sigma. This gives a clean OOD knob for L2O vs
LDRO-PEP analysis.
"""

from functools import partial
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.datasets import fetch_olivetti_faces
from tqdm import trange

np.set_printoptions(suppress=True, precision=5)

# Row-1 (training-distribution-like) and row-2 (OOD) Gaussian blur sigmas.
BLUR_SIGMA = 0.5
BLUR_SIGMA_OOD = 1.5

# Kernel half-width; None => auto = ceil(3*sigma) per row.
BLUR_HALF_WIDTH = None

# Weight on the L1 data-fit term in the LP objective; TV terms are unit-weighted.
DATA_FIT_LAMBDA = 5.0

# Optional additive Gaussian noise on b (per row).
NOISE_STD = 0.0
NOISE_STD_OOD = 0.0
NOISE_SEED = 7

# Olivetti face index; held fixed across rows (the OOD signal is sigma).
FACE_INDEX = 30

# Number of PDHG iterations to run in the __main__ demo. Tune to taste.
K_MAX = 8

# Number of times to repeat the loaded learned stepsize schedule end-to-end.
# Effective horizon for learned PDHG is K_MAX * NUM_REPS. NUM_REPS=1 keeps the
# original behavior (run the schedule once).
NUM_REPS = 1

# If True, the LP operates in [0, 1] (image left at the dataset's native scale,
# u = 1). If False, the LP operates in [0, 255] (image upscaled by 255, u = 255).
# Plotting always rescales to [0, 255] for imshow.
SCALED_LP_01 = True
LP_UPPER = 1.0 if SCALED_LP_01 else 255.0


class TVDeblurringMatrices(NamedTuple):
    """Container for the LP constraint data in the user's standard form."""
    c: np.ndarray
    A: sp.csr_matrix
    b: np.ndarray
    G: sp.csr_matrix
    h: np.ndarray
    l: np.ndarray
    u: np.ndarray
    K: int
    K_v: int
    K_h: int


def _reflect_index(idx: np.ndarray, dim: int) -> np.ndarray:
    """Reflect-pad indices into [0, dim). Mirrors at boundaries (no edge repeat)."""
    out = np.where(idx < 0, -idx - 1, idx)
    out = np.where(out >= dim, 2 * dim - out - 1, out)
    return out


def _build_blur_matrix(
    M: int, N: int, sigma: float, half_width: int | None = None
) -> sp.csr_matrix:
    """Sparse 2D Gaussian blur (K x K) with reflect padding, row-major flatten.

    Each output pixel (i, j) is sum_{di, dj} g[di, dj] * input[reflect(i+di), reflect(j+dj)].
    Rows sum to 1 (mass-preserving).
    """
    K = M * N
    if sigma <= 0:
        return sp.eye(K, format="csr")
    if half_width is None:
        half_width = max(1, int(np.ceil(3 * sigma)))

    offs = np.arange(-half_width, half_width + 1)
    g1 = np.exp(-(offs.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    g1 /= g1.sum()
    g2 = np.outer(g1, g1)

    rr = np.arange(M)[:, None].repeat(N, axis=1)  # (M, N)
    cc = np.arange(N)[None, :].repeat(M, axis=0)  # (M, N)
    out_idx = (rr * N + cc).reshape(-1)           # (K,)

    rows_all, cols_all, vals_all = [], [], []
    for di in offs:
        for dj in offs:
            w = float(g2[di + half_width, dj + half_width])
            if w == 0.0:
                continue
            in_i = _reflect_index(rr + di, M)
            in_j = _reflect_index(cc + dj, N)
            in_idx = (in_i * N + in_j).reshape(-1)
            rows_all.append(out_idx)
            cols_all.append(in_idx)
            vals_all.append(np.full(K, w, dtype=np.float64))

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    K_blur = sp.coo_matrix((vals, (rows, cols)), shape=(K, K)).tocsr()
    K_blur.sum_duplicates()
    return K_blur


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


def generate_tv_deblurring_problem(
    blur_sigma: float,
    noise_std: float,
    random_seed: int,
    face_index: int = FACE_INDEX,
) -> dict:
    """Load an Olivetti face (scaled by LP_UPPER), blur it, optionally add noise."""
    faces = fetch_olivetti_faces()
    image = faces.images[face_index].astype(np.float64) * LP_UPPER
    M, N = image.shape

    K_blur = _build_blur_matrix(M, N, blur_sigma, BLUR_HALF_WIDTH)
    p_true_flat = image.reshape(-1).copy()
    b = K_blur @ p_true_flat
    if noise_std > 0:
        rng = np.random.default_rng(random_seed)
        b = b + noise_std * rng.standard_normal(b.shape[0])

    return {
        "image": image,
        "K_blur": K_blur,
        "b": b,
        "blur_sigma": blur_sigma,
        "noise_std": noise_std,
        "M": M,
        "N": N,
        "p_true_flat": p_true_flat,
    }


def extract_constraint_matrices(
    K_blur: sp.csr_matrix,
    b: np.ndarray,
    M: int,
    N: int,
) -> TVDeblurringMatrices:
    """Build (c, A, b_eq, G, h, l, u) for the L1-TV deblurring LP.

    Variable layout x = [p; v; w; s] of length n_vars = 2K + K_v + K_h. A is a
    0-row matrix (no equality constraints). G stacks four TV-epigraph blocks
    and two L1-data-fit blocks.
    """
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = 2 * K + K_v + K_h

    c = np.concatenate([
        np.zeros(K),
        np.ones(K_v),
        np.ones(K_h),
        DATA_FIT_LAMBDA * np.ones(K),
    ])
    l = np.zeros(n_vars)
    u = LP_UPPER * np.ones(n_vars)

    A = sp.csr_matrix((0, n_vars))
    b_eq = np.zeros(0)

    D_v = _vertical_diff_matrix(M, N)
    D_h = _horizontal_diff_matrix(M, N)
    I_v = sp.eye(K_v, format="csr")
    I_h = sp.eye(K_h, format="csr")
    I_K = sp.eye(K, format="csr")

    # Six-block inequality stack:
    # [-D_v   I_v   0     0  ]   v >= -D_v p   (rhs 0)
    # [ D_v   I_v   0     0  ]   v >= +D_v p   (rhs 0)
    # [-D_h   0     I_h   0  ]   w >= -D_h p   (rhs 0)
    # [ D_h   0     I_h   0  ]   w >= +D_h p   (rhs 0)
    # [-K_b   0     0     I_K]   s >= -(K_b p - b)   (rhs -b)
    # [ K_b   0     0     I_K]   s >=  (K_b p - b)   (rhs +b)
    G = sp.bmat(
        [
            [-D_v,    I_v,  None, None],
            [ D_v,    I_v,  None, None],
            [-D_h,    None, I_h,  None],
            [ D_h,    None, I_h,  None],
            [-K_blur, None, None, I_K ],
            [ K_blur, None, None, I_K ],
        ],
        format="csr",
    )
    b_arr = np.asarray(b, dtype=np.float64)
    h = np.concatenate([
        np.zeros(2 * K_v + 2 * K_h),
        -b_arr,
        b_arr,
    ])

    return TVDeblurringMatrices(
        c=c, A=A, b=b_eq, G=G, h=h, l=l, u=u, K=K, K_v=K_v, K_h=K_h,
    )


def make_matrix_extractor(K_blur: sp.csr_matrix, M: int, N: int):
    """Closure with fixed K_blur + dimensions; only b varies."""
    return partial(
        extract_constraint_matrices,
        K_blur=K_blur,
        M=M,
        N=N,
    )


def solve_lp(matrices: TVDeblurringMatrices) -> dict:
    """Solve the LP via CVXPY in the user's standard form.

    Branches on `matrices.A.shape[0] == 0`: skips the equality constraint when
    the LP has none (deblurring case), otherwise mirrors the inpainting recipe.
    """
    import cvxpy as cp

    n_vars = matrices.c.shape[0]
    x = cp.Variable(n_vars)

    constraints = [
        matrices.G @ x >= matrices.h,
        x >= matrices.l,
        x <= matrices.u,
    ]
    has_eq = matrices.A.shape[0] > 0
    if has_eq:
        constraints.insert(0, matrices.A @ x == matrices.b)

    prob = cp.Problem(cp.Minimize(matrices.c @ x), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")

    raw_x = np.asarray(x.value)
    K = matrices.K
    K_v = matrices.K_v
    K_h = matrices.K_h

    # Dual recovery (matches PDHG convention y = [y_G; y_A], y_G >= 0):
    #   CVXPY mu_eq for (Ax == b),  mu_ineq for (Gx >= h),  with mu_ineq >= 0.
    #   PDHG sign convention: y_G = mu_ineq,  y_A = -mu_eq.
    if has_eq:
        mu_eq = np.asarray(constraints[0].dual_value)
        mu_ineq = np.asarray(constraints[1].dual_value)
        raw_y = np.concatenate([mu_ineq, -mu_eq])
    else:
        mu_ineq = np.asarray(constraints[0].dual_value)
        raw_y = mu_ineq

    return {
        "objective_value": prob.value,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "p": raw_x[:K],
        "vw": raw_x[K:K + K_v + K_h],
        "s": raw_x[K + K_v + K_h:],
    }


def verify_matrices(problem: dict, matrices: TVDeblurringMatrices) -> bool:
    """Compare the LP-form objective to a direct cp.norm1+cp.diff formulation."""
    import cvxpy as cp

    M, N = problem["M"], problem["N"]
    K_blur = problem["K_blur"]
    b = np.asarray(problem["b"], dtype=np.float64)

    sol_matrices = solve_lp(matrices)

    P = cp.Variable((M, N))
    # cp.vec is column-major; cp.vec(P.T) is the row-major flatten of P that
    # matches numpy's p.reshape(-1) (which K_blur was built against).
    p_flat = cp.vec(P.T)
    data_term = cp.norm1(K_blur @ p_flat - b)
    tv_term = (
        cp.sum(cp.abs(cp.diff(P, axis=0)))
        + cp.sum(cp.abs(cp.diff(P, axis=1)))
    )
    obj = DATA_FIT_LAMBDA * data_term + tv_term
    cons = [P >= 0, P <= LP_UPPER]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    p_matrix = sol_matrices["p"].reshape(M, N)
    p_ref = np.asarray(P.value)
    sup_diff = float(np.max(np.abs(p_matrix - p_ref)))

    print(f"Matrix formulation objective: {sol_matrices['objective_value']:.6f}")
    print(f"Direct CVXPY objective:       {prob.value:.6f}")
    print(f"Difference (objective):       {abs(sol_matrices['objective_value'] - prob.value):.2e}")
    print(f"Difference (||p||_inf):       {sup_diff:.2e}")

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
    """Chambolle-Pock PDHG with per-iteration stepsizes (K_max = len(tau_arr))."""
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
    blur_sigma: float,
    noise_std: float,
    random_seed: int,
    face_index: int = FACE_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a related TV deblurring LP and return its optimal (x, y)."""
    problem = generate_tv_deblurring_problem(
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        random_seed=random_seed,
        face_index=face_index,
    )
    extractor = make_matrix_extractor(problem["K_blur"], problem["M"], problem["N"])
    matrices = extractor(b=problem["b"])
    sol = solve_lp(matrices)
    return sol["raw_x"], sol["raw_y"]


if __name__ == "__main__":
    print("=" * 60)
    print("Generating TV Deblurring Problem (Olivetti face, 1-norm TV + L1 fit)")
    print("=" * 60)

    problem = generate_tv_deblurring_problem(
        blur_sigma=BLUR_SIGMA,
        noise_std=NOISE_STD,
        random_seed=NOISE_SEED,
        face_index=FACE_INDEX,
    )

    M, N = problem["M"], problem["N"]
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = 2 * K + K_v + K_h

    print(f"Image dimensions:    M={M}, N={N}")
    print(f"K (pixels):          {K}")
    print(f"K_v (vert diffs):    {K_v}")
    print(f"K_h (horz diffs):    {K_h}")
    print(f"n_vars:              {n_vars}")
    print(f"BLUR_SIGMA:          {BLUR_SIGMA}")
    print(f"DATA_FIT_LAMBDA:     {DATA_FIT_LAMBDA}")
    print(f"K_blur shape:        {problem['K_blur'].shape}, nnz={problem['K_blur'].nnz}")
    print(f"Inequality rows:     {2 * K_v + 2 * K_h + 2 * K}")

    print("\n" + "=" * 60)
    print("Extracting Sparse Constraint Matrices")
    print("=" * 60)

    extractor = make_matrix_extractor(problem["K_blur"], M, N)
    matrices = extractor(b=problem["b"])

    print(f"c shape: {matrices.c.shape}")
    print(f"A shape: {matrices.A.shape}, nnz: {matrices.A.nnz}, type: {type(matrices.A).__name__}")
    print(f"b shape: {matrices.b.shape}")
    print(f"G shape: {matrices.G.shape}, nnz: {matrices.G.nnz}, type: {type(matrices.G).__name__}")
    print(f"h shape: {matrices.h.shape}")
    print(f"l shape: {matrices.l.shape}, u shape: {matrices.u.shape}")

    assert sp.issparse(matrices.A)
    assert sp.issparse(matrices.G)

    # Sanity check: blur kernel rows sum to 1, and noiseless b matches K_blur @ p_true.
    row_sums = np.asarray(problem["K_blur"].sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-10), (
        f"Blur kernel rows do not sum to 1 (max dev {np.max(np.abs(row_sums - 1.0))})"
    )
    if NOISE_STD == 0.0:
        b_check = problem["K_blur"] @ problem["p_true_flat"]
        assert np.allclose(problem["b"], b_check), "b inconsistent with K_blur @ p_true"

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
    # this module's sibling at top level. Also prepend src/ to sys.path so direct
    # invocation (`python learning/tv_deblurring_test.py` from src/) resolves
    # the sibling package, not just `python -m ...`.
    import os, sys
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from learning_experiment_classes.pdlp import build_strict_interior_init

    m1 = matrices.G.shape[0]
    # S=0: deblurring LP has no equality constraints, so y0 reduces to 0.1 * 1_{m1}.
    x0, y0 = build_strict_interior_init(n_vars, m1, S=0, lp_upper=LP_UPPER)

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
    print("Generating second-row problem (same face, larger blur sigma = OOD)")
    print("=" * 60)

    problem2 = generate_tv_deblurring_problem(
        blur_sigma=BLUR_SIGMA_OOD,
        noise_std=NOISE_STD_OOD,
        random_seed=NOISE_SEED + 1,
        face_index=FACE_INDEX,
    )
    extractor2 = make_matrix_extractor(problem2["K_blur"], M, N)
    matrices2 = extractor2(b=problem2["b"])
    solution2 = solve_lp(matrices2)
    print(f"[row 2] sigma={BLUR_SIGMA_OOD}, optimal objective: {solution2['objective_value']:.4f}")

    m1_2 = matrices2.G.shape[0]
    x0_2, y0_2 = build_strict_interior_init(n_vars, m1_2, S=0, lp_upper=LP_UPPER)

    learned_iterates2: list[tuple[str, np.ndarray]] = []
    for label, subfolder in schedule_specs:
        s_path = os.path.join(
            stepsize_root, subfolder, f"learned_pdhg_stepsizes_K{K_MAX}.csv",
        )
        if not os.path.exists(s_path):
            continue
        arr = np.loadtxt(s_path, delimiter=",", skiprows=1)
        tau_arr = np.tile(arr[:, 0], NUM_REPS)
        sigma_arr = np.tile(arr[:, 1], NUM_REPS)
        theta_arr = np.tile(arr[:, 2], NUM_REPS)

        xk_learned2, _ = run_PDHG_with_stepsizes(
            matrices2.c,
            matrices2.G,
            matrices2.h,
            matrices2.A,
            matrices2.b,
            matrices2.l,
            matrices2.u,
            solution2["raw_x"],
            solution2["raw_y"],
            x0_2,
            y0_2,
            tau_arr,
            sigma_arr,
            theta_arr,
        )
        learned_iterates2.append((label, xk_learned2[:K].reshape(M, N)))

    print("\n" + "=" * 60)
    print("Operator norms and warm-start radii  R = ||[x0-x*; y0-y*]||")
    print("=" * 60)

    K_mat_row1 = sp.vstack([matrices.G, matrices.A], format="csr")
    K_mat_row2 = sp.vstack([matrices2.G, matrices2.A], format="csr")
    _, s1, _ = spla.svds(K_mat_row1, k=1, which="LM")
    _, s2, _ = spla.svds(K_mat_row2, k=1, which="LM")
    M_norm1 = float(s1[0])
    M_norm2 = float(s2[0])

    R_row1 = float(np.linalg.norm(
        np.concatenate([x0 - solution["raw_x"], y0 - solution["raw_y"]])
    ))
    R_row2 = float(np.linalg.norm(
        np.concatenate([x0_2 - solution2["raw_x"], y0_2 - solution2["raw_y"]])
    ))

    print(f"[row 1] sigma={BLUR_SIGMA:<4}  ||K_mat||_2 = {M_norm1:.4f}  R = {R_row1:.4f}")
    print(f"[row 2] sigma={BLUR_SIGMA_OOD:<4}  ||K_mat||_2 = {M_norm2:.4f}  R = {R_row2:.4f}")
    print(f"R_row2 / R_row1 = {R_row2 / R_row1:.3f}")

    print("\n" + "=" * 60)
    print("Plotting original / blurred / LP reconstruction / learned PDHG")
    print("=" * 60)

    import matplotlib.pyplot as plt

    original = problem["image"]
    blurred = problem["b"].reshape(M, N)
    reconstructed = solution["raw_x"][:K].reshape(M, N)

    blurred2 = problem2["b"].reshape(M, N)
    reconstructed2 = solution2["raw_x"][:K].reshape(M, N)

    horizon_label = (
        f"K={K_MAX}" if NUM_REPS == 1 else f"K={K_MAX}x{NUM_REPS}"
    )

    titles_row1 = [
        "Original",
        f"Blurred (sigma={BLUR_SIGMA})",
        "L1-TV Reconstruction (LP)",
    ]
    panels_row1 = [original, blurred, reconstructed]
    for label, img in learned_iterates:
        titles_row1.append(f"learned PDHG ({label}), {horizon_label}")
        panels_row1.append(img)

    titles_row2 = [
        None,
        f"Blurred (sigma={BLUR_SIGMA_OOD}, OOD)",
        "L1-TV Reconstruction (LP)",
    ]
    panels_row2 = [None, blurred2, reconstructed2]
    for label, img in learned_iterates2:
        titles_row2.append(f"learned PDHG ({label}), {horizon_label}")
        panels_row2.append(img)

    n_panels = max(len(panels_row1), len(panels_row2))
    panels_row1 += [None] * (n_panels - len(panels_row1))
    titles_row1 += [None] * (n_panels - len(titles_row1))
    panels_row2 += [None] * (n_panels - len(panels_row2))
    titles_row2 += [None] * (n_panels - len(titles_row2))

    fig, axes = plt.subplots(2, n_panels, figsize=(3.0 * n_panels, 6.4))
    plot_scale = 255.0 / LP_UPPER
    for row_axes, row_panels, row_titles in (
        (axes[0], panels_row1, titles_row1),
        (axes[1], panels_row2, titles_row2),
    ):
        for ax, img, title in zip(row_axes, row_panels, row_titles):
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(plot_scale * img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()
