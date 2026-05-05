"""
tv_averages.py — Average Lagrangian gap trajectories across all Olivetti faces.

For both missing-fraction values defined in tv_inpainting_test.py
(MISSING_FRACTION and MISSING_FRACTION_OOD), this script:
  1. Iterates over every Olivetti face (face_index = 0..399).
  2. Builds the TV inpainting LP with that face as ground truth.
  3. Runs the two learned PDHG schedules (l2o and ldro-pep) at K=K_MAX.
  4. Captures the Lagrangian gap at each iteration.
  5. Averages each gap trajectory across all images and prints the result.
"""

import os
import sys
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_olivetti_faces
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from learning.tv_inpainting_test import (  # noqa: E402
    K_MAX,
    LP_UPPER,
    MISSING_FRACTION,
    MISSING_FRACTION_OOD,
    NUM_REPS,
    generate_tv_inpainting_problem,
    make_matrix_extractor,
    solve_lp,
)
from learning_experiment_classes.pdlp import build_strict_interior_init  # noqa: E402


def run_pdhg_capture_gaps(
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
) -> np.ndarray:
    """PDHG with per-iteration stepsizes; returns gap array of length K_max+1.

    gaps[k] = lagrangian(x_k, raw_ys) - lagrangian(raw_xs, y_k). gaps[0] is
    the warm-start gap; gaps[K_max] is the gap after the final step.
    """
    K_mat = sp.vstack([G, A], format="csr")
    K_T = K_mat.T.tocsr()
    q = np.concatenate([h, b])
    m1 = G.shape[0]

    def satlin(v):
        return np.minimum(u, np.maximum(v, l))

    def partial_relu(v):
        out = v.copy()
        out[:m1] = np.maximum(out[:m1], 0.0)
        return out

    def lagrangian(x, y):
        return float(c @ x - y @ (K_mat @ x) + q @ y)

    def lagrangian_gap(xk, yk):
        return lagrangian(xk, raw_ys) - lagrangian(raw_xs, yk)

    xk = np.asarray(x0, dtype=np.float64).copy()
    yk = np.asarray(y0, dtype=np.float64).copy()
    K_max = len(tau_arr)
    assert len(sigma_arr) == K_max and len(theta_arr) == K_max

    gaps = np.empty(K_max + 1, dtype=np.float64)
    gaps[0] = lagrangian_gap(xk, yk)
    for k in range(K_max):
        tau_k = float(tau_arr[k])
        sigma_k = float(sigma_arr[k])
        theta_k = float(theta_arr[k])
        xkplus1 = satlin(xk - tau_k * (c - K_T @ yk))
        xbar = xkplus1 + theta_k * (xkplus1 - xk)
        ykplus1 = partial_relu(yk + sigma_k * (q - K_mat @ xbar))
        xk = xkplus1
        yk = ykplus1
        gaps[k + 1] = lagrangian_gap(xk, yk)
    return gaps


def load_schedule(subfolder: str, K_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stepsize_root = os.path.join(
        _SRC_DIR, "learning_experiment_classes", "pdhg_stepsizes",
    )
    s_path = os.path.join(stepsize_root, subfolder, f"learned_pdhg_stepsizes_K{K_max}.csv")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"No K={K_max} schedule CSV at {s_path}")
    arr = np.loadtxt(s_path, delimiter=",", skiprows=1)
    tau_arr = np.tile(arr[:, 0], NUM_REPS)
    sigma_arr = np.tile(arr[:, 1], NUM_REPS)
    theta_arr = np.tile(arr[:, 2], NUM_REPS)
    return tau_arr, sigma_arr, theta_arr


def evaluate_image(
    image_idx: int,
    missing_fraction: float,
    random_seed: int,
    schedules: dict,
) -> dict:
    """Run all schedules on one (image, mask) instance; return label -> gap array."""
    problem = generate_tv_inpainting_problem(
        missing_fraction=missing_fraction,
        random_seed=random_seed,
        face_index=image_idx,
    )
    M, N = problem["M"], problem["N"]
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = K + K_v + K_h
    S = len(problem["known_indices"])

    extractor = make_matrix_extractor(problem["known_indices"], M, N)
    matrices = extractor(known_values=problem["known_values"])
    solution = solve_lp(matrices)
    m1 = matrices.G.shape[0]
    x0, y0 = build_strict_interior_init(n_vars, m1, S, LP_UPPER)

    out = {}
    for label, (tau_arr, sigma_arr, theta_arr) in schedules.items():
        out[label] = run_pdhg_capture_gaps(
            matrices.c, matrices.G, matrices.h,
            matrices.A, matrices.b,
            matrices.l, matrices.u,
            solution["raw_x"], solution["raw_y"],
            x0, y0,
            tau_arr, sigma_arr, theta_arr,
        )
    return out


if __name__ == "__main__":
    schedule_specs = ["l2o", "ldro-pep"]
    schedules = {label: load_schedule(label, K_MAX) for label in schedule_specs}

    faces = fetch_olivetti_faces()
    n_images = len(faces.images)

    splits = [
        ("in",  MISSING_FRACTION,     42),
        ("ood", MISSING_FRACTION_OOD, 43),
    ]
    K_total = K_MAX * NUM_REPS

    print(
        f"Running {n_images} faces x {len(splits)} fractions x "
        f"{len(schedule_specs)} schedules; K_total={K_total} "
        f"(K_MAX={K_MAX}, NUM_REPS={NUM_REPS})"
    )
    print(
        f"MISSING_FRACTION={MISSING_FRACTION}, "
        f"MISSING_FRACTION_OOD={MISSING_FRACTION_OOD}"
    )

    results = {
        split_name: {label: [] for label in schedule_specs}
        for split_name, _, _ in splits
    }

    for idx in tqdm(range(n_images), desc="Images"):
        for split_name, frac, seed in splits:
            out = evaluate_image(idx, frac, seed, schedules)
            for label in schedule_specs:
                results[split_name][label].append(out[label])

    print("\n" + "=" * 60)
    print(f"Average Lagrangian gap loss over {n_images} Olivetti faces")
    print("=" * 60)

    for split_name, frac, _ in splits:
        print(f"\nmissing_fraction = {frac:.2f}  ({split_name})")
        header = "  iter | " + " | ".join(f"{label:>15s}" for label in schedule_specs)
        print(header)
        print("  -----+-" + "-+-".join("-" * 15 for _ in schedule_specs))

        means = {
            label: np.asarray(results[split_name][label]).mean(axis=0)
            for label in schedule_specs
        }
        for k in range(K_total + 1):
            row = " | ".join(f"{means[label][k]:>15.6e}" for label in schedule_specs)
            print(f"  {k:>4d} | {row}")

        if "l2o" in schedule_specs and "ldro-pep" in schedule_specs:
            l2o_final = np.asarray(results[split_name]["l2o"])[:, -1]
            ldro_final = np.asarray(results[split_name]["ldro-pep"])[:, -1]
            diff = l2o_final - ldro_final
            worst_idx = int(np.argmax(diff))
            print(
                f"  argmax(l2o - ldro-pep) at k={K_total}: "
                f"face_index={worst_idx}, "
                f"l2o={l2o_final[worst_idx]:.6e}, "
                f"ldro-pep={ldro_final[worst_idx]:.6e}, "
                f"diff={diff[worst_idx]:.6e}"
            )
