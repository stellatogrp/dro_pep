"""
tv_averages.py — Average Lagrangian gap trajectories per learned schedule.

Two splits are evaluated:
  in  : every Olivetti face at the fixed MISSING_FRACTION (shared mask seed).
  ood : the 7 grayscale test images stored in test_images_64.npy, all at the
        same MISSING_FRACTION but each with its own mask seed.

For each split the script:
  1. Builds the TV inpainting LP for every (image, fraction, mask) instance.
  2. Runs the learned PDHG schedules. ldro-pep and lpep use the single
     K=K_MAX trajectory; l2o instead pulls a separate K-specific schedule
     for each k in 1..K_MAX, runs it for k iterations, and reports the
     final-iterate gap at index k (so the curve at index k always reflects
     a schedule that was actually trained for k steps).
  3. Captures the Lagrangian gap at each iteration index 0..K_MAX.
  4. Reports mean / q10 / q90 trajectories, writes a CSV, and renders a PDF.
"""

import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_olivetti_faces
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
})

SCHEDULE_DISPLAY_NAMES = {"l2o": "L2O", "ldro-pep": "DR-L2O", "lpep": "OPT-PEP"}
SCHEDULE_COLORS = {
    "l2o": "tab:blue",
    "ldro-pep": "tab:orange",
    "lpep": "tab:green",
}

# Labels whose curve is built from per-K schedules (one CSV per k, where the
# point at index k uses the schedule trained for k iterations). All other
# labels use the single K=K_MAX trajectory.
PER_K_LABELS = {"l2o"}

# OOD set: 7 downsampled grayscale test images. PIXEL_SIZE selects which
# square-side .npy stack to load (test_images_<PIXEL_SIZE>.npy). Each image is
# corrupted at the same MISSING_FRACTION as the in-distribution split, but
# with its own random mask (different seed per image).
PIXEL_SIZE = 128
TEST_IMAGES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"test_images_{PIXEL_SIZE}.npy",
)
OOD_MASK_SEED_BASE = 1000

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
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    tau_arr = np.tile(arr[:, 0], NUM_REPS)
    sigma_arr = np.tile(arr[:, 1], NUM_REPS)
    theta_arr = np.tile(arr[:, 2], NUM_REPS)
    return tau_arr, sigma_arr, theta_arr


def load_per_k_schedules(subfolder: str, K_max: int) -> dict:
    """Load one CSV per k in 1..K_max from `subfolder`; return {k: (tau, sigma, theta)}."""
    stepsize_root = os.path.join(
        _SRC_DIR, "learning_experiment_classes", "pdhg_stepsizes",
    )
    out = {}
    for k in range(1, K_max + 1):
        s_path = os.path.join(
            stepsize_root, subfolder, f"learned_pdhg_stepsizes_K{k}.csv"
        )
        if not os.path.exists(s_path):
            raise FileNotFoundError(f"No K={k} schedule CSV at {s_path}")
        arr = np.loadtxt(s_path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != k:
            raise ValueError(
                f"Expected {k} rows in {s_path}, got {arr.shape[0]}"
            )
        out[k] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def _run_schedules(matrices, raw_x, raw_y, x0, y0, schedules: dict) -> dict:
    """Run each schedule on the given LP. Trajectory schedules (tuples) capture
    every iterate's gap; per-K schedules (dicts of {k: schedule}) capture only
    the final-iterate gap for each k.
    """
    out = {}
    for label, sched in schedules.items():
        if isinstance(sched, dict):
            K_max_local = max(sched.keys())
            gaps = np.empty(K_max_local + 1, dtype=np.float64)
            for k_idx in sorted(sched.keys()):
                tau_arr, sigma_arr, theta_arr = sched[k_idx]
                full = run_pdhg_capture_gaps(
                    matrices.c, matrices.G, matrices.h,
                    matrices.A, matrices.b,
                    matrices.l, matrices.u,
                    raw_x, raw_y, x0, y0,
                    tau_arr, sigma_arr, theta_arr,
                )
                if k_idx == 1:
                    gaps[0] = full[0]
                gaps[k_idx] = full[-1]
            out[label] = gaps
        else:
            tau_arr, sigma_arr, theta_arr = sched
            out[label] = run_pdhg_capture_gaps(
                matrices.c, matrices.G, matrices.h,
                matrices.A, matrices.b,
                matrices.l, matrices.u,
                raw_x, raw_y, x0, y0,
                tau_arr, sigma_arr, theta_arr,
            )
    return out


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

    return _run_schedules(
        matrices, solution["raw_x"], solution["raw_y"], x0, y0, schedules,
    )


def evaluate_image_from_array(
    image: np.ndarray,
    missing_fraction: float,
    random_seed: int,
    schedules: dict,
) -> dict:
    """Same as ``evaluate_image`` but on an arbitrary grayscale image array.

    The image is assumed to lie in [0, 1] and is rescaled by LP_UPPER, matching
    the in-house ``generate_tv_inpainting_problem`` path. A random Bernoulli
    mask at the given missing_fraction is drawn from the supplied seed.
    """
    img = np.asarray(image, dtype=np.float64) * LP_UPPER
    M, N = img.shape
    rng = np.random.default_rng(random_seed)
    mask = rng.random((M, N)) >= missing_fraction
    known_indices = np.flatnonzero(mask)
    known_values = img.reshape(-1)[known_indices].copy()

    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    n_vars = K + K_v + K_h
    S = len(known_indices)

    extractor = make_matrix_extractor(known_indices, M, N)
    matrices = extractor(known_values=known_values)
    solution = solve_lp(matrices)
    m1 = matrices.G.shape[0]
    x0, y0 = build_strict_interior_init(n_vars, m1, S, LP_UPPER)

    return _run_schedules(
        matrices, solution["raw_x"], solution["raw_y"], x0, y0, schedules,
    )


if __name__ == "__main__":
    schedule_specs = ["l2o", "ldro-pep", "lpep"]
    if any(label in PER_K_LABELS for label in schedule_specs) and NUM_REPS != 1:
        raise ValueError(
            f"per-K schedules require NUM_REPS == 1 (got {NUM_REPS}); "
            "the per-K curve has length K_MAX+1, while NUM_REPS>1 would "
            "produce a length-K_MAX*NUM_REPS+1 trajectory for the others."
        )
    schedules = {
        label: (
            load_per_k_schedules(label, K_MAX)
            if label in PER_K_LABELS
            else load_schedule(label, K_MAX)
        )
        for label in schedule_specs
    }

    faces = fetch_olivetti_faces()
    n_faces = len(faces.images)

    # In-distribution split: every Olivetti face at the fixed MISSING_FRACTION,
    # all sharing seed 42 to keep the original mask pattern.
    in_items = [
        (faces.images[i].astype(np.float64), MISSING_FRACTION, 42)
        for i in range(n_faces)
    ]

    # OOD split: 7 downsampled test images, all corrupted at MISSING_FRACTION,
    # each with its own random mask (distinct seed per image).
    test_images = np.load(TEST_IMAGES_PATH)
    ood_items = [
        (test_images[i].astype(np.float64), MISSING_FRACTION, OOD_MASK_SEED_BASE + i)
        for i in range(test_images.shape[0])
    ]

    splits = [
        ("in", in_items),
        ("ood", ood_items),
    ]
    K_total = K_MAX * NUM_REPS

    print(
        f"Running in={len(in_items)} (Olivetti) + ood={len(ood_items)} (test images) "
        f"x {len(schedule_specs)} schedules; K_total={K_total} "
        f"(K_MAX={K_MAX}, NUM_REPS={NUM_REPS})"
    )
    print(
        f"MISSING_FRACTION={MISSING_FRACTION}, "
        f"MISSING_FRACTION_OOD={MISSING_FRACTION_OOD}"
    )

    results = {
        split_name: {label: [] for label in schedule_specs}
        for split_name, _ in splits
    }

    for split_name, items in splits:
        for image, frac, seed in tqdm(items, desc=f"{split_name} images"):
            out = evaluate_image_from_array(image, frac, seed, schedules)
            for label in schedule_specs:
                results[split_name][label].append(out[label])

    output_dir = os.path.join(
        _SRC_DIR, "learning_experiment_classes", "pdhg_stepsizes"
    )
    os.makedirs(output_dir, exist_ok=True)
    stat_names = ("mean", "q10", "q90")

    print("\n" + "=" * 60)
    print("Lagrangian gap stats (mean, q10, q90)")
    print("=" * 60)

    all_stats = {}
    for split_name, items in splits:
        fractions = np.array([f for _, f, _ in items], dtype=np.float64)
        if np.allclose(fractions, fractions[0]):
            frac_label = f"missing_fraction = {fractions[0]:.2f}"
        else:
            frac_label = (
                f"missing_fractions in [{fractions.min():.2f}, "
                f"{fractions.max():.2f}]"
            )
        print(f"\n{frac_label}  ({split_name}, n={len(items)})")

        stats = {}
        for label in schedule_specs:
            arr = np.asarray(results[split_name][label])
            stats[label] = {
                "mean": arr.mean(axis=0),
                "q10": np.quantile(arr, 0.10, axis=0),
                "q90": np.quantile(arr, 0.90, axis=0),
            }
        all_stats[split_name] = stats

        cols = [f"{label}_{s}" for label in schedule_specs for s in stat_names]
        header = "  iter | " + " | ".join(f"{c:>22s}" for c in cols)
        print(header)
        print("  -----+-" + "-+-".join("-" * 22 for _ in cols))
        for k in range(K_total + 1):
            row_vals = [
                f"{stats[label][s][k]:>22.6e}"
                for label in schedule_specs
                for s in stat_names
            ]
            print(f"  {k:>4d} | " + " | ".join(row_vals))

        data = np.zeros((K_total + 1, 1 + len(cols)))
        data[:, 0] = np.arange(K_total + 1)
        for j, label in enumerate(schedule_specs):
            for s_idx, s in enumerate(stat_names):
                data[:, 1 + len(stat_names) * j + s_idx] = stats[label][s]
        csv_path = os.path.join(
            output_dir,
            f"tv_average_gap_{split_name}_K{K_MAX}_reps{NUM_REPS}.csv",
        )
        np.savetxt(
            csv_path, data,
            delimiter=",",
            header="iter," + ",".join(cols),
            comments="",
        )
        print(f"  saved CSV:  {csv_path}")

        if "l2o" in schedule_specs and "ldro-pep" in schedule_specs:
            l2o_final = np.asarray(results[split_name]["l2o"])[:, -1]
            ldro_final = np.asarray(results[split_name]["ldro-pep"])[:, -1]
            diff = l2o_final - ldro_final
            worst_idx = int(np.argmax(diff))
            print(
                f"  argmax(l2o - ldro-pep) at k={K_total}: "
                f"image_index={worst_idx}, "
                f"l2o={l2o_final[worst_idx]:.6e}, "
                f"ldro-pep={ldro_final[worst_idx]:.6e}, "
                f"diff={diff[worst_idx]:.6e}"
            )

    fig, axes = plt.subplots(1, len(splits), figsize=(12, 4.5))
    if len(splits) == 1:
        axes = [axes]
    ks = np.arange(K_total + 1)
    for col_idx, (split_name, items) in enumerate(splits):
        fractions = np.array([f for _, f, _ in items], dtype=np.float64)
        if np.allclose(fractions, fractions[0]):
            title = rf"missing fraction $= {fractions[0]:g}$"
        else:
            title = (
                rf"missing fraction $\in "
                rf"[{fractions.min():g}, {fractions.max():g}]$"
            )
        ax = axes[col_idx]
        for label in schedule_specs:
            color = SCHEDULE_COLORS.get(label)
            display = SCHEDULE_DISPLAY_NAMES.get(label, label)
            ax.plot(
                ks, all_stats[split_name][label]["mean"],
                marker="o", markersize=5,
                label=display, color=color,
            )
            ax.fill_between(
                ks,
                all_stats[split_name][label]["q10"],
                all_stats[split_name][label]["q90"],
                alpha=0.2, color=color,
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Lagrangian gap")
    fig.tight_layout()
    pdf_path = os.path.join(
        output_dir, f"tv_average_gap_K{K_MAX}_reps{NUM_REPS}.pdf"
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot: {pdf_path}")
