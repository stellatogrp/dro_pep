"""
tv_inpainting_color_averages.py — Average Lagrangian gap trajectories per
learned schedule, on the 20-image tiny-imagenet color sample.

Color-image analogue of ``tv_averages.py``:
  1. Pulls 20 RGB images (64x64x3) from ``zh-plus/tiny-imagenet``.
  2. For each image, draws a single Bernoulli mask at MISSING_FRACTION
     (shared across the three channels — corrupted pixels are blacked out
     to (0, 0, 0)).
  3. Builds the L1, anisotropic TV-inpainting LP as a block-diagonal stack
     of three grayscale LPs (one per RGB channel; same mask, channel-specific
     RHS).
  4. Solves the LP with Clarabel to recover the saddle point (raw_x, raw_y)
     used as the anchor for the Lagrangian gap.
  5. Runs the three learned PDHG schedules (l2o, ldro-pep, lpep) and captures
     the Lagrangian gap at each iteration index 0..K_MAX.
  6. Reports mean / q10 / q90 across the 20 images, writes a CSV, and renders
     a single-Axes PDF (analogue of one panel in tv_averages.py).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from datasets import load_dataset
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
})

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from learning.tv_averages import (  # noqa: E402
    PER_K_LABELS,
    SCHEDULE_COLORS,
    SCHEDULE_DISPLAY_NAMES,
    _run_schedules,
    load_per_k_schedules,
    load_schedule,
)
from learning.tv_inpainting_test import (  # noqa: E402
    K_MAX,
    LP_UPPER,
    MISSING_FRACTION,
    NUM_REPS,
    TVInpaintingMatrices,
    extract_constraint_matrices,
    solve_lp,
)
from learning_experiment_classes.pdlp import build_strict_interior_init  # noqa: E402

N_IMAGES = 40
MASK_SEED = 42
IMAGE_SAMPLE_SEED = 0

# Schedule order for the row plot and the frac-solved curves.
PLOT_ORDER = ["lpep", "l2o", "ldro-pep"]

# Relative thresholds for `pdlp_frac_problems_solved`: an instance counts as
# solved at level eta iff lagrangian_gap(x_K, y_K) <= eta * (1 + f^*), where
# f^* is the LP optimum c^T x^* on that instance. Mirrors the lasso plot in
# experiment_plots_icml/lasso/create_paper_plots.py.
ETA_VALS_FRAC = [1e-2, 5e-2, 1e-1]


def load_color_images(
    n: int = N_IMAGES, seed: int = IMAGE_SAMPLE_SEED,
) -> np.ndarray:
    """Pull a random sample of ``n`` images from ``zh-plus/tiny-imagenet`` train.

    Selection uses ``np.random.default_rng(seed).choice`` (without replacement)
    over the full split, so changing ``seed`` reshuffles which 20 images are
    drawn. Returns a uint8 array of shape (n, 64, 64, 3). Grayscale entries
    (rare in tiny-imagenet) are broadcast to 3 channels.
    """
    dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(len(dataset), size=n, replace=False)
    ).tolist()
    subset = dataset.select(indices)
    images = []
    for item in subset:
        arr = np.array(item["image"])
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        images.append(arr)
    return np.stack(images).astype(np.uint8)


def make_image_gallery_figure(
    images: np.ndarray, ncols: int = 8,
) -> "plt.Figure":
    """Grid of every sampled image, each panel titled with its array index."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.6 * ncols, 1.7 * nrows),
    )
    axes = np.atleast_2d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(str(i), fontsize=10)
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def build_color_lp(
    image_uint8: np.ndarray,
    missing_fraction: float,
    seed: int,
) -> TVInpaintingMatrices:
    """Block-diagonal TV-inpainting LP for an RGB image.

    Variable layout: x = [p_R, v_R, w_R, p_G, v_G, w_G, p_B, v_B, w_B].
    The mask is shared across channels; only the equality RHS differs per
    channel. Image is rescaled by LP_UPPER / 255 so all variables live in
    [0, LP_UPPER] (i.e. [0, 1] when SCALED_LP_01 is True).
    """
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError(
            f"expected (M, N, 3) image, got shape {image_uint8.shape}"
        )
    img = image_uint8.astype(np.float64) * (LP_UPPER / 255.0)
    M, N, _ = img.shape

    rng = np.random.default_rng(seed)
    mask = rng.random((M, N)) >= missing_fraction
    known_indices = np.flatnonzero(mask)
    S = known_indices.size

    g = extract_constraint_matrices(
        known_indices=known_indices,
        known_values=np.zeros(S, dtype=np.float64),
        M=M,
        N=N,
    )

    b_channels = [
        img[:, :, ch].reshape(-1)[known_indices].copy()
        for ch in range(3)
    ]

    c = np.tile(g.c, 3)
    A = sp.block_diag([g.A, g.A, g.A], format="csr")
    b = np.concatenate(b_channels)
    G = sp.block_diag([g.G, g.G, g.G], format="csr")
    h = np.zeros(3 * g.h.size, dtype=np.float64)
    l = np.zeros(c.size, dtype=np.float64)
    u = LP_UPPER * np.ones(c.size, dtype=np.float64)

    return TVInpaintingMatrices(c=c, A=A, b=b, G=G, h=h, l=l, u=u)


def _extract_rgb_pixels(x: np.ndarray, M: int, N: int) -> np.ndarray:
    """Pull the (p_R, p_G, p_B) blocks out of x and stack into (M, N, 3)."""
    K = M * N
    K_v = (M - 1) * N
    K_h = M * (N - 1)
    block = K + K_v + K_h
    return np.stack(
        [x[ch * block : ch * block + K].reshape(M, N) for ch in range(3)],
        axis=-1,
    )


def _pdhg_final_iterate(
    matrices: TVInpaintingMatrices,
    x0: np.ndarray,
    y0: np.ndarray,
    tau_arr: np.ndarray,
    sigma_arr: np.ndarray,
    theta_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Silent variant of run_PDHG_with_stepsizes — returns final (x, y)."""
    K_mat = sp.vstack([matrices.G, matrices.A], format="csr")
    K_T = K_mat.T.tocsr()
    q = np.concatenate([matrices.h, matrices.b])
    m1 = matrices.G.shape[0]
    xk = np.asarray(x0, dtype=np.float64).copy()
    yk = np.asarray(y0, dtype=np.float64).copy()
    for k in range(len(tau_arr)):
        xkplus1 = np.minimum(
            matrices.u,
            np.maximum(xk - float(tau_arr[k]) * (matrices.c - K_T @ yk), matrices.l),
        )
        xbar = xkplus1 + float(theta_arr[k]) * (xkplus1 - xk)
        yk_new = yk + float(sigma_arr[k]) * (q - K_mat @ xbar)
        yk_new[:m1] = np.maximum(yk_new[:m1], 0.0)
        xk = xkplus1
        yk = yk_new
    return xk, yk


def evaluate_color_image(
    image_uint8: np.ndarray,
    missing_fraction: float,
    seed: int,
    schedules: dict,
) -> tuple[dict, float]:
    """Build → solve → run all schedules. Return (label → gap array, f^*)."""
    matrices = build_color_lp(image_uint8, missing_fraction, seed)
    solution = solve_lp(matrices)

    n_vars = matrices.c.size
    m1 = matrices.G.shape[0]
    S_total = matrices.A.shape[0]
    x0, y0 = build_strict_interior_init(n_vars, m1, S_total, LP_UPPER)

    gaps = _run_schedules(
        matrices, solution["raw_x"], solution["raw_y"], x0, y0, schedules,
    )
    return gaps, float(solution["objective_value"])


def aggregate_avg_stats(results: dict, schedule_specs: list) -> dict:
    """{label: {'mean','q10','q90'}} from per-image gap trajectories."""
    stats = {}
    for label in schedule_specs:
        arr = np.asarray(results[label])
        stats[label] = {
            "mean": arr.mean(axis=0),
            "q10": np.quantile(arr, 0.10, axis=0),
            "q90": np.quantile(arr, 0.90, axis=0),
        }
    return stats


def save_avg_stats_csv(
    stats: dict, schedule_specs: list, K_total: int, csv_path: str,
) -> None:
    """Wide-form: iter,<label>_mean,<label>_q10,<label>_q90,..."""
    stat_names = ("mean", "q10", "q90")
    cols = [f"{label}_{s}" for label in schedule_specs for s in stat_names]
    data = np.zeros((K_total + 1, 1 + len(cols)))
    data[:, 0] = np.arange(K_total + 1)
    for j, label in enumerate(schedule_specs):
        for s_idx, s in enumerate(stat_names):
            data[:, 1 + len(stat_names) * j + s_idx] = stats[label][s]
    np.savetxt(
        csv_path, data,
        delimiter=",",
        header="iter," + ",".join(cols),
        comments="",
    )


def load_avg_stats_csv(csv_path: str, schedule_specs: list) -> dict:
    """Inverse of ``save_avg_stats_csv``."""
    df = pd.read_csv(csv_path)
    stats = {}
    for label in schedule_specs:
        stats[label] = {
            "mean": df[f"{label}_mean"].to_numpy(dtype=np.float64),
            "q10": df[f"{label}_q10"].to_numpy(dtype=np.float64),
            "q90": df[f"{label}_q90"].to_numpy(dtype=np.float64),
        }
    return stats


# Canonical key set inside the best-image NPZ. Hyphens aren't legal NPZ keys.
_LABEL_TO_NPZ_KEY = {"lpep": "lpep", "l2o": "l2o", "ldro-pep": "ldro_pep"}
_NPZ_KEY_TO_LABEL = {v: k for k, v in _LABEL_TO_NPZ_KEY.items()}


def save_best_image_npz(
    npz_path: str,
    best_idx: int,
    original: np.ndarray,
    corrupted: np.ndarray,
    lp_recon: np.ndarray,
    vanilla_recon: np.ndarray,
    learned_recons: dict,
) -> None:
    """All arrays should already be in [0, 1] for direct imshow."""
    payload = {
        "best_idx": np.array(int(best_idx)),
        "original": original.astype(np.float32),
        "corrupted": corrupted.astype(np.float32),
        "lp": lp_recon.astype(np.float32),
        "vanilla": vanilla_recon.astype(np.float32),
    }
    for label, arr in learned_recons.items():
        if label not in _LABEL_TO_NPZ_KEY:
            raise ValueError(f"unknown schedule label {label!r}")
        payload[_LABEL_TO_NPZ_KEY[label]] = arr.astype(np.float32)
    np.savez(npz_path, **payload)


def load_best_image_npz(npz_path: str) -> dict:
    """Returns dict with keys best_idx, original, corrupted, lp, vanilla, and
    one entry per learned schedule keyed by its display label (e.g. 'ldro-pep')."""
    z = np.load(npz_path)
    out = {
        "best_idx": int(z["best_idx"]),
        "original": z["original"],
        "corrupted": z["corrupted"],
        "lp": z["lp"],
        "vanilla": z["vanilla"],
        "learned": {},
    }
    for npz_key, label in _NPZ_KEY_TO_LABEL.items():
        if npz_key in z.files:
            out["learned"][label] = z[npz_key]
    return out


def compute_frac_solved(
    results: dict,
    f_opts: list,
    eta_vals: list,
    K_total: int,
) -> dict:
    """frac_data[eta][label] = list of (K, frac_solved) for K = 1..K_total."""
    f_opt_arr = np.asarray(f_opts, dtype=np.float64)
    scales = 1.0 + np.maximum(f_opt_arr, 0.0)
    out = {eta: {label: [] for label in results} for eta in eta_vals}
    for label, traj_list in results.items():
        gaps = np.asarray(traj_list, dtype=np.float64)
        for eta in eta_vals:
            thresh = eta * scales
            for K in range(1, K_total + 1):
                frac = float(np.mean(gaps[:, K] <= thresh))
                out[eta][label].append((K, frac))
    return out


def save_frac_solved_csv(frac_data: dict, csv_path: str) -> None:
    """Long-form: eta, K, schedule, frac_solved (matches the lasso convention)."""
    rows = []
    for eta, by_label in frac_data.items():
        for label, pairs in by_label.items():
            for K, frac in pairs:
                rows.append({
                    "eta": float(eta),
                    "K": int(K),
                    "schedule": label,
                    "frac_solved": float(frac),
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_frac_solved_csv(csv_path: str) -> dict:
    """Inverse of ``save_frac_solved_csv``: dict[eta][label] = [(K, frac), ...]."""
    df = pd.read_csv(csv_path)
    out = {}
    for eta_val in df["eta"].unique():
        sub = df[df["eta"] == eta_val]
        out[float(eta_val)] = {}
        for label in sub["schedule"].unique():
            sub_lab = sub[sub["schedule"] == label].sort_values("K")
            out[float(eta_val)][label] = list(
                zip(
                    sub_lab["K"].astype(int).tolist(),
                    sub_lab["frac_solved"].astype(float).tolist(),
                )
            )
    return out


def make_frac_solved_figure(frac_data: dict, plot_order: list) -> "plt.Figure":
    """1 x len(eta) panel: y = frac of color LPs hitting eta*(1+f^*) by step K."""
    eta_vals = sorted(frac_data.keys())
    n_eta = len(eta_vals)
    fig, axes = plt.subplots(1, n_eta, figsize=(5 * n_eta, 4.5), sharey=True)
    if n_eta == 1:
        axes = [axes]

    for col_idx, eta in enumerate(eta_vals):
        ax = axes[col_idx]
        for label in plot_order:
            pairs = frac_data[eta].get(label)
            if not pairs:
                continue
            Ks = [p[0] for p in pairs]
            fracs = [p[1] for p in pairs]
            ax.plot(
                Ks, fracs,
                marker="o", markersize=5,
                label=SCHEDULE_DISPLAY_NAMES.get(label, label),
                color=SCHEDULE_COLORS.get(label),
            )
        ax.set_title(rf"$\eta = {eta:g}$")
        ax.set_xlabel(r"$K$")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("Frac. problems solved")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-recompute",
        action="store_true",
        help="Invalidate every plot's cache and recompute all three plots "
             "from scratch (re-solves every LP + reruns every schedule).",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Render a labeled gallery of all sampled color images "
             "(one panel per index) and exit.",
    )
    args = parser.parse_args()

    if args.show_images:
        images = load_color_images(N_IMAGES, IMAGE_SAMPLE_SEED)
        gallery_pdf_path = os.path.join(
            _SRC_DIR,
            "learning_experiment_classes",
            "pdhg_stepsizes",
            "tv_color_image_gallery.pdf",
        )
        os.makedirs(os.path.dirname(gallery_pdf_path), exist_ok=True)
        fig = make_image_gallery_figure(images)
        fig.savefig(gallery_pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved image gallery: {gallery_pdf_path}")
        sys.exit(0)

    schedule_specs = ["l2o", "ldro-pep", "lpep"]
    if any(label in PER_K_LABELS for label in schedule_specs) and NUM_REPS != 1:
        raise ValueError(
            f"per-K schedules require NUM_REPS == 1 (got {NUM_REPS})."
        )
    schedules = {
        label: (
            load_per_k_schedules(label, K_MAX)
            if label in PER_K_LABELS
            else load_schedule(label, K_MAX)
        )
        for label in schedule_specs
    }

    if "ldro-pep" not in schedule_specs or len(schedule_specs) < 2:
        raise RuntimeError(
            "best-image selection requires ldro-pep plus at least one other schedule"
        )

    K_total = K_MAX * NUM_REPS
    output_dir = os.path.join(
        _SRC_DIR, "learning_experiment_classes", "pdhg_stepsizes"
    )
    os.makedirs(output_dir, exist_ok=True)

    avg_csv_path = os.path.join(
        output_dir,
        f"tv_color_average_gap_K{K_MAX}_reps{NUM_REPS}.csv",
    )
    avg_pdf_path = os.path.join(
        output_dir,
        f"tv_color_average_gap_K{K_MAX}_reps{NUM_REPS}.pdf",
    )
    best_npz_path = os.path.join(
        output_dir,
        f"tv_color_best_ldro_image_K{K_MAX}_reps{NUM_REPS}.npz",
    )
    best_pdf_path = os.path.join(
        output_dir,
        f"tv_color_best_ldro_image_K{K_MAX}_reps{NUM_REPS}.pdf",
    )
    frac_csv_path = os.path.join(
        output_dir,
        f"pdlp_frac_problems_solved_K{K_MAX}_reps{NUM_REPS}.csv",
    )
    frac_pdf_path = os.path.join(
        output_dir,
        f"pdlp_frac_problems_solved_K{K_MAX}_reps{NUM_REPS}.pdf",
    )

    avg_needs = args.recompute or not os.path.exists(avg_csv_path)
    best_needs = args.recompute or not os.path.exists(best_npz_path)
    frac_needs = args.recompute or not os.path.exists(frac_csv_path)

    # The gap-trajectory loop feeds plots 1, 2, and 3, so skip it iff every
    # plot's cache hits.
    needs_gap_loop = avg_needs or best_needs or frac_needs

    results = None
    f_opts = None
    images = None

    if needs_gap_loop:
        images = load_color_images(N_IMAGES)
        print(f"Loaded color images: shape={images.shape}, dtype={images.dtype}")
        print(
            f"Running {N_IMAGES} color images x {len(schedule_specs)} schedules; "
            f"K_total={K_total} (K_MAX={K_MAX}, NUM_REPS={NUM_REPS}); "
            f"MISSING_FRACTION={MISSING_FRACTION}"
        )
        results = {label: [] for label in schedule_specs}
        f_opts = []
        for img in tqdm(images, desc="color images"):
            gaps_dict, f_opt = evaluate_color_image(
                img, MISSING_FRACTION, MASK_SEED, schedules,
            )
            for label in schedule_specs:
                results[label].append(gaps_dict[label])
            f_opts.append(f_opt)
    else:
        print("All three plot caches present; skipping LP loop.")

    # ------------------------------------------------------------------
    # Plot 1: average Lagrangian gap (mean / q10 / q90 over the 20 images).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Plot 1: average Lagrangian gap")
    print("=" * 60)

    if avg_needs:
        stats = aggregate_avg_stats(results, schedule_specs)
        save_avg_stats_csv(stats, schedule_specs, K_total, avg_csv_path)
        print(f"  saved CSV:  {avg_csv_path}")
    else:
        print(f"  loading cache: {avg_csv_path}")
        stats = load_avg_stats_csv(avg_csv_path, schedule_specs)

    stat_names = ("mean", "q10", "q90")
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

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ks = np.arange(K_total + 1)
    for label in schedule_specs:
        color = SCHEDULE_COLORS.get(label)
        display = SCHEDULE_DISPLAY_NAMES.get(label, label)
        ax.plot(
            ks, stats[label]["mean"],
            marker="o", markersize=5,
            label=display, color=color,
        )
        ax.fill_between(
            ks, stats[label]["q10"], stats[label]["q90"],
            alpha=0.2, color=color,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Lagrangian gap")
    ax.set_title(rf"missing fraction $= {MISSING_FRACTION:g}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(avg_pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot: {avg_pdf_path}")

    # ------------------------------------------------------------------
    # Plot 2: row of reconstructions for the image where ldro-pep beats the
    # next-best schedule by the largest final-iteration gap margin.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Plot 2: best-ldro-pep image reconstructions")
    print("=" * 60)

    if best_needs:
        ldro_final = np.array(
            [results["ldro-pep"][i][-1] for i in range(N_IMAGES)]
        )
        other_finals = np.array(
            [
                [results[label][i][-1] for label in schedule_specs if label != "ldro-pep"]
                for i in range(N_IMAGES)
            ]
        )
        margins = other_finals.min(axis=1) - ldro_final
        best_idx = int(np.argmax(margins))
        print(f"  best image index : {best_idx}")
        print(f"  ldro-pep final gap : {ldro_final[best_idx]:.6e}")
        for j, label in enumerate(
            [lab for lab in schedule_specs if lab != "ldro-pep"]
        ):
            print(f"  {label:<8s} final gap : {other_finals[best_idx, j]:.6e}")
        print(f"  margin (next-best - ldro-pep): {margins[best_idx]:.6e}")

        best_img = images[best_idx]
        matrices0 = build_color_lp(best_img, MISSING_FRACTION, MASK_SEED)
        solution0 = solve_lp(matrices0)
        n_vars0 = matrices0.c.size
        m1_0 = matrices0.G.shape[0]
        S_total0 = matrices0.A.shape[0]
        x0_init, y0_init = build_strict_interior_init(
            n_vars0, m1_0, S_total0, LP_UPPER,
        )

        K_mat0 = sp.vstack([matrices0.G, matrices0.A], format="csr")
        _, s0, _ = spla.svds(K_mat0, k=1, which="LM")
        M_norm0 = float(s0[0])
        tau_v = (0.9 / M_norm0) * np.ones(K_total)
        sigma_v = (0.9 / M_norm0) * np.ones(K_total)
        theta_v = np.ones(K_total)
        xk_vanilla, _ = _pdhg_final_iterate(
            matrices0, x0_init, y0_init, tau_v, sigma_v, theta_v,
        )

        final_iters = {}
        for label in schedule_specs:
            sched = schedules[label]
            if isinstance(sched, dict):
                tau_arr, sigma_arr, theta_arr = sched[K_MAX]
            else:
                tau_arr, sigma_arr, theta_arr = sched
            xk, _ = _pdhg_final_iterate(
                matrices0, x0_init, y0_init, tau_arr, sigma_arr, theta_arr,
            )
            final_iters[label] = xk

        M0, N0 = best_img.shape[:2]
        rng0 = np.random.default_rng(MASK_SEED)
        mask0 = rng0.random((M0, N0)) >= MISSING_FRACTION

        original_disp = best_img.astype(np.float64) / 255.0
        corrupted_disp = np.where(mask0[:, :, None], original_disp, 0.0)
        recon_lp_disp = _extract_rgb_pixels(solution0["raw_x"], M0, N0) / LP_UPPER
        recon_van_disp = _extract_rgb_pixels(xk_vanilla, M0, N0) / LP_UPPER
        learned_disp = {
            label: _extract_rgb_pixels(final_iters[label], M0, N0) / LP_UPPER
            for label in schedule_specs
        }

        save_best_image_npz(
            best_npz_path, best_idx, original_disp, corrupted_disp,
            recon_lp_disp, recon_van_disp, learned_disp,
        )
        print(f"  saved cache: {best_npz_path}")
    else:
        print(f"  loading cache: {best_npz_path}")
        cache = load_best_image_npz(best_npz_path)
        best_idx = cache["best_idx"]
        original_disp = cache["original"]
        corrupted_disp = cache["corrupted"]
        recon_lp_disp = cache["lp"]
        recon_van_disp = cache["vanilla"]
        learned_disp = cache["learned"]

    horizon_label = f"K={K_MAX}" if NUM_REPS == 1 else f"K={K_MAX}x{NUM_REPS}"
    miss_pct = int(round(MISSING_FRACTION * 100))
    panels = [
        ("Original", original_disp),
        (f"Corrupted ({miss_pct}\\% missing)", corrupted_disp),
        ("L1-TV Reconstruction (LP)", recon_lp_disp),
        ("vanilla", recon_van_disp),
    ]
    for label in PLOT_ORDER:
        if label not in learned_disp:
            continue
        display = SCHEDULE_DISPLAY_NAMES.get(label, label)
        panels.append(
            (f"learned PDHG ({display}), {horizon_label}", learned_disp[label])
        )

    fig2, axes2 = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (title, img) in zip(axes2, panels):
        ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.tight_layout()
    fig2.savefig(best_pdf_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved plot (image {best_idx}): {best_pdf_path}")

    # ------------------------------------------------------------------
    # Plot 3: fraction of LPs hitting gap_K <= eta * (1 + f^*) per step K.
    # Mirrors the lasso plot in
    # experiment_plots_icml/lasso/create_paper_plots.py.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Plot 3: fraction of problems solved per eta")
    print("=" * 60)

    if frac_needs:
        frac_data = compute_frac_solved(
            results, f_opts, ETA_VALS_FRAC, K_total,
        )
        save_frac_solved_csv(frac_data, frac_csv_path)
        print(f"  saved CSV:  {frac_csv_path}")
    else:
        print(f"  loading cache: {frac_csv_path}")
        frac_data = load_frac_solved_csv(frac_csv_path)

    for eta in sorted(frac_data.keys()):
        print(f"  eta = {eta:g}")
        for label in PLOT_ORDER:
            pairs = frac_data[eta].get(label, [])
            if not pairs:
                continue
            tail = "  ".join(f"K={K}:{frac:.2f}" for K, frac in pairs[-3:])
            print(
                f"    {SCHEDULE_DISPLAY_NAMES.get(label, label):<8s} "
                f"final-three: {tail}"
            )

    fig3 = make_frac_solved_figure(frac_data, PLOT_ORDER)
    fig3.savefig(frac_pdf_path, bbox_inches="tight")
    plt.close(fig3)
    print(f"  saved plot: {frac_pdf_path}")
