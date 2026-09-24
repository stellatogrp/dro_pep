"""
Paper plots for PDLP (TV inpainting) experiments.

Three paper-ready figures plus a LaTeX timings table, styled to match the
Quad / Lasso versions in ``../{quad,lasso}/create_paper_plots.py``:

``pdlp_losses.pdf``  (1 x 2 grid, mean + [q10, q90] band, log-y)
    Left:  in-distribution Lagrangian gap vs. K (every Olivetti face,
           ``MISSING_FRACTION = 0.1``, shared mask seed 42).
    Right: out-of-distribution Lagrangian gap vs. K (40 RGB tiny-imagenet
           images, ``MISSING_FRACTION = 0.1``, mask seed 42).

``pdlp_frac_problems_solved.pdf``  (2 x 3 grid)
    Rows: in-distribution / out-of-distribution.
    Cols: eta in [1e-2, 5e-2, 1e-1].
    y: fraction of LPs with gap_K <= eta * (1 + max(f^*, 0)).

``pdlp_times.csv``  (one row per (arch, K), printed LaTeX tabular)
    Mean +/- 2*sigma SGD iteration time (seconds) extracted from
    ``{subdir}_timings/K_{K}/progress.csv`` (where ``subdir`` follows
    ``ARCH_TO_SUBDIR``, e.g. ``ldro-pep_timings/``). Reports K in {1, 5, 10}.

``pdlp_reconstructions.pdf``  (2 x 6 grid, similar to ``tv_inpainting_test.py``)
    Top row:    the Olivetti face that maximizes
                ``l2o_final_gap - ldro_pep_final_gap`` (the face where
                DR-L2O beats L2O by the largest margin at iteration
                K_MAX). Mask seed 42, MISSING_FRACTION=0.1.
    Bottom row: the tiny-imagenet image (within the deterministic
                ``IMAGE_SAMPLE_SEED`` sample) that maximizes the same
                ``l2o - ldro_pep`` final-gap margin. Mask seed 42.
    Cols: Original, Corrupted, L1-TV LP, OPT-PEP, L2O, DR-L2O.

``pdlp_more_reconstructions.pdf``  (2 x 6 grid, tiny-imagenet only)
    Same six-column layout as ``pdlp_reconstructions.pdf``. Both rows are
    color images: rank-4 and rank-5 in the same ``l2o - ldro_pep``
    final-gap margin sort (rank-1 already appears in the bottom row of
    ``pdlp_reconstructions.pdf``).

For each architecture:
    * ``l2o`` uses *per-K* schedules — at column K we run the schedule
      trained for K iterations for K steps and report gap_K.
    * ``ldro_pep`` and ``lpep`` use a single K=K_MAX schedule, run once for
      K_MAX iterations; intermediate iterates feed every K column.

Stepsize CSVs live in this directory under ``{l2o, ldro-pep, lpep}/``.
Heavy compute is cached as NPZ arrays under ``paper_plots/``; downstream
CSVs (``pdlp_losses.csv``, ``pdlp_frac_problems_solved.csv``) drive the
figures and are regenerated on ``-recompute``.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "figure.figsize": (12, 5),
})

PDLP_DIR = Path(__file__).parent
PAPER_PLOTS_DIR = PDLP_DIR / "paper_plots"
# Consolidated layout: this file lives at iclr_data_outputs/archive/pdlp/,
# so src/ is three levels up (two in the original experiment_plots_icml/pdlp/).
SRC_DIR = PDLP_DIR.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learning.tv_averages import _run_schedules  # noqa: E402
from learning.tv_inpainting_color_averages import (  # noqa: E402
    MASK_SEED,
    N_IMAGES,
    _extract_rgb_pixels,
    _pdhg_final_iterate,
    aggregate_avg_stats,
    build_color_lp,
    compute_frac_solved,
    evaluate_color_image,
    load_color_images,
)
from learning.tv_inpainting_test import (  # noqa: E402
    K_MAX,
    LP_UPPER,
    MISSING_FRACTION,
    NUM_REPS,
    generate_tv_inpainting_problem,
    make_matrix_extractor,
    solve_lp,
)
from learning_experiment_classes.pdlp import build_strict_interior_init  # noqa: E402

# ==================== Configuration ====================

# Canonical underscore arch keys (match quad/lasso). Translate to hyphenated
# directory names with ARCH_TO_SUBDIR only when reading CSVs.
ARCHS = ["l2o", "ldro_pep", "lpep"]
PER_K_ARCHS = {"l2o"}
ARCH_TO_SUBDIR = {"l2o": "l2o", "ldro_pep": "ldro-pep", "lpep": "lpep"}

ARCH_DISPLAY_NAMES = {
    "l2o": "L2O",
    "ldro_pep": "DR-L2O",
    "lpep": "OPT-PEP",
}

ARCH_COLORS = {
    "l2o": "#DC3220",
    "ldro_pep": "#005AB5",
    "lpep": "#00B32D",
}

ARCH_MARKERS = {
    "l2o": "o",
    "ldro_pep": "s",
    "lpep": "^",
}

ROW_LABELS = ["test", "ood"]
ROW_TITLES = {
    "test": "In-dist.",
    "ood": "Out-of-dist.",
}

# Skipping k=0 to match the lasso convention (K_VALS_LOSSES = range(1, ...)).
K_VALS_LOSSES = list(range(1, K_MAX + 1))

ETA_VALS_FRAC = [1e-2, 5e-2, 1e-1]

# Even-integer x-ticks for K up to K_MAX.
X_TICKS = [k for k in (2, 4, 6, 8, 10, 12, 14) if k <= K_MAX]

# Column order for the learned reconstructions in pdlp_reconstructions.pdf.
RECON_ARCH_ORDER = ["lpep", "l2o", "ldro_pep"]

# Timings table. K subset and warmup match lasso/quad/times.py.
K_VALS_TIMES = [1, 5, 10]
WARMUP_ITERS = 5  # drop the first 5 SGD iters (JAX warmup)
FRAMEWORK_ORDER_TIMES = ["l2o", "ldro_pep", "lpep"]


# ==================== Schedule loaders ====================

def _stepsize_csv_path(arch: str, k: int) -> Path:
    return PDLP_DIR / ARCH_TO_SUBDIR[arch] / f"learned_pdhg_stepsizes_K{k}.csv"


def load_schedule_from_dir(arch: str, K_max: int):
    """Load a single ``learned_pdhg_stepsizes_K{K_max}.csv`` and return
    ``(tau, sigma, theta)`` arrays tiled by ``NUM_REPS`` (matches
    ``tv_averages.load_schedule``)."""
    path = _stepsize_csv_path(arch, K_max)
    if not path.exists():
        raise FileNotFoundError(f"No K={K_max} schedule CSV at {path}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    tau = np.tile(arr[:, 0], NUM_REPS)
    sigma = np.tile(arr[:, 1], NUM_REPS)
    theta = np.tile(arr[:, 2], NUM_REPS)
    return tau, sigma, theta


def load_per_k_schedules_from_dir(arch: str, K_max: int) -> dict:
    """Return ``{k: (tau, sigma, theta)}`` for k in 1..K_max from the per-K
    CSVs (matches ``tv_averages.load_per_k_schedules``)."""
    out = {}
    for k in range(1, K_max + 1):
        path = _stepsize_csv_path(arch, k)
        if not path.exists():
            raise FileNotFoundError(f"No K={k} schedule CSV at {path}")
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != k:
            raise ValueError(
                f"Expected {k} rows in {path}, got {arr.shape[0]}"
            )
        out[k] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def build_schedules_dict(K_max: int) -> dict:
    """Return ``{arch: schedule}`` keyed by underscore arch names. Per-K archs
    map to a dict; single-K archs map to a tuple."""
    if PER_K_ARCHS and NUM_REPS != 1:
        raise ValueError(
            f"per-K schedules require NUM_REPS == 1 (got {NUM_REPS})."
        )
    schedules = {}
    for arch in ARCHS:
        if arch in PER_K_ARCHS:
            schedules[arch] = load_per_k_schedules_from_dir(arch, K_max)
        else:
            schedules[arch] = load_schedule_from_dir(arch, K_max)
    return schedules


# ==================== Heavy compute ====================

def _evaluate_olivetti_face(
    face_index: int,
    missing_fraction: float,
    random_seed: int,
    schedules: dict,
):
    """Build LP for one Olivetti face, solve it, and run all schedules.

    Returns ``(gaps_dict, f_opt)`` — the same shape as
    ``tv_inpainting_color_averages.evaluate_color_image`` so both splits
    feed the same downstream pipeline.
    """
    problem = generate_tv_inpainting_problem(
        missing_fraction=missing_fraction,
        random_seed=random_seed,
        face_index=face_index,
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

    gaps = _run_schedules(
        matrices, solution["raw_x"], solution["raw_y"], x0, y0, schedules,
    )
    return gaps, float(solution["objective_value"])


def compute_in_split_gaps(schedules: dict):
    """Run every Olivetti face at MISSING_FRACTION (mask seed 42)."""
    from sklearn.datasets import fetch_olivetti_faces

    faces = fetch_olivetti_faces()
    n_faces = len(faces.images)
    K_total = K_MAX * NUM_REPS

    results = {arch: np.empty((n_faces, K_total + 1), dtype=np.float64)
               for arch in ARCHS}
    f_opts = np.empty(n_faces, dtype=np.float64)

    for i in tqdm(range(n_faces), desc="in (Olivetti)"):
        gaps, f_opt = _evaluate_olivetti_face(
            face_index=i,
            missing_fraction=MISSING_FRACTION,
            random_seed=42,
            schedules=schedules,
        )
        for arch in ARCHS:
            results[arch][i] = gaps[arch]
        f_opts[i] = f_opt

    return results, f_opts


_COLOR_IMAGES_CACHE: np.ndarray | None = None


def _get_color_images() -> np.ndarray:
    """Memoized ``load_color_images(N_IMAGES)``. Both the OOD gap loop and the
    reconstructions step pull from the same deterministic batch, so we pay
    the HF-dataset materialization at most once per process."""
    global _COLOR_IMAGES_CACHE
    if _COLOR_IMAGES_CACHE is None:
        _COLOR_IMAGES_CACHE = load_color_images(N_IMAGES)
    return _COLOR_IMAGES_CACHE


def compute_ood_split_gaps(schedules: dict):
    """Run 40 tiny-imagenet RGB images at MISSING_FRACTION (mask seed 42)."""
    images = _get_color_images()
    K_total = K_MAX * NUM_REPS

    results = {arch: np.empty((N_IMAGES, K_total + 1), dtype=np.float64)
               for arch in ARCHS}
    f_opts = np.empty(N_IMAGES, dtype=np.float64)

    for i in tqdm(range(N_IMAGES), desc="ood (color)"):
        gaps, f_opt = evaluate_color_image(
            images[i], MISSING_FRACTION, MASK_SEED, schedules,
        )
        for arch in ARCHS:
            results[arch][i] = gaps[arch]
        f_opts[i] = f_opt

    return results, f_opts


# ==================== NPZ cache ====================

def _split_npz_path(split: str) -> Path:
    return PAPER_PLOTS_DIR / f"pdlp_{split}_gaps_K{K_MAX}_reps{NUM_REPS}.npz"


def save_split_gaps_npz(split: str, results: dict, f_opts: np.ndarray) -> Path:
    path = _split_npz_path(split)
    payload = {arch: results[arch].astype(np.float64) for arch in ARCHS}
    payload["f_opts"] = f_opts.astype(np.float64)
    payload["K_max"] = np.int64(K_MAX)
    payload["num_reps"] = np.int64(NUM_REPS)
    np.savez(path, **payload)
    return path


def load_split_gaps_npz(split: str):
    path = _split_npz_path(split)
    z = np.load(path)
    results = {arch: np.asarray(z[arch], dtype=np.float64) for arch in ARCHS}
    f_opts = np.asarray(z["f_opts"], dtype=np.float64)
    return results, f_opts


# ==================== pdlp_losses ====================

def _losses_csv_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_losses.csv"


def _losses_pdf_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_losses.pdf"


def aggregate_losses_data(gaps_by_split: dict) -> dict:
    """Pivot per-split per-arch gap arrays into the panel-keyed structure
    consumed by ``make_losses_figure``.

    Returns ``data[row_label][arch] -> list of (K, mean, q10, q90)`` for K in
    ``K_VALS_LOSSES``. Mean/q10/q90 come from
    ``aggregate_avg_stats`` so quantile conventions match the reference
    scripts.
    """
    data = {row: defaultdict(list) for row in ROW_LABELS}
    for row in ROW_LABELS:
        results, _ = gaps_by_split[row]
        stats = aggregate_avg_stats(results, ARCHS)
        for arch in ARCHS:
            mean = stats[arch]["mean"]
            q10 = stats[arch]["q10"]
            q90 = stats[arch]["q90"]
            # Median is not part of aggregate_avg_stats; the paper's
            # appendix tables need it, so compute it here per iterate.
            med = np.quantile(np.asarray(results[arch]), 0.5, axis=0)
            for K in K_VALS_LOSSES:
                data[row][arch].append((
                    K,
                    float(mean[K]),
                    float(q10[K]),
                    float(q90[K]),
                    float(med[K]),
                ))
    return data


def save_losses_csv(data: dict, csv_path: Path) -> None:
    rows = []
    for row_label, arch_to_points in data.items():
        for arch, points in arch_to_points.items():
            for K, mean, q10, q90, med in points:
                rows.append({
                    "row": row_label,
                    "K": K,
                    "arch": arch,
                    "gap_mean": mean,
                    "gap_q10": q10,
                    "gap_q90": q90,
                    "gap_median": med,
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_losses_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    data = {row: defaultdict(list) for row in ROW_LABELS}
    for _, r in df.iterrows():
        data[r["row"]][r["arch"]].append((
            int(r["K"]),
            float(r["gap_mean"]),
            float(r["gap_q10"]),
            float(r["gap_q90"]),
            float(r["gap_median"]) if "gap_median" in df.columns else float("nan"),
        ))
    for row in data:
        for arch in data[row]:
            data[row][arch].sort(key=lambda p: p[0])
    return data


def make_losses_figure(data: dict) -> "plt.Figure":
    """1 x 2 figure: in-distribution (left), out-of-distribution (right).

    Mean line + [q10, q90] shaded band, log-y. Style copied from
    ``lasso/losses.py:make_losses_figure``; only the y-axis label switches
    from ``f(x^K) - f(x^*)`` to "Lagrangian gap" since that is PDLP's metric.
    """
    panel_specs = [
        ("test", "In-distribution",     "Avg. Lagrangian gap"),
        ("ood",  "Out-of-distribution", "Avg. Lagrangian gap"),
    ]
    log_floor = 1e-30

    with plt.rc_context({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True)

        for col_idx, (row_label, title, ylabel) in enumerate(panel_specs):
            ax = axes[col_idx]
            arch_to_points = data[row_label]
            if not arch_to_points:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                for arch in ARCHS:
                    points = arch_to_points.get(arch)
                    if not points:
                        continue
                    Ks = np.array([p[0] for p in points])
                    means = np.array([p[1] for p in points])
                    q10s = np.array([p[2] for p in points])
                    q90s = np.array([p[3] for p in points])
                    color = ARCH_COLORS.get(arch)
                    ax.plot(
                        Ks, means,
                        marker=ARCH_MARKERS.get(arch, "o"), markersize=5,
                        label=ARCH_DISPLAY_NAMES.get(arch, arch),
                        color=color,
                    )
                    if np.all(np.isfinite(q10s)) and np.all(np.isfinite(q90s)):
                        lower = np.maximum(q10s, log_floor)
                        ax.fill_between(Ks, lower, q90s, color=color,
                                        alpha=0.2, linewidth=0)
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(r"$K$")
            ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(X_TICKS)

        handles, labels = [], []
        seen = set()
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    handles.append(h)
                    labels.append(l)
                    seen.add(l)

        fig.tight_layout()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.05),
                ncol=len(handles),
                frameon=True,
            )
    return fig


# ==================== pdlp_frac_problems_solved ====================

def _frac_csv_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_frac_problems_solved.csv"


def _frac_pdf_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_frac_problems_solved.pdf"


def compute_frac_solved_for_splits(gaps_by_split: dict) -> dict:
    """Returns ``frac[row_label][eta][arch] -> sorted list of (K, frac)``.

    Reuses ``tv_inpainting_color_averages.compute_frac_solved`` per split,
    which already implements the eta * (1 + max(f^*, 0)) thresholding.
    """
    K_total = K_MAX * NUM_REPS
    out = {}
    for row in ROW_LABELS:
        results, f_opts = gaps_by_split[row]
        per_eta = compute_frac_solved(results, list(f_opts), ETA_VALS_FRAC, K_total)
        out[row] = per_eta
    return out


def save_frac_solved_csv(frac_data: dict, csv_path: Path) -> None:
    rows = []
    for row_label, by_eta in frac_data.items():
        for eta, by_arch in by_eta.items():
            for arch, pairs in by_arch.items():
                for K, frac in pairs:
                    rows.append({
                        "row": row_label,
                        "eta": float(eta),
                        "K": int(K),
                        "arch": arch,
                        "frac_solved": float(frac),
                    })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_frac_solved_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    out = {row: {} for row in ROW_LABELS}
    for row_label, sub in df.groupby("row"):
        for eta_val, sub_eta in sub.groupby("eta"):
            arch_map = {}
            for arch, sub_arch in sub_eta.groupby("arch"):
                pairs = list(zip(
                    sub_arch["K"].astype(int).tolist(),
                    sub_arch["frac_solved"].astype(float).tolist(),
                ))
                pairs.sort(key=lambda p: p[0])
                arch_map[arch] = pairs
            out[row_label][float(eta_val)] = arch_map
    return out


def make_frac_solved_figure(frac_data: dict) -> "plt.Figure":
    """2 x len(eta) grid. Mirrors
    ``lasso/create_paper_plots.py:make_frac_solved_figure``."""
    eta_vals = sorted(set().union(*(frac_data[row].keys() for row in ROW_LABELS)))
    n_eta = len(eta_vals)

    with plt.rc_context({
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }):
        # Sized to match the paper's textwidth (~7 in) so the rc_context
        # font sizes render at full scale rather than getting shrunk by
        # \includegraphics[width=\textwidth].
        fig, axes = plt.subplots(2, n_eta, figsize=(7, 2.5), sharex=True)
        if n_eta == 1:
            axes = axes.reshape(2, 1)

        for row_idx, row_label in enumerate(ROW_LABELS):
            for col_idx, eta in enumerate(eta_vals):
                ax = axes[row_idx, col_idx]
                arch_to_points = frac_data[row_label].get(eta, {})

                if not arch_to_points:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                else:
                    for arch in ARCHS:
                        pairs = arch_to_points.get(arch)
                        if not pairs:
                            continue
                        Ks = [p[0] for p in pairs]
                        fracs = [p[1] for p in pairs]
                        ax.plot(
                            Ks, fracs,
                            marker=ARCH_MARKERS.get(arch, "o"), markersize=5,
                            label=ARCH_DISPLAY_NAMES.get(arch, arch),
                            color=ARCH_COLORS.get(arch),
                        )
                    ax.set_ylim([0, 1.05])
                    ax.set_yticks([0, 0.5, 1])

                ax.grid(True, alpha=0.3)
                ax.set_xticks(X_TICKS)

                if row_idx == 0:
                    ax.set_title(rf"$\eta$ = {eta:g}")
                if row_idx == len(ROW_LABELS) - 1:
                    ax.set_xlabel(r"$K$")
                if col_idx == 0:
                    ax.set_ylabel(ROW_TITLES[row_label])

        fig.suptitle("Test set, fraction of problems solved", y=0.995)

        handles, labels = [], []
        seen = set()
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    handles.append(h)
                    labels.append(l)
                    seen.add(l)

        # Squeeze the suptitle/axes gap by giving tight_layout almost the full
        # height, then explicitly pull the axes top up close to the suptitle
        # bottom. Legend is pushed up into the figure rect so it hugs the
        # K-axis labels.
        # Explicit layout: keeps the panels as tall as the 2.5 in figure
        # allows once the titles, K-axis labels and legend have room.
        fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.19,
                            hspace=0.24, wspace=0.16)
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.045),
                ncol=3,
                frameon=True,
                fontsize=10,
            )
    return fig


# ==================== pdlp_reconstructions ====================

def _recon_npz_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_reconstructions.npz"


def _recon_pdf_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_reconstructions.pdf"


def _final_x_for_schedule(matrices, x0, y0, schedule) -> np.ndarray:
    """Run silent PDHG for K_MAX iterations and return the final primal x.

    Per-K schedule dicts use the K=K_MAX entry — i.e. the schedule trained
    for the full horizon, mirroring ``tv_inpainting_color_averages.py``'s
    best-image plot (line 583).
    """
    if isinstance(schedule, dict):
        tau, sigma, theta = schedule[K_MAX]
    else:
        tau, sigma, theta = schedule
    xk, _ = _pdhg_final_iterate(matrices, x0, y0, tau, sigma, theta)
    return xk


def pick_best_drl2o_index(results: dict) -> int:
    """Return the per-instance argmax of ``l2o_final - ldro_pep_final`` —
    i.e. the index where DR-L2O beats L2O by the biggest final-iterate
    gap margin."""
    l2o_final = results["l2o"][:, -1]
    ldro_final = results["ldro_pep"][:, -1]
    return int(np.argmax(l2o_final - ldro_final))


def pick_top_drl2o_indices(results: dict, n: int) -> list:
    """Return the top-``n`` instance indices in descending order of
    ``l2o_final - ldro_pep_final``. Index 0 of the result equals
    ``pick_best_drl2o_index(results)``."""
    margin = results["l2o"][:, -1] - results["ldro_pep"][:, -1]
    sorted_idx = np.argsort(-margin)
    return [int(i) for i in sorted_idx[:n]]


def compute_olivetti_reconstructions(schedules: dict, face_index: int) -> dict:
    """Reconstructions for one Olivetti face. Returns ``{key: (M, N) array
    in [0, 1]}`` with keys ``original``, ``corrupted``, ``lp``, and one per
    arch in ``ARCHS``."""
    problem = generate_tv_inpainting_problem(
        missing_fraction=MISSING_FRACTION,
        random_seed=42,
        face_index=face_index,
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

    original = problem["image"] / LP_UPPER
    out = {
        "original": original,
        "corrupted": np.where(problem["mask"], original, 0.0),
        "lp": solution["raw_x"][:K].reshape(M, N) / LP_UPPER,
    }
    for arch in ARCHS:
        xk = _final_x_for_schedule(matrices, x0, y0, schedules[arch])
        out[arch] = xk[:K].reshape(M, N) / LP_UPPER
    return out


def compute_color_reconstructions(schedules: dict, image_index: int) -> dict:
    """Reconstructions for one tiny-imagenet RGB image. Returns
    ``{key: (M, N, 3) array in [0, 1]}``."""
    images = _get_color_images()
    img = images[image_index]
    matrices = build_color_lp(img, MISSING_FRACTION, MASK_SEED)
    solution = solve_lp(matrices)
    n_vars = matrices.c.size
    m1 = matrices.G.shape[0]
    S_total = matrices.A.shape[0]
    x0, y0 = build_strict_interior_init(n_vars, m1, S_total, LP_UPPER)

    M, N = img.shape[:2]
    rng = np.random.default_rng(MASK_SEED)
    mask = rng.random((M, N)) >= MISSING_FRACTION
    original = img.astype(np.float64) / 255.0

    out = {
        "original": original,
        "corrupted": np.where(mask[:, :, None], original, 0.0),
        "lp": _extract_rgb_pixels(solution["raw_x"], M, N) / LP_UPPER,
    }
    for arch in ARCHS:
        xk = _final_x_for_schedule(matrices, x0, y0, schedules[arch])
        out[arch] = _extract_rgb_pixels(xk, M, N) / LP_UPPER
    return out


def save_reconstructions_npz(
    olivetti: dict, color: dict, face_index: int, image_index: int, path: Path,
) -> None:
    payload = {f"olivetti_{k}": v.astype(np.float32) for k, v in olivetti.items()}
    payload.update({f"color_{k}": v.astype(np.float32) for k, v in color.items()})
    payload["olivetti_face_index"] = np.int64(face_index)
    payload["color_image_index"] = np.int64(image_index)
    np.savez(path, **payload)


def load_reconstructions_npz(path: Path):
    z = np.load(path)
    olivetti, color = {}, {}
    for key in z.files:
        if key in ("olivetti_face_index", "color_image_index"):
            continue
        if key.startswith("olivetti_"):
            olivetti[key[len("olivetti_"):]] = np.asarray(z[key])
        elif key.startswith("color_"):
            color[key[len("color_"):]] = np.asarray(z[key])
    face_index = int(z["olivetti_face_index"]) if "olivetti_face_index" in z.files else -1
    image_index = int(z["color_image_index"]) if "color_image_index" in z.files else -1
    return olivetti, color, face_index, image_index


def make_reconstructions_figure(olivetti: dict, color: dict) -> "plt.Figure":
    """2 x 6 grid. Rows: Olivetti / tiny-imagenet. Cols: Original, Corrupted,
    LP, OPT-PEP, L2O, DR-L2O. Top-row column titles only; row labels on the
    leftmost panel."""
    miss_pct = int(round(MISSING_FRACTION * 100))
    column_titles = (
        ["Original", f"Corrupted", "Opt. LP Reconstruction"]
        + [ARCH_DISPLAY_NAMES[a] for a in RECON_ARCH_ORDER]
    )
    rows = [
        ("Olivetti", olivetti),
        ("Tiny-ImageNet", color),
    ]

    with plt.rc_context({
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
    }):
        fig, axes = plt.subplots(2, 6, figsize=(3.0 * 6, 3.2 * 2))
        for row_idx, (row_label, recons) in enumerate(rows):
            col_imgs = (
                [recons["original"], recons["corrupted"], recons["lp"]]
                + [recons[a] for a in RECON_ARCH_ORDER]
            )
            cmap = "gray" if col_imgs[0].ndim == 2 else None
            for col_idx, img in enumerate(col_imgs):
                ax = axes[row_idx, col_idx]
                ax.imshow(np.clip(img, 0.0, 1.0), cmap=cmap, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(column_titles[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(row_label)
        fig.tight_layout()
    return fig


# ==================== pdlp_more_reconstructions ====================

def _more_recon_npz_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_more_reconstructions.npz"


def _more_recon_pdf_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_more_reconstructions.pdf"


def save_more_reconstructions_npz(
    row1: dict, row2: dict, idx1: int, idx2: int, path: Path,
) -> None:
    payload = {}
    for prefix, recons in (("row1", row1), ("row2", row2)):
        for k, v in recons.items():
            payload[f"{prefix}_{k}"] = v.astype(np.float32)
    payload["row1_image_index"] = np.int64(idx1)
    payload["row2_image_index"] = np.int64(idx2)
    np.savez(path, **payload)


def load_more_reconstructions_npz(path: Path):
    z = np.load(path)
    row1, row2 = {}, {}
    for key in z.files:
        if key in ("row1_image_index", "row2_image_index"):
            continue
        if key.startswith("row1_"):
            row1[key[len("row1_"):]] = np.asarray(z[key])
        elif key.startswith("row2_"):
            row2[key[len("row2_"):]] = np.asarray(z[key])
    idx1 = int(z["row1_image_index"]) if "row1_image_index" in z.files else -1
    idx2 = int(z["row2_image_index"]) if "row2_image_index" in z.files else -1
    return row1, row2, idx1, idx2


def make_more_reconstructions_figure(color_rows: list) -> "plt.Figure":
    """2 x 6 grid of tiny-imagenet reconstructions. Same column layout as
    ``make_reconstructions_figure``; both rows are color images."""
    miss_pct = int(round(MISSING_FRACTION * 100))
    column_titles = (
        ["Original", f"Corrupted", "Opt. LP Reconstruction"]
        + [ARCH_DISPLAY_NAMES[a] for a in RECON_ARCH_ORDER]
    )

    with plt.rc_context({
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
    }):
        fig, axes = plt.subplots(2, 6, figsize=(3.0 * 6, 3.2 * 2))
        for row_idx, recons in enumerate(color_rows):
            col_imgs = (
                [recons["original"], recons["corrupted"], recons["lp"]]
                + [recons[a] for a in RECON_ARCH_ORDER]
            )
            for col_idx, img in enumerate(col_imgs):
                ax = axes[row_idx, col_idx]
                ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(column_titles[col_idx])
        fig.tight_layout()
    return fig


# ==================== pdlp_times ====================

def _times_csv_path() -> Path:
    return PAPER_PLOTS_DIR / "pdlp_times.csv"


def _progress_csv_for(arch: str, K: int) -> "Path | None":
    """Resolve ``{ARCH_TO_SUBDIR[arch]}_timings/K_{K}/progress.csv``."""
    p = PDLP_DIR / f"{ARCH_TO_SUBDIR[arch]}_timings" / f"K_{K}" / "progress.csv"
    return p if p.exists() else None


def _collect_iter_times(progress_path: Path) -> "np.ndarray | None":
    """Read ``iter_time`` and drop the first ``WARMUP_ITERS`` rows. Returns
    None if the CSV is missing or shorter than the warmup window."""
    if not progress_path.exists():
        return None
    df = pd.read_csv(progress_path, usecols=["iter_time"])
    times = df["iter_time"].to_numpy()
    if times.size <= WARMUP_ITERS:
        return None
    return times[WARMUP_ITERS:]


def collect_times_rows() -> list:
    """Return ``[(arch, K, mean_s, two_sigma_s), ...]`` skipping any (arch, K)
    cell with missing or too-short progress data."""
    out = []
    for arch in FRAMEWORK_ORDER_TIMES:
        for K in K_VALS_TIMES:
            progress_path = _progress_csv_for(arch, K)
            if progress_path is None:
                continue
            iter_times = _collect_iter_times(progress_path)
            if iter_times is None or iter_times.size < 2:
                continue
            mean_s = float(iter_times.mean())
            # ddof=1 for unbiased sigma; n is at most a few hundred.
            two_sigma = float(2.0 * iter_times.std(ddof=1))
            out.append((arch, K, mean_s, two_sigma))
    return out


def format_times_rows(rows: list) -> pd.DataFrame:
    """First-row-of-each-framework-only layout, mirroring lasso/quad."""
    out = []
    last_arch = None
    for arch, K, mean_s, two_sigma in rows:
        framework = ARCH_DISPLAY_NAMES.get(arch, arch) if arch != last_arch else ""
        time_str = f"${mean_s:.3f} \\pm {two_sigma:.3f}$"
        out.append({"Framework": framework, "K": K, "Time": time_str})
        last_arch = arch
    return pd.DataFrame(out, columns=["Framework", "K", "Time"])


def format_times_latex(rows: list) -> str:
    """Emit a copy/pasteable ``booktabs`` tabular environment."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Framework & $K$ & Time per SGD iter (s) \\",
        r"\midrule",
    ]
    last_arch = None
    for arch, K, mean_s, two_sigma in rows:
        framework = ARCH_DISPLAY_NAMES.get(arch, arch) if arch != last_arch else ""
        lines.append(
            f"{framework} & {K} & ${mean_s:.3f} \\pm {two_sigma:.3f}$ \\\\"
        )
        last_arch = arch
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_times() -> None:
    """Write ``paper_plots/pdlp_times.csv`` and print the LaTeX tabular."""
    rows = collect_times_rows()
    if not rows:
        print("  No timing data found, skipping times.")
        return
    df = format_times_rows(rows)
    csv_path = _times_csv_path()
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.relative_to(PDLP_DIR)}")
    print("\n  LaTeX table (copy/paste):\n")
    print(format_times_latex(rows))


# ==================== Main ====================

def _ensure_split_gaps(split: str, schedules: dict, recompute: bool):
    """Return ``(results, f_opts)`` for ``split``, hitting NPZ cache when
    available."""
    npz_path = _split_npz_path(split)
    if not recompute and npz_path.exists():
        print(f"  Loading cached gaps: {npz_path.name}")
        return load_split_gaps_npz(split)

    if split == "in":
        results, f_opts = compute_in_split_gaps(schedules)
    elif split == "ood":
        results, f_opts = compute_ood_split_gaps(schedules)
    else:
        raise ValueError(f"unknown split {split!r}")

    save_split_gaps_npz(split, results, f_opts)
    print(f"  Saved {npz_path.relative_to(PDLP_DIR)}")
    return results, f_opts


def main():
    parser = argparse.ArgumentParser(description="PDLP paper plots.")
    parser.add_argument(
        "-recompute", action="store_true",
        help="Invalidate every cache (NPZ gaps + per-figure CSVs) and "
             "recompute from scratch.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Paper plots (PDLP)")
    print("=" * 60)

    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    schedules = build_schedules_dict(K_MAX)

    print("\n  [gap trajectories]")
    gaps_by_split = {
        "test": _ensure_split_gaps("in", schedules, args.recompute),
        "ood":  _ensure_split_gaps("ood", schedules, args.recompute),
    }

    print("\n  [losses]")
    losses_csv = _losses_csv_path()
    if not args.recompute and losses_csv.exists():
        print(f"  Loading cached losses data ({losses_csv.name})...")
        losses_data = load_losses_csv(losses_csv)
    else:
        losses_data = aggregate_losses_data(gaps_by_split)
        save_losses_csv(losses_data, losses_csv)
        print(f"  Saved {losses_csv.relative_to(PDLP_DIR)}")
    fig = make_losses_figure(losses_data)
    fig.savefig(_losses_pdf_path(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_losses_pdf_path().relative_to(PDLP_DIR)}")

    print("\n  [frac_problems_solved]")
    frac_csv = _frac_csv_path()
    if not args.recompute and frac_csv.exists():
        print(f"  Loading cached frac-solved data ({frac_csv.name})...")
        frac_data = load_frac_solved_csv(frac_csv)
    else:
        frac_data = compute_frac_solved_for_splits(gaps_by_split)
        save_frac_solved_csv(frac_data, frac_csv)
        print(f"  Saved {frac_csv.relative_to(PDLP_DIR)}")
    fig = make_frac_solved_figure(frac_data)
    fig.savefig(_frac_pdf_path(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_frac_pdf_path().relative_to(PDLP_DIR)}")

    print("\n  [reconstructions]")
    recon_npz = _recon_npz_path()
    if not args.recompute and recon_npz.exists():
        print(f"  Loading cached reconstructions ({recon_npz.name})...")
        olivetti, color, face_idx, image_idx = load_reconstructions_npz(recon_npz)
        print(f"  cached olivetti face_index = {face_idx}, "
              f"color image_index = {image_idx}")
    else:
        in_results, _ = gaps_by_split["test"]
        ood_results, _ = gaps_by_split["ood"]
        face_idx = pick_best_drl2o_index(in_results)
        image_idx = pick_best_drl2o_index(ood_results)
        in_margin = float(in_results["l2o"][face_idx, -1]
                          - in_results["ldro_pep"][face_idx, -1])
        ood_margin = float(ood_results["l2o"][image_idx, -1]
                           - ood_results["ldro_pep"][image_idx, -1])
        print(f"  olivetti face_index = {face_idx}  "
              f"(l2o - ldro_pep final-gap margin = {in_margin:.6e})")
        print(f"  color image_index   = {image_idx}  "
              f"(l2o - ldro_pep final-gap margin = {ood_margin:.6e})")
        olivetti = compute_olivetti_reconstructions(schedules, face_idx)
        color = compute_color_reconstructions(schedules, image_idx)
        save_reconstructions_npz(olivetti, color, face_idx, image_idx, recon_npz)
        print(f"  Saved {recon_npz.relative_to(PDLP_DIR)}")
    fig = make_reconstructions_figure(olivetti, color)
    fig.savefig(_recon_pdf_path(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_recon_pdf_path().relative_to(PDLP_DIR)}")

    print("\n  [more reconstructions]")
    more_recon_npz = _more_recon_npz_path()
    # Ranks (1-indexed) of the OOD images to display, by descending
    # ``l2o_final - ldro_pep_final`` margin. Rank 1 already appears in
    # pdlp_reconstructions.pdf; here we show the next two we want.
    more_recon_ranks = (6, 7)
    row1 = row2 = None
    if not args.recompute and more_recon_npz.exists():
        print(f"  Loading cached more reconstructions ({more_recon_npz.name})...")
        row1, row2, idx1, idx2 = load_more_reconstructions_npz(more_recon_npz)
        print(f"  cached image indices: row1 = {idx1}, row2 = {idx2}")
    else:
        ood_results, _ = gaps_by_split["ood"]
        max_rank = max(more_recon_ranks)
        top_indices = pick_top_drl2o_indices(ood_results, n=max_rank)
        if len(top_indices) < max_rank:
            print(f"  Need at least {max_rank} OOD instances for ranks "
                  f"{more_recon_ranks} (have {len(top_indices)}); skipping.")
        else:
            r1, r2 = more_recon_ranks
            idx1 = top_indices[r1 - 1]
            idx2 = top_indices[r2 - 1]
            for rank, idx in ((r1, idx1), (r2, idx2)):
                margin = float(ood_results["l2o"][idx, -1]
                               - ood_results["ldro_pep"][idx, -1])
                print(f"  rank{rank}: image_index = {idx}  "
                      f"(l2o - ldro_pep final-gap margin = {margin:.6e})")
            row1 = compute_color_reconstructions(schedules, idx1)
            row2 = compute_color_reconstructions(schedules, idx2)
            save_more_reconstructions_npz(row1, row2, idx1, idx2, more_recon_npz)
            print(f"  Saved {more_recon_npz.relative_to(PDLP_DIR)}")
    if row1 is not None and row2 is not None:
        fig = make_more_reconstructions_figure([row1, row2])
        fig.savefig(_more_recon_pdf_path(), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {_more_recon_pdf_path().relative_to(PDLP_DIR)}")

    print("\n  [times]")
    make_times()

    print("\nDone.")


if __name__ == "__main__":
    main()
