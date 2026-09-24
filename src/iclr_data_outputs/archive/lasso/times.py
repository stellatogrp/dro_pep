"""
Lasso `times` paper-plot artifact.

For the validation-best stepsize schedule used by the losses figure (one per
arch per K), reach back to the SGD progress.csv that produced it and report
mean +/- 2*sigma wall-time per iteration. The first WARMUP_ITERS SGD
iterations are dropped to discard JAX warmup. Output is a LaTeX-ready CSV
with one row per (framework, K) combo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from create_paper_plots import (
    ARCH_DISPLAY_NAMES,
    find_config_with_N,
    output_paths,
)
from create_test_plots import ARCH_TO_CSV
from data_scrape import get_leaf_path


# ==================== Configuration ====================

K_VALS_TIMES = [1, 5, 10, 15]
WARMUP_ITERS = 5  # drop the first 5 SGD iters per JAX warmup
TRAINING_SAMPLE_N_FOR_TIMES = 1000  # match the losses plot

# Display-row order for the table.
FRAMEWORK_ORDER = ['l2o', 'l2o_alista', 'ldro_pep', 'lpep']


# ==================== Data Collection ====================

def collect_iter_times(progress_path: Path):
    """Return the post-warmup `iter_time` values from a training progress CSV,
    or None if the file is missing or too short to drop the warmup window.
    """
    if not progress_path.exists():
        return None
    df = pd.read_csv(progress_path, usecols=['iter_time'])
    times = df['iter_time'].to_numpy()
    if times.size <= WARMUP_ITERS:
        return None
    return times[WARMUP_ITERS:]


def collect_times_data(target_cfg, problem_dir: Path):
    """For each (arch, K) in FRAMEWORK_ORDER x K_VALS_TIMES, locate the
    progress.csv that produced the validation-best schedule and aggregate
    iter_time. Returns a list of (arch, K, mean_s, two_sigma_s) tuples,
    skipping combos with missing data.
    """
    leaf_path = get_leaf_path(target_cfg)
    out = []
    for arch in FRAMEWORK_ORDER:
        if arch not in ARCH_TO_CSV:
            continue
        for K in K_VALS_TIMES:
            best_csv = leaf_path / f'K_{K}' / ARCH_TO_CSV[arch]
            if not best_csv.exists():
                continue
            best_df = pd.read_csv(best_csv)
            source_dir = best_df.iloc[0].get('source_dir')
            if not isinstance(source_dir, str):
                continue
            progress_path = (
                problem_dir / source_dir / 'learn_dro_outputs'
                / f'K_{K}' / 'progress.csv'
            )
            iter_times = collect_iter_times(progress_path)
            if iter_times is None or iter_times.size < 2:
                continue
            mean_s = float(iter_times.mean())
            # ddof=1 for an unbiased sigma estimate; n is at most a few
            # hundred so the bias matters more than the variance.
            two_sigma = float(2.0 * iter_times.std(ddof=1))
            out.append((arch, K, mean_s, two_sigma))
    return out


# ==================== Formatting ====================

def format_rows(rows):
    """Build a DataFrame with the 'first-row-of-each-framework only' layout.

    The Framework cell is filled only on the first K of each arch block so
    the LaTeX-rendered table has visual whitespace down the leftmost column.
    """
    out = []
    last_arch = None
    for arch, K, mean_s, two_sigma in rows:
        framework = ARCH_DISPLAY_NAMES.get(arch, arch) if arch != last_arch else ''
        time_str = f'${mean_s:.3f} \\pm {two_sigma:.3f}$'
        out.append({'Framework': framework, 'K': K, 'Time': time_str})
        last_arch = arch
    return pd.DataFrame(out, columns=['Framework', 'K', 'Time'])


# ==================== Entry Point (called by create_paper_plots.main) ====================

def make_times_for_group(cfg_list, group_label, paper_plots_dir,
                         single_group, problem_dir):
    """Compute the timings table for one config group and write the CSV.

    No PDF is produced; we reuse `output_paths` only for the CSV naming.
    """
    _, csv_path = output_paths(
        paper_plots_dir, 'times', group_label, single_group,
    )
    target_cfg = find_config_with_N(cfg_list, TRAINING_SAMPLE_N_FOR_TIMES)
    if target_cfg is None:
        print(f'  No config with training_sample_N={TRAINING_SAMPLE_N_FOR_TIMES}, '
              f'skipping times.')
        return

    rows = collect_times_data(target_cfg, problem_dir)
    if not rows:
        print('  No timing data found, skipping times.')
        return

    df = format_rows(rows)
    df.to_csv(csv_path, index=False)
    print(f'  Saved {csv_path.relative_to(problem_dir)}')
