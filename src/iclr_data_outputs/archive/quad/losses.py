"""
Quad `losses` paper plot.

1 x 2 grid:
    - Left:  in-distribution (test set) final-iterate loss vs. K.
    - Right: out-of-distribution            final-iterate loss vs. K.

For each K, uses that K's *own* best schedule (loaded from
`ldro_pep_best_stepsize_schedule.csv` etc. under `plots/.../K_{K}/`) and
recomputes per-instance final-iterate losses on the test and OOD batches so we
can report the mean along with the 10th and 90th percentiles. The plot shows
the mean as a line and [q10, q90] as shading.

Pulls a single `training_sample_N` slice (TRAINING_SAMPLE_N_FOR_LOSSES).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_paper_plots import (
    ARCH_COLORS,
    ARCH_DISPLAY_NAMES,
    ARCH_MARKERS,
    find_config_with_N,
    output_paths,
    parse_config_list_to_dict,
)
from create_test_plots import (
    ARCH_TO_CSV,
    compute_loss_trajectory,
    load_best_stepsize,
    load_ood_test_data,
    load_test_data,
)
from data_scrape import get_leaf_path


# ==================== Configuration ====================

TRAINING_SAMPLE_N_FOR_LOSSES = 1000
K_VALS_LOSSES = list(range(1, 16))

PANEL_SPECS = [
    {
        'label': 'test',
        'csv_name': 'out_of_sample_loss_data.csv',
        'title': 'In-distribution',
        'ylabel': r'Avg. $f(x^K) - f(x^\star)$',
    },
    {
        'label': 'ood',
        'csv_name': 'out_of_distribution_loss_data.csv',
        'title': 'Out-of-distribution',
        'ylabel': r'Avg. $f(x^K) - f(x^\star)$',
    },
]

# Even-integer x-ticks (avoid auto .5 labels on a discrete-K axis).
X_TICKS = [2, 4, 6, 8, 10, 12, 14]


# ==================== Data Collection ====================

def collect_losses_data(target_cfg, K_vals):
    """Compute per-instance final-iterate losses on test and OOD sets, using
    each K's best schedule from the K_{K} plot dir.

    Returns: data[panel_label][arch] -> sorted list of (K, mean, q10, q90).
    """
    cfg_dict = parse_config_list_to_dict(target_cfg)
    alg = cfg_dict['alg']
    stepsize_type = 'vector'

    leaf_path = get_leaf_path(target_cfg)

    print('  Loading test and OOD problem data...')
    Q_test, z0_test, zs_test, fs_test = load_test_data()
    Q_ood, z0_ood, zs_ood, fs_ood = load_ood_test_data()
    panel_batches = {
        'test': (Q_test, z0_test, zs_test, fs_test),
        'ood':  (Q_ood,  z0_ood,  zs_ood,  fs_ood),
    }

    data = {label: defaultdict(list) for label in panel_batches}

    for K in K_vals:
        k_dir = leaf_path / f'K_{K}'
        if not k_dir.exists():
            continue
        for arch in ARCH_TO_CSV.keys():
            stepsizes, _ = load_best_stepsize(k_dir, arch, alg, stepsize_type, K)
            if stepsizes is None:
                continue
            for label, (Q, z0, zs, fs) in panel_batches.items():
                traj = compute_loss_trajectory(stepsizes, Q, z0, zs, fs, K, alg)
                final = np.asarray(traj[:, K - 1])  # f(z_K) per instance
                q10, med, q90 = np.quantile(final, [0.1, 0.5, 0.9])
                data[label][arch].append(
                    (K, float(final.mean()), float(q10), float(q90), float(med))
                )
        print(f'    K={K}: done')

    for panel_label in data:
        for arch in data[panel_label]:
            data[panel_label][arch].sort(key=lambda p: p[0])

    return data


# ==================== Plot ====================

def make_losses_figure(data):
    """Build the 1 x 2 final-loss-vs-K figure (mean line + [q10, q90] band).

    A single figure-level legend is anchored below both panels (auto-ncol so
    it fits in one row); `bbox_inches='tight'` at savefig time crops to it.
    """
    log_floor = 1e-30  # for log-scale lower bound clipping

    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True)

        for col_idx, panel in enumerate(PANEL_SPECS):
            ax = axes[col_idx]
            arch_to_points = data[panel['label']]

            if not arch_to_points:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                for arch in sorted(arch_to_points.keys()):
                    points = arch_to_points[arch]
                    if not points:
                        continue
                    Ks = np.array([p[0] for p in points])
                    means = np.array([p[1] for p in points])
                    q10s = np.array([p[2] for p in points])
                    q90s = np.array([p[3] for p in points])
                    color = ARCH_COLORS.get(arch)
                    ax.plot(
                        Ks, means,
                        marker=ARCH_MARKERS.get(arch, 'o'), markersize=5,
                        label=ARCH_DISPLAY_NAMES.get(arch, arch),
                        color=color,
                    )
                    if np.all(np.isfinite(q10s)) and np.all(np.isfinite(q90s)):
                        lower = np.maximum(q10s, log_floor)
                        ax.fill_between(Ks, lower, q90s, color=color, alpha=0.2,
                                        linewidth=0)
                ax.set_yscale('log')

            ax.grid(True, alpha=0.3)
            ax.set_xlabel(r'$K$')
            ax.set_title(panel['title'])
            if col_idx == 0:
                ax.set_ylabel(panel['ylabel'])
            ax.set_xticks(X_TICKS)

        # Single shared legend below both panels; dedupe across axes;
        # ncol = #archs so it stays one row.
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
                loc='upper center',
                bbox_to_anchor=(0.5, 0.05),
                ncol=len(handles),
                frameon=True,
            )

    return fig


# ==================== I/O ====================

def save_losses_csv(data, training_sample_N, csv_path):
    """Flatten losses data into long-form CSV."""
    rows = []
    for panel_label, arch_to_points in data.items():
        for arch, points in arch_to_points.items():
            for K, mean, q10, q90, med in points:
                rows.append({
                    'row': panel_label,
                    'K': K,
                    'training_sample_N': training_sample_N,
                    'arch': arch,
                    'final_loss_mean': mean,
                    'final_loss_q10': q10,
                    'final_loss_q90': q90,
                    'final_loss_median': med,
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_losses_csv(csv_path):
    """Inverse of `save_losses_csv`. Returns (data, training_sample_N).

    Backward compat: legacy CSVs with only a `final_loss` column (or with
    `final_loss_std` from an earlier schema) are loaded with q10/q90 set to
    NaN, which suppresses the shading band in the figure.
    """
    df = pd.read_csv(csv_path)
    if 'final_loss_mean' in df.columns:
        mean_col = 'final_loss_mean'
    else:
        mean_col = 'final_loss'  # legacy CSVs
    has_quantiles = (
        'final_loss_q10' in df.columns and 'final_loss_q90' in df.columns
    )

    data = {panel['label']: defaultdict(list) for panel in PANEL_SPECS}
    for _, r in df.iterrows():
        if has_quantiles:
            q10 = float(r['final_loss_q10'])
            q90 = float(r['final_loss_q90'])
            med = (float(r['final_loss_median'])
                   if 'final_loss_median' in df.columns else float('nan'))
        else:
            q10 = float('nan')
            q90 = float('nan')
            med = float('nan')
        data[r['row']][r['arch']].append(
            (int(r['K']), float(r[mean_col]), q10, q90, med)
        )
    for panel_label in data:
        for arch in data[panel_label]:
            data[panel_label][arch].sort(key=lambda p: p[0])
    training_N = int(df['training_sample_N'].iloc[0])
    return data, training_N


# ==================== Entry Point (called by create_paper_plots.main) ====================

def make_losses_for_group(cfg_list, group_label, paper_plots_dir,
                          single_group, quad_dir, recompute=False):
    pdf_path, csv_path = output_paths(
        paper_plots_dir, 'losses', group_label, single_group,
    )

    if not recompute and csv_path.exists():
        print(f'  Loading cached losses data ({csv_path.name})...')
        data, _ = load_losses_csv(csv_path)
    else:
        target_cfg = find_config_with_N(cfg_list, TRAINING_SAMPLE_N_FOR_LOSSES)
        if target_cfg is None:
            print(f'  No config with training_sample_N={TRAINING_SAMPLE_N_FOR_LOSSES}, '
                  f'skipping losses.')
            return

        data = collect_losses_data(target_cfg, K_VALS_LOSSES)
        any_data = any(arch_to_points for arch_to_points in data.values())
        if not any_data:
            print('  No cached loss data found, skipping losses.')
            return

        save_losses_csv(data, TRAINING_SAMPLE_N_FOR_LOSSES, csv_path)
        print(f'  Saved {csv_path.relative_to(quad_dir)}')

    fig = make_losses_figure(data)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {pdf_path.relative_to(quad_dir)}')
