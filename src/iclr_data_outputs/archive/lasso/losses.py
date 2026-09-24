"""
Lasso `losses` paper plot.

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

import jax.numpy as jnp
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

# Variant used for the `lasso_intro` figure: drops ALISTA, rewrites y-labels,
# and caps the OOD panel for readability.
INTRO_PANEL_SPECS = [
    {
        'label': 'test',
        'csv_name': 'out_of_sample_loss_data.csv',
        'title': 'In-distribution',
        'ylabel': 'Test loss',
    },
    {
        'label': 'ood',
        'csv_name': 'out_of_distribution_loss_data.csv',
        'title': 'Out-of-distribution',
        'ylabel': 'Test loss',
    },
]
INTRO_EXCLUDE_ARCHS = frozenset({'l2o_alista'})
INTRO_Y_CAPS = [None, 1e3]

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
    A_test, b_test, x_opt_test, f_opt_test, lambd_test, _ = load_test_data()
    A_ood, b_ood, x_opt_ood, f_opt_ood, lambd_ood = load_ood_test_data()
    x0_test = jnp.zeros((b_test.shape[0], A_test.shape[1]))
    x0_ood = jnp.zeros((b_ood.shape[0], A_ood.shape[1]))

    panel_batches = {
        'test': (A_test, b_test, x0_test, x_opt_test, f_opt_test, lambd_test),
        'ood':  (A_ood,  b_ood,  x0_ood,  x_opt_ood,  f_opt_ood,  lambd_ood),
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
            for label, (A, b, x0, x_opt, f_opt, lambd) in panel_batches.items():
                traj = compute_loss_trajectory(
                    stepsizes, A, b, x0, x_opt, f_opt, lambd, K, alg,
                )
                final = np.asarray(traj[:, K - 1])  # f(x_K) per instance
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

def make_losses_figure(data, panel_specs=None, exclude_archs=(), y_caps=None,
                       display_name_overrides=None, figsize=(7, 2.8)):
    """Build the 1 x 2 final-loss-vs-K figure (mean line + [q10, q90] band).

    `panel_specs` overrides the default y-labels per panel, `exclude_archs`
    drops architectures from the plot (still respecting cached CSV data),
    `y_caps[col_idx]` if non-None caps that panel's log-y-axis at the value,
    `display_name_overrides` patches the per-arch legend labels, and
    `figsize` overrides the figure dimensions in inches.

    A single figure-level legend is anchored below both panels (auto-ncol so
    it fits in one row); `bbox_inches='tight'` at savefig time crops to it.
    """
    if panel_specs is None:
        panel_specs = PANEL_SPECS
    display_names = dict(ARCH_DISPLAY_NAMES)
    if display_name_overrides:
        display_names.update(display_name_overrides)

    log_floor = 1e-30  # for log-scale lower bound clipping

    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

        for col_idx, panel in enumerate(panel_specs):
            ax = axes[col_idx]
            arch_to_points = data[panel['label']]

            if not arch_to_points:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                for arch in sorted(arch_to_points.keys()):
                    if arch in exclude_archs:
                        continue
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
                        label=display_names.get(arch, arch),
                        color=color,
                    )
                    if np.all(np.isfinite(q10s)) and np.all(np.isfinite(q90s)):
                        lower = np.maximum(q10s, log_floor)
                        ax.fill_between(Ks, lower, q90s, color=color, alpha=0.2,
                                        linewidth=0)
                ax.set_yscale('log')
                if y_caps is not None and y_caps[col_idx] is not None:
                    ax.set_ylim(top=y_caps[col_idx])

            ax.grid(True, alpha=0.3)
            ax.set_xlabel(r'$K$')
            ax.set_title(panel['title'])
            if col_idx == 0:
                ax.set_ylabel(panel['ylabel'])
            ax.set_xticks(X_TICKS)

        # Single shared legend below both panels. Dedupe across axes in case
        # future panel-specific archs appear; ncol = #archs so it stays one row.
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

    Backward compat: legacy CSVs with only a `final_loss` column are loaded
    with q10/q90 set to NaN, which suppresses the shading band in the figure.
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
                          single_group, lasso_dir, recompute=False):
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
        print(f'  Saved {csv_path.relative_to(lasso_dir)}')

    fig = make_losses_figure(data, y_caps=[None, 1e3])
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {pdf_path.relative_to(lasso_dir)}')


def make_losses_intro_for_group(cfg_list, group_label, paper_plots_dir,
                                single_group, lasso_dir):
    """Render the `lasso_intro` variant: ALISTA-free, relabeled, OOD capped.

    Reuses the cached `lasso_losses.csv` so no recomputation is needed; if the
    cache is missing we skip (the standard losses pass should populate it).
    """
    _, losses_csv_path = output_paths(
        paper_plots_dir, 'losses', group_label, single_group,
    )
    intro_pdf_path, _ = output_paths(
        paper_plots_dir, 'intro', group_label, single_group,
    )

    if not losses_csv_path.exists():
        print(f'  No cached losses CSV ({losses_csv_path.name}); '
              f'skipping lasso_intro.')
        return

    data, _ = load_losses_csv(losses_csv_path)
    fig = make_losses_figure(
        data,
        panel_specs=INTRO_PANEL_SPECS,
        exclude_archs=INTRO_EXCLUDE_ARCHS,
        y_caps=INTRO_Y_CAPS,
        # Without ALISTA there is no other ISTA-flavored arch to disambiguate
        # against, so the intro figure shows the plain "L2O" name.
        display_name_overrides={'l2o': 'L2O'},
        figsize=(7, 2.1),
    )
    fig.savefig(intro_pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {intro_pdf_path.relative_to(lasso_dir)}')
