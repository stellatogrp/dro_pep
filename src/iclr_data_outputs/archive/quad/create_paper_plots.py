"""
Paper plots for Quad experiments.

`training_set_effect`: 2 x 3 grid.
    - Rows: test set (top), out-of-distribution set (bottom).
    - Cols: K = 5, K = 10, K = 15.
    - x-axis: training_sample_N (training set size).
    - y-axis: final-iterate loss.

`frac_problems_solved`: 2 x 3 grid.
    - Rows: test set (top), out-of-distribution set (bottom).
    - Cols: eta = 1e-2, 1e-1, 1.
    - x-axis: K (number of iterations).
    - y-axis: fraction of problems with f(x_K) <= eta.
    - Uses a single training_sample_N (TRAINING_SAMPLE_N_FOR_FRAC).

`losses` (defined in losses.py): 1 x 2 grid of final-iterate test / OOD loss
vs. K, where each K uses its own K-specific best schedule.

Assumes `find_best_stepsizes.py` and `create_test_plots.py` have already been
run.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (12, 6),
})

script_dir = Path(__file__).parent
# Consolidated layout: this file lives at
#   iclr_data_outputs/archive/<problem>/ , so src/ is three levels up,
# where it was two in the original experiment_plots_icml/<problem>/.
src_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from data_scrape import ALL_QUAD_CONFIGS, get_leaf_path
from create_test_plots import (
    ARCH_TO_CSV,
    compute_per_problem_losses,
    load_best_stepsize,
    load_ood_test_data,
    load_test_data,
)


# ==================== Configuration ====================

K_VALS = [5, 10, 15]

# frac_problems_solved plot
K_VALS_FRAC = list(range(1, 16))
ETA_VALS_FRAC = [1e-3, 1e-2, 1e-1]
TRAINING_SAMPLE_N_FOR_FRAC = 1000  # which N to slice for the frac-solved plot

ARCH_DISPLAY_NAMES = {
    'l2o': 'L2O',
    'ldro_pep': 'DR-L2O',
    'lpep': 'OPT-PEP',
}

ARCH_COLORS = {
    'l2o': '#DC3220',
    'ldro_pep': '#005AB5',
    'lpep': '#00B32D',
}

ARCH_MARKERS = {
    'l2o': 'o',          # circle
    'ldro_pep': 's',     # square
    'lpep': '^',         # triangle up
}

# Files to read for the two rows of the figure.
ROW_SPECS = [
    {
        'label': 'test',
        'csv_name': 'out_of_sample_loss_data.csv',
        'ylabel': 'Final test loss',
    },
    {
        'label': 'ood',
        'csv_name': 'out_of_distribution_loss_data.csv',
        'ylabel': 'Final OOD loss',
    },
]


# ==================== Helpers ====================

def parse_config_list_to_dict(config_list):
    """Convert ['alg=vanilla_gd', 'training_sample_N=100', ...] to dict."""
    out = {}
    for item in config_list:
        key, value = item.split('=', 1)
        try:
            out[key] = int(value) if '.' not in value else float(value)
        except ValueError:
            out[key] = value
    return out


def group_key_for_config(config_list):
    """Tuple identifying configs that share a paper-plot figure.

    All keys except `training_sample_N` form the group key, so each unique
    combination of (alg, pep_obj, dro_obj, ...) produces its own figure.
    """
    parts = [item for item in config_list if not item.startswith('training_sample_N=')]
    return tuple(parts)


def read_final_losses(k_dir, csv_name):
    """Read a cached loss-trajectory CSV and return the final-iterate values.

    Returns dict {arch_name: final_loss_float} or None if the file is absent.
    """
    csv_path = k_dir / csv_name
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    last = df.iloc[-1]
    return {
        col: float(last[col])
        for col in df.columns
        if col != 'k' and pd.notna(last[col])
    }


# ==================== Data Collection ====================

def collect_group_data(config_lists):
    """For one group of configs (varying only in training_sample_N), gather
    the final losses keyed by (row_label, K, training_sample_N, arch).
    """
    # nested: data[row_label][K][arch] -> list of (N, final_loss)
    data = {row['label']: {K: defaultdict(list) for K in K_VALS} for row in ROW_SPECS}

    for cfg in config_lists:
        cfg_dict = parse_config_list_to_dict(cfg)
        N = cfg_dict.get('training_sample_N')
        if N is None:
            continue
        leaf_path = get_leaf_path(cfg)
        for K in K_VALS:
            k_dir = leaf_path / f'K_{K}'
            if not k_dir.exists():
                continue
            for row in ROW_SPECS:
                finals = read_final_losses(k_dir, row['csv_name'])
                if finals is None:
                    continue
                for arch, val in finals.items():
                    data[row['label']][K][arch].append((N, val))

    # sort each list by training_sample_N
    for row_label in data:
        for K in data[row_label]:
            for arch in data[row_label][K]:
                data[row_label][K][arch].sort(key=lambda pair: pair[0])

    return data


# ==================== Plot ====================

def make_training_set_effect_figure(data, group_label):
    """Build the 2 x 3 figure for one config group."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for row_idx, row in enumerate(ROW_SPECS):
        for col_idx, K in enumerate(K_VALS):
            ax = axes[row_idx, col_idx]
            arch_to_points = data[row['label']][K]

            if not arch_to_points:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                for arch in sorted(arch_to_points.keys()):
                    pairs = arch_to_points[arch]
                    if not pairs:
                        continue
                    ns = [p[0] for p in pairs]
                    vals = [p[1] for p in pairs]
                    ax.plot(
                        ns, vals,
                        marker='o', markersize=5,
                        label=ARCH_DISPLAY_NAMES.get(arch, arch),
                        color=ARCH_COLORS.get(arch),
                    )
                ax.set_yscale('log')
                ax.legend(fontsize=9)

            ax.grid(True, alpha=0.3)

            if row_idx == 0:
                ax.set_title(f'K = {K}')
            if row_idx == len(ROW_SPECS) - 1:
                ax.set_xlabel('Training set size')
            if col_idx == 0:
                ax.set_ylabel(row['ylabel'])

    fig.suptitle(f'Training set effect ({group_label})', y=1.0, fontsize=12)
    fig.tight_layout()
    return fig


def save_long_csv(data, csv_path):
    """Flatten the nested data dict into a long-form CSV."""
    rows = []
    for row_label, by_K in data.items():
        for K, arch_to_points in by_K.items():
            for arch, pairs in arch_to_points.items():
                for N, val in pairs:
                    rows.append({
                        'row': row_label,
                        'K': K,
                        'training_sample_N': N,
                        'arch': arch,
                        'final_loss': val,
                    })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ==================== frac_problems_solved Data Collection ====================

def find_config_with_N(group_configs, target_N):
    """Return the config in `group_configs` matching the desired
    training_sample_N, or None.
    """
    for cfg in group_configs:
        d = parse_config_list_to_dict(cfg)
        if d.get('training_sample_N') == target_N:
            return cfg
    return None


def collect_frac_solved_data(target_cfg, K_vals, eta_vals):
    """Compute fraction of problems solved at each (K, eta) on test and OOD
    data, for every architecture with a best-stepsize CSV under the chosen
    config's leaf path.

    Returns nested dict: data[row_label][eta][arch] -> list of (K, frac).
    """
    cfg_dict = parse_config_list_to_dict(target_cfg)
    alg = cfg_dict['alg']
    pep_obj = cfg_dict.get('pep_obj', 'obj_val')
    stepsize_type = 'vector'

    leaf_path = get_leaf_path(target_cfg)

    print('  Loading test and OOD problem data...')
    Q_test, z0_test, zs_test, fs_test = load_test_data()
    Q_ood, z0_ood, zs_ood, fs_ood = load_ood_test_data()

    data = {
        'test': {eta: defaultdict(list) for eta in eta_vals},
        'ood': {eta: defaultdict(list) for eta in eta_vals},
    }

    for K in K_vals:
        k_dir = leaf_path / f'K_{K}'
        if not k_dir.exists():
            continue
        for arch in ARCH_TO_CSV.keys():
            stepsizes, _ = load_best_stepsize(k_dir, arch, alg, stepsize_type, K)
            if stepsizes is None:
                continue

            test_losses = np.asarray(compute_per_problem_losses(
                stepsizes, Q_test, z0_test, zs_test, fs_test, K, alg, pep_obj
            ))
            ood_losses = np.asarray(compute_per_problem_losses(
                stepsizes, Q_ood, z0_ood, zs_ood, fs_ood, K, alg, pep_obj
            ))

            for eta in eta_vals:
                data['test'][eta][arch].append((K, float(np.mean(test_losses <= eta))))
                data['ood'][eta][arch].append((K, float(np.mean(ood_losses <= eta))))
        print(f'    K={K}: done')

    for row_label in data:
        for eta in data[row_label]:
            for arch in data[row_label][eta]:
                data[row_label][eta][arch].sort(key=lambda p: p[0])

    return data


def save_frac_solved_csv(data, training_sample_N, csv_path):
    """Flatten frac-solved data into long-form CSV."""
    rows = []
    for row_label, by_eta in data.items():
        for eta, arch_to_points in by_eta.items():
            for arch, pairs in arch_to_points.items():
                for K, frac in pairs:
                    rows.append({
                        'row': row_label,
                        'eta': eta,
                        'K': K,
                        'training_sample_N': training_sample_N,
                        'arch': arch,
                        'frac_solved': frac,
                    })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_frac_solved_csv(csv_path):
    """Inverse of `save_frac_solved_csv`."""
    df = pd.read_csv(csv_path)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, row in df.iterrows():
        data[row['row']][float(row['eta'])][row['arch']].append(
            (int(row['K']), float(row['frac_solved']))
        )
    out = {}
    for row_label, by_eta in data.items():
        out[row_label] = {}
        for eta, arch_to_points in by_eta.items():
            out[row_label][eta] = {
                arch: sorted(pts, key=lambda p: p[0])
                for arch, pts in arch_to_points.items()
            }
    return out


# ==================== frac_problems_solved Plot ====================

def make_frac_solved_figure(data):
    """Build the 2 x len(eta) frac-solved figure."""
    eta_vals = sorted(data['test'].keys())
    n_eta = len(eta_vals)

    # Even-integer x-ticks (avoid auto .5 labels on a discrete-K axis).
    x_ticks = [2, 4, 6, 8, 10, 12, 14]

    row_labels = ['test', 'ood']
    row_titles = ['In-dist.', 'Out-of-dist.']

    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }):
        # Sized to match the paper's textwidth (~7 in) so the rc_context
        # font sizes render at full scale rather than getting shrunk by
        # \includegraphics[width=\textwidth].
        fig, axes = plt.subplots(2, n_eta, figsize=(7, 2.5), sharex=True)

        if n_eta == 1:
            axes = axes.reshape(2, 1)

        for row_idx, row_label in enumerate(row_labels):
            for col_idx, eta in enumerate(eta_vals):
                ax = axes[row_idx, col_idx]
                arch_to_points = data[row_label].get(eta, {})

                if not arch_to_points:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            transform=ax.transAxes, color='gray')
                else:
                    for arch in sorted(arch_to_points.keys()):
                        pairs = arch_to_points[arch]
                        if not pairs:
                            continue
                        Ks = [p[0] for p in pairs]
                        fracs = [p[1] for p in pairs]
                        ax.plot(
                            Ks, fracs,
                            marker=ARCH_MARKERS.get(arch, 'o'), markersize=5,
                            label=ARCH_DISPLAY_NAMES.get(arch, arch),
                            color=ARCH_COLORS.get(arch),
                        )
                    ax.set_ylim([0, 1.05])
                    ax.set_yticks([0, 0.5, 1])

                ax.grid(True, alpha=0.3)
                ax.set_xticks(x_ticks)

                if row_idx == 0:
                    ax.set_title(rf'$\eta$ = {eta:g}')
                if row_idx == len(row_labels) - 1:
                    ax.set_xlabel(r'$K$')
                if col_idx == 0:
                    ax.set_ylabel(row_titles[row_idx])

        fig.suptitle('Test set, fraction of problems solved', y=0.995)

        # Single shared legend below the grid; dedupe across panels.
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
                loc='upper center',
                bbox_to_anchor=(0.5, 0.045),
                ncol=3,
                frameon=True,
                fontsize=10,
            )
    return fig


# ==================== Main ====================

def slugify_group_key(key_tuple):
    """Turn a config-group tuple into a filesystem-safe label."""
    if not key_tuple:
        return 'all'
    return '__'.join(item.replace('=', '_') for item in key_tuple)


FILE_PREFIX = 'quad_'


def output_paths(paper_plots_dir, base_name, group_label, single_group):
    """Pick file names with a group suffix only when there are multiple groups.

    All quad paper-plot artifacts are prefixed with `quad_` so they can be
    dropped alongside the lasso ones (which use `lasso_`) without colliding.
    """
    if single_group:
        pdf = paper_plots_dir / f'{FILE_PREFIX}{base_name}.pdf'
        csv = paper_plots_dir / f'{FILE_PREFIX}{base_name}.csv'
    else:
        pdf = paper_plots_dir / f'{FILE_PREFIX}{base_name}__{group_label}.pdf'
        csv = paper_plots_dir / f'{FILE_PREFIX}{base_name}__{group_label}.csv'
    return pdf, csv


def make_training_set_effect_for_group(cfg_list, group_label, paper_plots_dir,
                                        single_group, quad_dir):
    data = collect_group_data(cfg_list)

    any_data = any(
        arch_to_points
        for by_K in data.values()
        for arch_to_points in by_K.values()
    )
    if not any_data:
        print('  No cached loss data found, skipping training_set_effect.')
        return

    pdf_path, csv_path = output_paths(
        paper_plots_dir, 'training_set_effect', group_label, single_group,
    )
    fig = make_training_set_effect_figure(data, group_label)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    save_long_csv(data, csv_path)
    print(f'  Saved {pdf_path.relative_to(quad_dir)}')
    print(f'  Saved {csv_path.relative_to(quad_dir)}')


def make_frac_solved_for_group(cfg_list, group_label, paper_plots_dir,
                                single_group, quad_dir, recompute):
    pdf_path, csv_path = output_paths(
        paper_plots_dir, 'frac_problems_solved', group_label, single_group,
    )

    if not recompute and csv_path.exists():
        print(f'  Loading cached frac-solved data ({csv_path.name})...')
        data = load_frac_solved_csv(csv_path)
    else:
        target_cfg = find_config_with_N(cfg_list, TRAINING_SAMPLE_N_FOR_FRAC)
        if target_cfg is None:
            print(f'  No config with training_sample_N={TRAINING_SAMPLE_N_FOR_FRAC}, '
                  f'skipping frac_problems_solved.')
            return
        data = collect_frac_solved_data(target_cfg, K_VALS_FRAC, ETA_VALS_FRAC)
        save_frac_solved_csv(data, TRAINING_SAMPLE_N_FOR_FRAC, csv_path)
        print(f'  Saved {csv_path.relative_to(quad_dir)}')

    any_data = any(
        arch_to_points
        for by_eta in data.values()
        for arch_to_points in by_eta.values()
    )
    if not any_data:
        print('  No frac-solved data, skipping plot.')
        return

    fig = make_frac_solved_figure(data)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {pdf_path.relative_to(quad_dir)}')


def main():
    parser = argparse.ArgumentParser(description='Quad paper plots.')
    parser.add_argument('-recompute', action='store_true',
                        help='Recompute frac_problems_solved and losses data '
                             'instead of reading the cached CSVs in '
                             'paper_plots/.')
    args = parser.parse_args()

    print('=' * 60)
    print('Paper plots (Quad)')
    print('=' * 60)

    quad_dir = Path(__file__).parent
    paper_plots_dir = quad_dir / 'paper_plots'
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for cfg in ALL_QUAD_CONFIGS:
        groups[group_key_for_config(cfg)].append(cfg)

    single_group = (len(groups) == 1)
    print(f'Found {len(groups)} config group(s)')

    # Deferred import: losses.py / times.py import shared helpers/constants
    # from this module, so we wait until create_paper_plots is fully defined.
    from losses import make_losses_for_group
    from times import make_times_for_group

    for key, cfg_list in groups.items():
        group_label = slugify_group_key(key)
        print(f'\n--- group: {group_label} ({len(cfg_list)} configs) ---')

        print('\n  [training_set_effect]')
        make_training_set_effect_for_group(
            cfg_list, group_label, paper_plots_dir, single_group, quad_dir,
        )

        print('\n  [frac_problems_solved]')
        make_frac_solved_for_group(
            cfg_list, group_label, paper_plots_dir, single_group, quad_dir,
            args.recompute,
        )

        print('\n  [losses]')
        make_losses_for_group(
            cfg_list, group_label, paper_plots_dir, single_group, quad_dir,
            args.recompute,
        )

        print('\n  [times]')
        make_times_for_group(
            cfg_list, group_label, paper_plots_dir, single_group, quad_dir,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
