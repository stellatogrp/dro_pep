"""
Lasso median-vs-mean summary (no plots).

Mirrors the loss aggregation in `losses.py:collect_losses_data` but, instead of
producing a figure, computes BOTH the mean and the median of the per-instance
final-iterate loss for each (distribution, K, learning framework) and writes the
result to a markdown file.

Rationale: the paper `losses` plot reports the *mean* (+ [q10, q90] band). The
mean is heavily skewed by outliers, so this script reports the median alongside
it for the in-distribution (test) and out-of-distribution sets — for comparison
only. No plots are produced.

Output: `paper_plots/median_summary.md`.

Run from the lasso/ directory:
    python compute_medians.py
"""
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from create_paper_plots import (
    ALL_LASSO_CONFIGS,
    find_config_with_N,
    group_key_for_config,
    parse_config_list_to_dict,
    slugify_group_key,
)
from create_test_plots import (
    ARCH_DISPLAY_NAMES,
    ARCH_TO_CSV,
    compute_loss_trajectory,
    load_best_stepsize,
    load_ood_test_data,
    load_test_data,
)
from data_scrape import get_leaf_path
from losses import K_VALS_LOSSES, TRAINING_SAMPLE_N_FOR_LOSSES


PANELS = [
    ('test', 'In-distribution'),
    ('ood', 'Out-of-distribution'),
]
METRIC_NAME = r'f(x^K) - f(x^\star)'


def collect_mean_median(target_cfg, K_vals):
    """data[panel][arch] -> sorted list of (K, mean, median)."""
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

    data = {label: defaultdict(list) for label, _ in PANELS}

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
                data[label][arch].append(
                    (K, float(final.mean()), float(np.median(final)))
                )
        print(f'    K={K}: done')

    for panel_label in data:
        for arch in data[panel_label]:
            data[panel_label][arch].sort(key=lambda p: p[0])

    return data


def write_markdown(data, training_sample_N, md_path):
    lines = []
    lines.append('# Lasso — mean vs. median final-iterate loss')
    lines.append('')
    lines.append(
        f'Metric: `{METRIC_NAME}` at the final iterate, per problem instance, '
        f'aggregated over the test / OOD batches.'
    )
    lines.append(
        f'Training sample N = {training_sample_N}. Each K uses that K\'s own '
        f'best learned schedule (same source as the `losses` paper plot). '
        f'Medians are for comparison only — no plots are produced.'
    )
    lines.append('')

    archs = sorted({a for panel in data.values() for a in panel})
    for arch in archs:
        disp = ARCH_DISPLAY_NAMES.get(arch, arch)
        lines.append(f'## {disp}')
        lines.append('')
        lines.append(
            '| K | In-dist mean | In-dist median | OOD mean | OOD median |'
        )
        lines.append('|---|---|---|---|---|')

        by_k = defaultdict(dict)
        for label, _ in PANELS:
            for K, mean, median in data[label].get(arch, []):
                by_k[K][label] = (mean, median)

        for K in sorted(by_k):
            tm, tmd = by_k[K].get('test', (float('nan'), float('nan')))
            om, omd = by_k[K].get('ood', (float('nan'), float('nan')))
            lines.append(
                f'| {K} | {tm:.6e} | {tmd:.6e} | {om:.6e} | {omd:.6e} |'
            )
        lines.append('')

    md_path.write_text('\n'.join(lines))


def main():
    lasso_dir = Path(__file__).parent
    paper_plots_dir = lasso_dir / 'paper_plots'
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for cfg in ALL_LASSO_CONFIGS:
        groups[group_key_for_config(cfg)].append(cfg)
    single_group = (len(groups) == 1)

    for key, cfg_list in groups.items():
        group_label = slugify_group_key(key)
        target_cfg = find_config_with_N(cfg_list, TRAINING_SAMPLE_N_FOR_LOSSES)
        if target_cfg is None:
            print(f'  group {group_label}: no config with N='
                  f'{TRAINING_SAMPLE_N_FOR_LOSSES}, skipping.')
            continue

        print(f'--- group: {group_label} ---')
        data = collect_mean_median(target_cfg, K_VALS_LOSSES)

        name = 'median_summary.md' if single_group \
            else f'median_summary_{group_label}.md'
        md_path = paper_plots_dir / name
        write_markdown(data, TRAINING_SAMPLE_N_FOR_LOSSES, md_path)
        print(f'  Saved {md_path.relative_to(lasso_dir)}')


if __name__ == '__main__':
    main()
