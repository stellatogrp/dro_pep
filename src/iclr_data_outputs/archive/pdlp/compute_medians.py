"""
PDLP median-vs-mean summary (no plots).

Mirrors the loss aggregation in `create_paper_plots.py:aggregate_losses_data`
but, instead of producing a figure, computes BOTH the mean and the median of the
per-image final Lagrangian gap for each (distribution, K, learning framework)
and writes the result to a markdown file.

Rationale: the paper `losses` plot reports the *mean* (+ [q10, q90] band). The
mean is heavily skewed by outliers, so this script reports the median alongside
it for the in-distribution (Olivetti faces) and out-of-distribution (color
image tiles) sets — for comparison only. No plots are produced.

Reuses the cached gap NPZs (`pdlp_in_gaps_K{K}_reps{R}.npz` /
`pdlp_ood_gaps_...npz`) written by `create_paper_plots.py`; nothing is
recomputed. If a cache is missing, run `python create_paper_plots.py` first.

Output: `paper_plots/median_summary.md`.

Run from the pdlp/ directory:
    python compute_medians.py
"""
from collections import defaultdict

import numpy as np

from create_paper_plots import (
    ARCH_DISPLAY_NAMES,
    ARCHS,
    K_VALS_LOSSES,
    PAPER_PLOTS_DIR,
    PDLP_DIR,
    _split_npz_path,
    load_split_gaps_npz,
)


# panel label -> npz split name used by create_paper_plots
PANELS = [
    ('test', 'in', 'In-distribution'),
    ('ood', 'ood', 'Out-of-distribution'),
]
METRIC_NAME = 'Lagrangian gap'


def collect_mean_median():
    """data[panel][arch] -> sorted list of (K, mean, median).

    Matches `aggregate_losses_data`: mean/median are the per-image statistics of
    the gap trajectory, indexed at each K in `K_VALS_LOSSES` (NUM_REPS == 1, so
    column K is the gap after K iterations).
    """
    data = {label: defaultdict(list) for label, _, _ in PANELS}

    for label, split, _ in PANELS:
        npz_path = _split_npz_path(split)
        if not npz_path.exists():
            raise FileNotFoundError(
                f'Missing cached gaps {npz_path.name}. Run '
                f'`python create_paper_plots.py` first to generate it.'
            )
        results, _ = load_split_gaps_npz(split)
        for arch in ARCHS:
            arr = np.asarray(results[arch])          # (n_images, K_total + 1)
            mean = arr.mean(axis=0)
            median = np.quantile(arr, 0.50, axis=0)
            for K in K_VALS_LOSSES:
                data[label][arch].append(
                    (K, float(mean[K]), float(median[K]))
                )

    return data


def write_markdown(data, md_path):
    lines = []
    lines.append('# PDLP — mean vs. median final Lagrangian gap')
    lines.append('')
    lines.append(
        f'Metric: `{METRIC_NAME}` after K iterations, per problem instance, '
        f'aggregated over the in-distribution (Olivetti faces) / '
        f'out-of-distribution (color image tiles) sets.'
    )
    lines.append(
        'Reuses the same cached gap trajectories as the `losses` paper plot. '
        'Medians are for comparison only — no plots are produced.'
    )
    lines.append('')

    for arch in ARCHS:
        disp = ARCH_DISPLAY_NAMES.get(arch, arch)
        lines.append(f'## {disp}')
        lines.append('')
        lines.append(
            '| K | In-dist mean | In-dist median | OOD mean | OOD median |'
        )
        lines.append('|---|---|---|---|---|')

        by_k = defaultdict(dict)
        for label, _, _ in PANELS:
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
    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_mean_median()
    md_path = PAPER_PLOTS_DIR / 'median_summary.md'
    write_markdown(data, md_path)
    print(f'Saved {md_path.relative_to(PDLP_DIR)}')


if __name__ == '__main__':
    main()
