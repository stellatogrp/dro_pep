"""Appendix figures: learned schedules vs standard backtracking line search.

2x2 grid in the style of the paper's ``*_losses.pdf`` figures. Top row: mean
final-iterate loss vs horizon K. Bottom row: mean loss at an equal
matrix-vector product budget (a learned schedule costs 2 products per
iteration; backtracking returns the last accepted iterate within the budget).
Shading is the [q10, q90] band over instances. Logistic regression gets only
the top row (1x2): its backtracking reuses A g across trials, so it also costs 2
products per iteration and the budget row would repeat the iteration row.

Input is the per-instance output of ``src/tools/run_linesearch_audit.py
--mode backtracking`` (``<results>/<problem>/raw/*.npz``). The learned curves
are checked against the paper figure CSVs before plotting.

    python src/iclr_data_outputs/plotting/plot_backtracking.py \
        --input results/linesearch-backtracking

Writes the PDFs and the long-form ``backtracking_curves.csv`` (read by
``make_paper_tables.py``) to ``figures/backtracking/``.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (ARCH_COLORS, ARCH_DISPLAY_NAMES, ARCH_MARKERS,  # noqa: E402
                    BASELINE_COLOR, use_paper_style)

ICLR_DIR = Path(__file__).resolve().parents[1]
ARCHS = ['ldro_pep', 'lpep']
PANELS = [('test', 'In-distribution'), ('ood', 'Out-of-distribution')]
K_MAX = 15
X_TICKS = [2, 4, 6, 8, 10, 12, 14]
BUDGET_TICKS = [4, 8, 12, 16, 20, 24, 28]
BT_LABEL = 'Backtracking'
BUDGET_ROW = {'lasso': True, 'logreg_fgm': False, 'logreg_gd': False}
# Paper figure files; logistic regression in the paper is FGM.
OUT_NAMES = {'lasso': 'lasso_backtracking', 'logreg_fgm': 'logreg_backtracking',
             'logreg_gd': 'logreg_gd_backtracking'}
PAPER_CSVS = {'lasso': ICLR_DIR / 'archive' / 'lasso' / 'paper_plots' / 'lasso_losses.csv',
              'logreg_fgm': ICLR_DIR / 'figures' / 'logreg_fgm_losses.csv',
              'logreg_gd': ICLR_DIR / 'figures' / 'logreg_gd_losses.csv'}


def stats(x):
    return x.mean(), np.quantile(x, 0.1), np.quantile(x, 0.9), np.quantile(x, 0.5)


def load_curves(raw, split):
    """Per-series (x, mean, q10, q90, median) for the iteration and budget rows."""
    curves = {}
    for arch in ARCHS:
        it, bud = [], []
        for K in range(1, K_MAX + 1):
            z = np.load(raw / f'{arch}_K{K}_{split}.npz')
            assert np.all(z['matvec'][:, K] == 2 * K), (arch, K)
            s = stats(z['losses'][:, K])
            it.append((K, *s))
            bud.append((2 * K, *s))
        curves[arch] = (np.array(it).T, np.array(bud).T)
    z = np.load(raw / f'backtracking_{split}.npz')
    losses, matvec = z['losses'], z['matvec']
    it = [(K, *stats(losses[:, K])) for K in range(1, K_MAX + 1)]
    bud = []
    for K in range(1, K_MAX + 1):
        ok = matvec <= 2 * K
        idx = np.maximum.accumulate(np.where(ok, np.arange(K_MAX + 1), 0), axis=1)[:, -1]
        bud.append((2 * K, *stats(losses[np.arange(len(idx)), idx])))
    curves['backtracking'] = (np.array(it).T, np.array(bud).T)
    return curves


def check_learned(problem, curves_by_split):
    ref = pd.read_csv(PAPER_CSVS[problem])
    if problem == 'lasso':
        ref = ref.rename(columns={'row': 'split', 'final_loss_mean': 'mean',
                                  'final_loss_q10': 'q10', 'final_loss_q90': 'q90'})
    else:
        ref['arch'] = ref.arch.map({v: k for k, v in ARCH_DISPLAY_NAMES.items()})
    worst = 0.0
    for split, curves in curves_by_split.items():
        for arch in ARCHS:
            (K, mean, q10, q90, _med), _ = curves[arch]
            r = ref[(ref.arch == arch) & (ref.split == split)].sort_values('K')
            assert np.array_equal(r.K.to_numpy(), K.astype(int)), (problem, split, arch)
            for got, col in [(mean, 'mean'), (q10, 'q10'), (q90, 'q90')]:
                np.testing.assert_allclose(got, r[col], rtol=1e-8)
                worst = max(worst, float(np.max(np.abs(got / r[col].to_numpy() - 1))))
    return worst


def make_figure(problem, curves_by_split, out_dir):
    log_floor = 1e-30
    series = [(arch, ARCH_COLORS[arch], ARCH_MARKERS[arch], '-', ARCH_DISPLAY_NAMES[arch])
              for arch in ARCHS] + [('backtracking', BASELINE_COLOR, 'v', '--', BT_LABEL)]
    with plt.rc_context({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
                         'legend.fontsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10}):
        n_rows = 2 if BUDGET_ROW[problem] else 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(7, 5.3 if n_rows == 2 else 2.8), squeeze=False)
        for col, (split, title) in enumerate(PANELS):
            for row in range(n_rows):
                ax = axes[row, col]
                for key, color, marker, ls, label in series:
                    x, mean, q10, q90, _med = curves_by_split[split][key][row]
                    ax.plot(x, mean, color=color, marker=marker, markersize=5,
                            linestyle=ls, label=label)
                    ax.fill_between(x, np.maximum(q10, log_floor), q90,
                                    color=color, alpha=0.2, linewidth=0)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(title)
                    ax.set_xlabel(r'$K$')
                    ax.set_xticks(X_TICKS)
                else:
                    ax.set_xlabel('Matrix-vector products')
                    ax.set_xticks(BUDGET_TICKS)
        axes[0, 0].set_ylabel(r'Avg. $f(x^K) - f(x^\star)$')
        if n_rows == 2:
            axes[1, 0].set_ylabel(r'Avg. $f(x) - f(x^\star)$')
        fig.tight_layout()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                   ncol=len(handles), frameon=True)
        pdf = out_dir / f'{OUT_NAMES[problem]}.pdf'
        fig.savefig(pdf, bbox_inches='tight')
        fig.savefig(pdf.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    return pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ICLR_DIR / 'figures' / 'backtracking')
    ap.add_argument('--problems', nargs='+', default=['lasso', 'logreg_fgm', 'logreg_gd'])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    use_paper_style()
    rows = []
    for problem in args.problems:
        raw = args.input / problem / 'raw'
        curves = {split: load_curves(raw, split) for split, _ in PANELS}
        err = check_learned(problem, curves)
        pdf = make_figure(problem, curves, args.output)
        for split, c in curves.items():
            for key in ARCHS + ['backtracking']:
                for row, xname in enumerate(['K', 'matvec_budget']):
                    for x, m, lo, hi, med in zip(*c[key][row]):
                        rows.append(dict(problem=problem, split=split, method=key, axis=xname,
                                         x=int(x), mean=m, q10=lo, median=med, q90=hi))
        print(f'{problem}: learned curves match paper CSV (max rel err {err:.1e}) -> {pdf}')
    pd.DataFrame(rows).to_csv(args.output / 'backtracking_curves.csv', index=False)


if __name__ == '__main__':
    main()
