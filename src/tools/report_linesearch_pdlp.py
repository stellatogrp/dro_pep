#!/usr/bin/env python3
"""Plot PDLP backtracking against the archived paper Lagrangian gaps.

Same layout as ``report_linesearch_audit.curves``: mean gap vs iterations (top)
and vs matrix-product budget (bottom), in-distribution and OOD columns.
No simulations occur here; input is ``run_linesearch_pdlp.py`` output.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SERIES = [('DR-L2O', 'DR-L2O'), ('OPT-PEP', 'OPT-PEP'),
          ('backtracking', 'Backtracking, PDLP step condition (unit reset)')]
COLORS = ['#487ca5', '#83a17d', '#c59858']


def curves(frame, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3))
    for row, column in enumerate(['mean', 'equal_matvec_mean']):
        for col, split in enumerate(['test', 'ood']):
            ax = axes[row, col]
            dat = frame[frame.split == split]
            for j, (key, label) in enumerate(SERIES):
                d = dat[dat.label == key].sort_values('K')
                x = d.K if row == 0 else 2 * d.K
                ax.plot(x, d[column], color=COLORS[j], label=label, linewidth=1.8,
                        marker='o' if j < 2 else None, markersize=2.5,
                        linestyle='-' if j < 2 else '--')
            ax.set_yscale('log')
            ax.grid(alpha=.18)
            n = int(dat.n.iloc[0])
            ax.set_title(('In-distribution' if split == 'test' else 'Out-of-distribution')
                         + (f' ({n} instances)' if row == 0 else ''))
            ax.set_xlabel(r'Iterations $K$' if row == 0 else 'Matrix-product budget')
            if col == 0:
                ax.set_ylabel('Mean Lagrangian gap')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('TV inpainting / PDHG', y=.99)
    fig.tight_layout(rect=(0, .09, 1, .965))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'pdlp.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'results' / 'linesearch-pdlp')
    ap.add_argument('--output', type=Path, default=ROOT / 'results' / 'linesearch-pdlp' / 'plots')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input / 'pdlp' / 'summary.csv')
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
                         'axes.titlesize': 11, 'axes.labelsize': 11, 'legend.fontsize': 9})
    curves(frame, args.output)
    k10 = frame[frame.K == 10].set_index(['split', 'label'])
    rows = []
    for split in ['test', 'ood']:
        dr, bt = k10.loc[(split, 'DR-L2O')], k10.loc[(split, 'backtracking')]
        rows.append(dict(split=split, dr_gap=dr['mean'], bt_gap=bt['mean'],
                         bt_over_dr=bt['mean'] / dr['mean'],
                         bt_gap_20_products=frame[(frame.split == split) & (frame.label == 'backtracking')
                                                  & (frame.K == 10)].equal_matvec_mean.iloc[0],
                         bt_products=bt.matvec_mean))
    pd.DataFrame(rows).to_csv(args.output / 'decision.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
