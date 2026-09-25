#!/usr/bin/env python3
"""Build the paper's appendix result tables from the plot CSVs.

    python make_paper_tables.py                 # all four experiments, then collect
    python make_paper_tables.py lasso pdlp      # a subset
    python make_paper_tables.py --collect       # collect only, skip the rebuild

For every experiment this writes two CSVs shaped like the `*_times.csv` tables
the paper already typesets with pgfplotstable (Framework filled on a
framework's first row and blank below it, one row per K, every K):

    <exp>_losses_table.csv
        Framework,K,IDmean,IDq10,IDmed,IDq90,OODmean,OODq10,OODmed,OODq90
        the mean, 10th quantile, median and 90th quantile of the final-iterate
        test loss on the in-distribution (ID) and out-of-distribution (OOD)
        sets, i.e. the two panels of <exp>_losses.pdf, to 3 significant
        figures in compact scientific notation (1.23e-4).
    <exp>_frac_solved_table.csv
        Framework,K,IDeta1,IDeta2,IDeta3,OODeta1,OODeta2,OODeta3
        the fraction of test instances solved at each tolerance eta (ascending,
        the column order of <exp>_frac_problems_solved.pdf), to 3 decimals.

The numbers are read from the long-form CSVs the plotting scripts cache next
to their PDFs, so a table row is exactly what the figure plots. Nothing is
recomputed here: no JAX, no matplotlib (the archived scripts switch on
text.usetex at import, which this script must not do). The median column
requires losses CSVs written after the `*_median` column was added; rerun
`make_all_figures.py` after deleting a stale `<exp>_losses.csv` if it is
missing.

Backtracking line search (plot_backtracking.py -> figures/backtracking/
backtracking_curves.csv): the logreg losses table gets a Backtracking block
(its budget and iteration curves coincide), and lasso gets a separate
lasso_backtracking_table.csv with an iteration block and a matrix-vector
product budget block (Budget,x,IDmean,...,OODq90).

quad / lasso / pdlp tables are written into archive/<exp>/paper_plots/, LogReg
into figures/ (as logreg_fgm_*), mirroring where their sources live. The
collect step copies all eight into figures/paper_tables/ under the names the
paper source uses (tables/<exp>_losses.csv, tables/<exp>_frac_solved.csv).
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ICLR_OUT = HERE.parent
ARCHIVE = ICLR_OUT / 'archive'
FIGURES = ICLR_OUT / 'figures'
PAPER_TABLES = FIGURES / 'paper_tables'
BACKTRACKING_CURVES = FIGURES / 'backtracking' / 'backtracking_curves.csv'

# (value in the split column, column prefix in the output tables)
SPLITS = [('test', 'ID'), ('ood', 'OOD')]
STATS = [('mean', 'mean'), ('q10', 'q10'), ('median', 'med'), ('q90', 'q90')]

# Framework order and display names follow the paper's times tables
# (archive/<exp>/times.py FRAMEWORK_ORDER / ARCH_DISPLAY_NAMES).
_QL_STATS = {'mean': 'final_loss_mean', 'q10': 'final_loss_q10',
             'median': 'final_loss_median', 'q90': 'final_loss_q90'}

EXPERIMENTS = {
    'quad': dict(
        losses=ARCHIVE / 'quad' / 'paper_plots' / 'quad_losses.csv',
        frac=ARCHIVE / 'quad' / 'paper_plots' / 'quad_frac_problems_solved.csv',
        out_dir=ARCHIVE / 'quad' / 'paper_plots', prefix='quad',
        split_col='row', stat_cols=_QL_STATS,
        frameworks=[('l2o', 'L2O'), ('ldro_pep', 'DR-L2O'), ('lpep', 'OPT-PEP')],
        K_vals=list(range(1, 16)), etas=[1e-3, 1e-2, 1e-1],
    ),
    'lasso': dict(
        losses=ARCHIVE / 'lasso' / 'paper_plots' / 'lasso_losses.csv',
        frac=ARCHIVE / 'lasso' / 'paper_plots' / 'lasso_frac_problems_solved.csv',
        out_dir=ARCHIVE / 'lasso' / 'paper_plots', prefix='lasso',
        split_col='row', stat_cols=_QL_STATS,
        frameworks=[('l2o', 'L2O-ISTA'), ('l2o_alista', 'L2O-ALISTA'),
                    ('ldro_pep', 'DR-L2O'), ('lpep', 'OPT-PEP')],
        K_vals=list(range(1, 16)), etas=[1e-2, 5e-2, 1e-1],
    ),
    'pdlp': dict(
        losses=ARCHIVE / 'pdlp' / 'paper_plots' / 'pdlp_losses.csv',
        frac=ARCHIVE / 'pdlp' / 'paper_plots' / 'pdlp_frac_problems_solved.csv',
        out_dir=ARCHIVE / 'pdlp' / 'paper_plots', prefix='pdlp',
        split_col='row',
        stat_cols={'mean': 'gap_mean', 'q10': 'gap_q10',
                   'median': 'gap_median', 'q90': 'gap_q90'},
        frameworks=[('l2o', 'L2O'), ('ldro_pep', 'DR-L2O'), ('lpep', 'OPT-PEP')],
        K_vals=list(range(1, 11)), etas=[1e-2, 5e-2, 1e-1],
    ),
    'logreg': dict(
        losses=FIGURES / 'logreg_fgm_losses.csv',
        frac=FIGURES / 'logreg_fgm_frac_problems_solved.csv',
        out_dir=FIGURES, prefix='logreg_fgm',
        split_col='split',
        stat_cols={'mean': 'mean', 'q10': 'q10', 'median': 'median', 'q90': 'q90'},
        # logreg_figures.py already writes display names into `arch`.
        frameworks=[('L2O', 'L2O'), ('DR-L2O', 'DR-L2O'), ('OPT-PEP', 'OPT-PEP')],
        K_vals=list(range(1, 16)), etas=[1e-4, 1e-3, 1e-2],
        backtracking='logreg_fgm',
    ),
}
DEFAULT_TARGETS = ['quad', 'lasso', 'pdlp', 'logreg']


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_loss(x):
    """3 significant figures, compact exponent: 1.23e-4, 9.36e1, 5.01e-2."""
    x = float(x)
    if not math.isfinite(x):
        return '--'
    if x == 0.0:
        return '0'
    mant, exp = f'{x:.2e}'.split('e')
    return f'{mant}e{int(exp)}'


def format_frac(x):
    """Fractions of 250 (or 40/80 for pdlp) instances: 3 decimals is exact."""
    x = float(x)
    if not math.isfinite(x):
        return '--'
    return f'{x:.3f}'


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def _one(df, mask, what):
    sub = df[mask]
    if len(sub) != 1:
        raise SystemExit(f'expected exactly one row for {what}, found {len(sub)}')
    return sub.iloc[0]


def losses_table(spec):
    df = pd.read_csv(spec['losses'])
    missing = [c for c in spec['stat_cols'].values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{spec['losses']} lacks {missing}; delete it and rerun "
            f"make_all_figures.py so the losses cache is rebuilt with the "
            f"median column")
    rows = []
    for arch, name in spec['frameworks']:
        for i, K in enumerate(spec['K_vals']):
            out = {'Framework': name if i == 0 else '', 'K': K}
            for split_val, split_pre in SPLITS:
                r = _one(df, (df[spec['split_col']] == split_val)
                         & (df['arch'] == arch) & (df['K'] == K),
                         f'losses {split_val}/{arch}/K={K}')
                for stat, stat_pre in STATS:
                    out[f'{split_pre}{stat_pre}'] = format_loss(
                        r[spec['stat_cols'][stat]])
            rows.append(out)
    if spec.get('backtracking'):
        rows.extend(backtracking_rows(spec['backtracking'], 'K', 'Backtracking',
                                      spec['K_vals'], 'Framework'))
    return pd.DataFrame(rows)


def backtracking_rows(problem, axis, label, xs, label_col):
    """One block of backtracking rows from backtracking_curves.csv."""
    if not BACKTRACKING_CURVES.is_file():
        raise SystemExit(f'{BACKTRACKING_CURVES} missing; run plot_backtracking.py')
    df = pd.read_csv(BACKTRACKING_CURVES)
    rows = []
    for i, x in enumerate(xs):
        out = {label_col: label if i == 0 else '', 'K' if label_col == 'Framework' else 'x': x}
        for split_val, split_pre in SPLITS:
            r = _one(df, (df.problem == problem) & (df.method == 'backtracking')
                     & (df.axis == axis) & (df.x == x) & (df.split == split_val),
                     f'backtracking {problem}/{axis}/{split_val}/x={x}')
            for stat, stat_pre in STATS:
                out[f'{split_pre}{stat_pre}'] = format_loss(r[stat])
        rows.append(out)
    return rows


def lasso_backtracking_table():
    """Iteration block (x = K) then product-budget block (x = 2K)."""
    Ks = list(range(1, 16))
    rows = backtracking_rows('lasso', 'K', 'Iterations', Ks, 'Budget')
    rows += backtracking_rows('lasso', 'matvec_budget', 'Products', [2 * K for K in Ks], 'Budget')
    return pd.DataFrame(rows)


def frac_table(spec):
    df = pd.read_csv(spec['frac'])
    etas = sorted(spec['etas'])
    for eta in etas:
        if not np.isclose(df['eta'].to_numpy(), eta).any():
            raise SystemExit(f"eta={eta} not in {spec['frac']}")
    rows = []
    for arch, name in spec['frameworks']:
        for i, K in enumerate(spec['K_vals']):
            out = {'Framework': name if i == 0 else '', 'K': K}
            for split_val, split_pre in SPLITS:
                for j, eta in enumerate(etas, start=1):
                    r = _one(df, (df[spec['split_col']] == split_val)
                             & (df['arch'] == arch) & (df['K'] == K)
                             & np.isclose(df['eta'].to_numpy(), eta),
                             f'frac {split_val}/{arch}/K={K}/eta={eta}')
                    out[f'{split_pre}eta{j}'] = format_frac(r['frac_solved'])
            rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(exp):
    spec = EXPERIMENTS[exp]
    n_expected = (len(spec['frameworks']) + bool(spec.get('backtracking'))) * len(spec['K_vals'])
    n_frac = len(spec['frameworks']) * len(spec['K_vals'])
    out_dir = spec['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    lt = losses_table(spec)
    assert len(lt) == n_expected, (exp, len(lt), n_expected)
    lt_path = out_dir / f"{spec['prefix']}_losses_table.csv"
    lt.to_csv(lt_path, index=False)

    ft = frac_table(spec)
    assert len(ft) == n_frac, (exp, len(ft), n_frac)
    ft_path = out_dir / f"{spec['prefix']}_frac_solved_table.csv"
    ft.to_csv(ft_path, index=False)

    if exp == 'lasso':
        bt = lasso_backtracking_table()
        bt_path = out_dir / 'lasso_backtracking_table.csv'
        bt.to_csv(bt_path, index=False)
        print(f'  {bt_path.relative_to(ICLR_OUT)}  ({len(bt)} rows)')

    etas = ', '.join(f'{e:g}' for e in sorted(spec['etas']))
    print(f'  {lt_path.relative_to(ICLR_OUT)}  ({len(lt)} rows)')
    print(f'  {ft_path.relative_to(ICLR_OUT)}  ({len(ft)} rows; '
          f'eta1..eta{len(spec["etas"])} = {etas})')


def collect():
    """Copy the 9 tables into figures/paper_tables/ under the paper's names."""
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    missing = []
    for exp, spec in EXPERIMENTS.items():
        for kind in ('losses', 'frac_solved'):
            src = spec['out_dir'] / f"{spec['prefix']}_{kind}_table.csv"
            name = f'{exp}_{kind}.csv'
            if not src.is_file():
                print(f'  MISSING  {src.relative_to(ICLR_OUT)}')
                missing.append(name)
                continue
            shutil.copy2(src, PAPER_TABLES / name)
            print(f'  {name}')
    src = EXPERIMENTS['lasso']['out_dir'] / 'lasso_backtracking_table.csv'
    if src.is_file():
        shutil.copy2(src, PAPER_TABLES / 'lasso_backtracking.csv')
        print('  lasso_backtracking.csv')
    else:
        missing.append('lasso_backtracking.csv')
    return missing


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('targets', nargs='*',
                   help=f'{DEFAULT_TARGETS}; default: all')
    p.add_argument('--collect', action='store_true',
                   help='skip the rebuild and only copy the 8 tables')
    p.add_argument('--no-collect', action='store_true',
                   help='rebuild only; leave figures/paper_tables/ alone')
    args = p.parse_args()

    targets = args.targets or DEFAULT_TARGETS
    unknown = [t for t in targets if t not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f'unknown target(s) {unknown}; choose from {DEFAULT_TARGETS}')

    if not args.collect:
        for exp in targets:
            print(f'\n===== {exp} =====')
            build(exp)

    missing = []
    if not args.no_collect:
        print(f'\n===== collect -> {PAPER_TABLES} =====')
        missing = collect()

    if missing:
        print('NOT COLLECTED:', ', '.join(missing))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
