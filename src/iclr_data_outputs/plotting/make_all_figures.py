#!/usr/bin/env python3
"""Regenerate every paper figure, then collect the 8 that go in the paper.

    python make_all_figures.py              # build all four, then collect
    python make_all_figures.py logreg       # just the LogReg figures
    python make_all_figures.py --collect    # collect only, skip the rebuild

quad / lasso / pdlp run the ORIGINAL, validated plotting code, copied verbatim
into archive/<problem>/ alongside its data (only the src/ path depth was
adjusted, since the scripts resolve it relative to their own location). Running
them here reproduces the paper figures exactly -- verified: pdlp_losses.csv,
pdlp_times.csv, quad_times.csv and lasso_times.csv all come back identical to
the committed originals.

Those three write into archive/<problem>/paper_plots/; LogReg writes into
figures/, because it is new work rather than a reproduction. The collect step
then copies the 8 paper figures -- a losses panel and a frac-solved grid for
each of the four experiments -- into figures/paper_figures/ unchanged, so they
can be dropped into the paper source from one directory.

LogReg is FGM only. plot_logreg_gd.py still works and is still a target
(`make_all_figures.py logreg_gd`), but GD is not part of the paper set, so it
is neither built by default nor collected.

quad/lasso use cached CSVs by default; pass -recompute to the individual
scripts to rebuild from problem_instances/ and the best-stepsize schedules.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICLR_OUT = HERE.parent
ARCHIVE = ICLR_OUT / 'archive'
FIGURES = ICLR_OUT / 'figures'
PAPER_FIGURES = FIGURES / 'paper_figures'

# name -> (working directory, command)
TARGETS = {
    'quad':  (ARCHIVE / 'quad',  [sys.executable, 'create_paper_plots.py']),
    'lasso': (ARCHIVE / 'lasso', [sys.executable, 'create_paper_plots.py']),
    'pdlp':  (ARCHIVE / 'pdlp',  [sys.executable, 'create_paper_plots.py']),
    'logreg_fgm': (HERE, [sys.executable, 'plot_logreg_fgm.py']),
    # Not in DEFAULT_TARGETS: the paper uses FGM only for LogReg.
    'logreg_gd':  (HERE, [sys.executable, 'plot_logreg_gd.py']),
}
DEFAULT_TARGETS = ['quad', 'lasso', 'pdlp', 'logreg_fgm']
GROUPS = {'logreg': ['logreg_fgm'],
          'paper': ['quad', 'lasso', 'pdlp']}

# The 8 paper figures: source path -> name inside figures/paper_figures/.
# Names are kept as-is so a \includegraphics in the paper source does not have
# to change depending on whether it points here or at the archive.
PAPER_SET = [
    (ARCHIVE / 'quad'  / 'paper_plots' / 'quad_losses.pdf',                 'quad_losses.pdf'),
    (ARCHIVE / 'quad'  / 'paper_plots' / 'quad_frac_problems_solved.pdf',   'quad_frac_problems_solved.pdf'),
    (ARCHIVE / 'lasso' / 'paper_plots' / 'lasso_losses.pdf',                'lasso_losses.pdf'),
    (ARCHIVE / 'lasso' / 'paper_plots' / 'lasso_frac_problems_solved.pdf',  'lasso_frac_problems_solved.pdf'),
    (ARCHIVE / 'lasso' / 'paper_plots' / 'lasso_intro.pdf',                 'lasso_intro.pdf'),
    (ARCHIVE / 'pdlp'  / 'paper_plots' / 'pdlp_losses.pdf',                 'pdlp_losses.pdf'),
    (ARCHIVE / 'pdlp'  / 'paper_plots' / 'pdlp_frac_problems_solved.pdf',   'pdlp_frac_problems_solved.pdf'),
    (FIGURES / 'logreg_fgm_losses.pdf',                                     'logreg_losses.pdf'),
    (FIGURES / 'logreg_fgm_frac_problems_solved.pdf',                       'logreg_frac_problems_solved.pdf'),
]


def expand(names):
    if not names:
        return list(DEFAULT_TARGETS)
    out = []
    for n in names:
        if n in GROUPS:
            out += GROUPS[n]
        elif n in TARGETS:
            out.append(n)
        else:
            raise SystemExit(
                f"unknown target '{n}'; choose from "
                f"{sorted(TARGETS)} or groups {sorted(GROUPS)}")
    return out


def collect():
    """Copy the 8 paper figures into figures/paper_figures/, byte-for-byte.

    copy2 rather than move: the generators write into their own output dirs and
    re-running one must not depend on whether a previous collect emptied them.
    Missing sources are reported and counted, not raised, so a partial rebuild
    still tells you exactly which figure did not get made.
    """
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    missing = []
    for src, name in PAPER_SET:
        if not src.is_file():
            print(f'  MISSING  {src.relative_to(ICLR_OUT)}')
            missing.append(name)
            continue
        shutil.copy2(src, PAPER_FIGURES / name)
        print(f'  {name}')
    return missing


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('targets', nargs='*',
                   help=f'{sorted(TARGETS)} or groups {sorted(GROUPS)}; '
                        f'default: {DEFAULT_TARGETS}')
    p.add_argument('--collect', action='store_true',
                   help='skip the rebuild and only copy the 8 paper figures')
    p.add_argument('--no-collect', action='store_true',
                   help='rebuild only; leave figures/paper_figures/ alone')
    args = p.parse_args()

    failures = []
    if not args.collect:
        for name in expand(args.targets):
            cwd, cmd = TARGETS[name]
            if not cwd.is_dir():
                print(f'== {name}: SKIP ({cwd} missing)')
                failures.append(name)
                continue
            print(f'\n===== {name} =====')
            r = subprocess.run(cmd, cwd=cwd)
            if r.returncode != 0:
                failures.append(name)

    missing = []
    if not args.no_collect:
        print(f'\n===== collect -> {PAPER_FIGURES} =====')
        missing = collect()

    print('\n' + '=' * 60)
    if failures:
        print('FAILED:', ', '.join(failures))
    if missing:
        print('NOT COLLECTED:', ', '.join(missing))
    if failures or missing:
        return 1
    if not args.collect:
        print('all figures regenerated')
    print(f'  paper set ({len(PAPER_SET)} figures): {PAPER_FIGURES}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
