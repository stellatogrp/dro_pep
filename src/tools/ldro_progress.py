"""Summarise DR-L2O training progress from a pulled iclr_data_outputs tree.

Safe to run against a campaign that is still going: unified_trainer rewrites
progress.csv in full after every SGD step, so each file is a consistent
snapshot up to the last completed iteration.

    python tools/ldro_progress.py                 # every experiment
    python tools/ldro_progress.py LogReg          # one experiment
    python tools/ldro_progress.py LogReg --date 2026-09-21

Columns:
    iters   rows in progress.csv, minus the initial-loss row = SGD steps done
    train   training loss at the last completed step
    best    lowest training loss seen, and the step it happened on -- the
            final value is routinely worse than the best, so this is what says
            whether the run had converged or was still improving when it
            stopped
    val     validation loss at the last completed step
    s/it    mean seconds per step over the last 10, for ETA
"""
import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

EXPECTED = 500  # sgd_iters across the ldro sweeps


def find_runs(root, exp=None, date=None):
    """Yield (exp, run_id, K, csv_path) for every progress.csv under root."""
    base = os.path.join(root, 'learn_dro_outputs')
    if not os.path.isdir(base):
        sys.exit(f'no learn_dro_outputs under {root} -- pull first?')
    for e in sorted(os.listdir(base)):
        if exp and e.lower() != exp.lower():
            continue
        for d in sorted(os.listdir(os.path.join(base, e))):
            if date and d != date:
                continue
            day = os.path.join(base, e, d)
            if not os.path.isdir(day):
                continue
            for run in sorted(os.listdir(day)):
                # <HH-MM-SS>_<array task id>. The timestamp is NOT decoration:
                # resubmitting a sweep the same day reuses task ids, so two
                # different arrays collide on <date>/<task> and only the launch
                # time tells them apart.
                stamp, _, task = run.rpartition('_')
                inner = os.path.join(day, run, 'learn_dro_outputs')
                if not os.path.isdir(inner):
                    continue
                for kdir in sorted(os.listdir(inner)):
                    if not kdir.startswith('K_'):
                        continue
                    csv = os.path.join(inner, kdir, 'progress.csv')
                    if os.path.exists(csv):
                        yield e, f'{d} {stamp} #{task}', int(kdir[2:]), csv


def _num(v, w=11):
    """Right-aligned fixed-width number, or blank padding when absent."""
    return f'{v:>{w}.6f}' if v is not None else ' ' * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', nargs='?', help='Quad | Lasso | LogReg | PDLP')
    ap.add_argument('--date', help='restrict to one YYYY-MM-DD output day')
    ap.add_argument('--root', default='iclr_data_outputs')
    ap.add_argument('--expected', type=int, default=EXPECTED)
    a = ap.parse_args()

    rows = []
    for exp, run, K, csv in find_runs(a.root, a.exp, a.date):
        try:
            df = pd.read_csv(csv)
        except Exception as err:            # caught mid-write by rsync
            rows.append((exp, run, K, -1, f'unreadable ({type(err).__name__})'))
            continue
        # Row 0 is the pre-training initial loss, not an SGD step.
        iters = max(len(df) - 1, 0)
        tr = df.get('training_loss')
        va = df.get('validation_loss')
        it = df.get('iter_time')
        best = bi = None
        if tr is not None and len(tr.dropna()):
            best = tr.min()
            bi = int(tr.idxmin())
        rows.append((
            exp, run, K, iters,
            None if tr is None or not len(tr) else tr.iloc[-1],
            best, bi,
            None if va is None or not len(va) else va.iloc[-1],
            None if it is None or not len(it) else it.tail(10).mean(),
        ))

    if not rows:
        sys.exit('no progress.csv found -- check --root/--date')

    per_exp = defaultdict(list)
    for r in rows:
        per_exp[r[0]].append(r)

    for exp in sorted(per_exp):
        print(f'\n=== {exp} ===')
        print(f'{"run (date time #task)":>26} {"K":>3} {"iters":>9} {"train":>11} '
              f'{"best (@it)":>17} {"val":>11} {"s/it":>7}  eta')
        for r in sorted(per_exp[exp], key=lambda x: (x[2], x[1])):
            if r[3] < 0:
                print(f'{r[1]:>26} {r[2]:>3}   {r[4]}')
                continue
            _, run, K, iters, tr, best, bi, va, sit = r
            frac = f'{iters}/{a.expected}'
            eta = ''
            if sit and iters < a.expected:
                left = (a.expected - iters) * sit
                eta = f'{left/3600:.1f}h' if left >= 3600 else f'{left/60:.0f}m'
            elif iters >= a.expected:
                eta = 'done'
            bs = f'{best:.6f} @{bi}' if best is not None else ''
            print(f'{run:>26} {K:>3} {frac:>9} {_num(tr)} {bs:>17} {_num(va)} '
                  f'{sit if sit else 0:>7.2f}  {eta}')

    done = sum(1 for r in rows if r[3] >= a.expected)
    print(f'\n{done}/{len(rows)} (K, task) pairs have reached '
          f'{a.expected} iterations.')


if __name__ == '__main__':
    main()
