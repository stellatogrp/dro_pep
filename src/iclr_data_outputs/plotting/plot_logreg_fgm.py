#!/usr/bin/env python3
"""Paper-style LogReg (german.numer) figures for FGM.

Writes into iclr_data_outputs/figures/:
    logreg_fgm_losses.pdf / .csv                  in-dist | OOD, mean + [q10,q90]
    logreg_fgm_frac_problems_solved.pdf / .csv    2 x 3 grid over eta
    logreg_fgm_times.csv                          mean +/- 2 sigma per SGD iter

The three series are L2O / DR-L2O / OPT-PEP in the paper's colours and markers
-- no handcrafted reference curve, matching the quad/lasso/pdlp losses figures.
DR-L2O covers K = 1..15; the L2O and OPT-PEP baselines cover only K = {5, 10,
15}; see logreg_figures.py for the full rationale.

Usage (from anywhere):
    python plot_logreg_fgm.py
    python plot_logreg_fgm.py --data-dir DIR --runs-root DIR --out-dir DIR
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logreg_figures


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs-root', default=None,
                   help='dir holding learn_{dro,l2o,lpep}_outputs '
                        '(default: iclr_data_outputs)')
    p.add_argument('--data-dir', default=None,
                   help='LogReg sample-creation dir with test_set.npz / '
                        'ood_set.npz (default: the newest one)')
    p.add_argument('--out-dir', default=None,
                   help='default: iclr_data_outputs/figures')
    args = p.parse_args()
    logreg_figures.build('fgm', runs_root=args.runs_root, data_dir=args.data_dir,
                         out_dir=args.out_dir)


if __name__ == '__main__':
    main()
