#!/usr/bin/env python
"""Run the coarse line-search baseline on the paper's test / OOD instance sets.

    python tools/run_coarse_ls_baseline.py quad
    python tools/run_coarse_ls_baseline.py lasso
    python tools/run_coarse_ls_baseline.py logreg --alg fgm
    python tools/run_coarse_ls_baseline.py logreg --alg gd
    python tools/run_coarse_ls_baseline.py pdlp --workers 8 --offline

Writes, under --out-dir (default src/iclr_data_outputs/baselines/coarse_ls/):
    <exp>_<alg>_coarse_ls.npz          per split: losses (N, K_max+1), steps,
                                       accepted, fallback_would_fail, f_opt,
                                       n_extra_matvec; plus the run config
    <exp>_<alg>_coarse_ls_summary.csv  split, K, mean, q10, median, q90, accept_rate

One adaptive run of K_max steps gives every horizon K = 1..K_max, so the
summary has one row per (split, K). Figure / table integration is deliberately
not done here.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from learning.baselines import coarse_line_search as cls  # noqa: E402
from learning.baselines import data_loaders as dl  # noqa: E402

DEFAULT_ALG = {'quad': 'gd', 'lasso': 'ista', 'logreg': 'fgm', 'pdlp': 'cp'}
DEFAULT_K_MAX = {'quad': 15, 'lasso': 15, 'logreg': 15, 'pdlp': 10}
DEFAULT_OUT_DIR = dl.ICLR_DIR / 'baselines' / 'coarse_ls'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('exp', choices=sorted(DEFAULT_ALG))
    p.add_argument('--alg', default=None,
                   help='quad: gd | lasso: ista | logreg: fgm, gd | pdlp: cp')
    p.add_argument('--growth', type=float, default=cls.GROWTH_DEFAULT,
                   help='candidate = growth * previous step')
    p.add_argument('--armijo-c', type=float, default=cls.ARMIJO_C_DEFAULT)
    p.add_argument('--no-compound', action='store_true',
                   help='candidate = growth * t_default every iteration, '
                        'instead of growth * previous step')
    p.add_argument('--t-default', type=float, default=None,
                   help='fallback step; default = the learned sequences\' init')
    p.add_argument('--K-max', type=int, default=None)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    p.add_argument('--splits', nargs='+', default=['test', 'ood'])
    p.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)),
                   help='pdlp only: processes for the LP solves')
    p.add_argument('--recompute-lp', action='store_true', help='pdlp only: ignore the LP cache')
    p.add_argument('--offline', action='store_true',
                   help='pdlp only: HF datasets offline mode (tiny-imagenet from cache)')
    args = p.parse_args(argv)
    args.alg = args.alg or DEFAULT_ALG[args.exp]
    args.K_max = args.K_max or DEFAULT_K_MAX[args.exp]
    args.compound = not args.no_compound
    return args


# ---------------------------------------------------------------------------

def run_quad(args):
    data = dl.load_quad(args.data_dir)
    t_def = args.t_default or data['t_default']
    out = {}
    for split in args.splits:
        s = data['splits'][split]
        out[split] = cls.gd_quad(s['Q'], s['z0'], t_default=t_def, K_max=args.K_max,
                                 growth=args.growth, c=args.armijo_c,
                                 compound=args.compound), s['f_opt']
    return out, dict(t_default=t_def, L=data['L'], mu=data['mu'], data_dir=data['data_dir'])


def run_lasso(args):
    if args.alg != 'ista':
        raise SystemExit('lasso: only --alg ista is implemented')
    data = dl.load_lasso(args.data_dir)
    t_def = args.t_default or data['t_default']
    out = {}
    for split in args.splits:
        s = data['splits'][split]
        out[split] = cls.ista_lasso(s['A'], s['b'], s['f_opt'], data['lambd'],
                                    t_default=t_def, K_max=args.K_max,
                                    growth=args.growth, c=args.armijo_c,
                                    compound=args.compound), s['f_opt']
    return out, dict(t_default=t_def, L=data['L'], lambd=data['lambd'],
                     data_dir=data['data_dir'])


def run_logreg(args):
    from learning.acceleration_stepsizes import get_nesterov_fgm_beta_sequence
    data = dl.load_logreg(args.data_dir)
    t_def = args.t_default or data['t_default']
    beta = get_nesterov_fgm_beta_sequence(0.0, data['L'], args.K_max)
    out = {}
    for split in args.splits:
        s = data['splits'][split]
        if args.alg == 'fgm':
            res = cls.fgm_logreg(s['A'], s['b'], s['f_opt'], beta, t_default=t_def,
                                 K_max=args.K_max, growth=args.growth, c=args.armijo_c,
                                 compound=args.compound)
        elif args.alg == 'gd':
            res = cls.gd_logreg(s['A'], s['b'], s['f_opt'], t_default=t_def,
                                K_max=args.K_max, growth=args.growth, c=args.armijo_c,
                                compound=args.compound)
        else:
            raise SystemExit('logreg: --alg must be fgm or gd')
        out[split] = res, s['f_opt']
    return out, dict(t_default=t_def, L=data['L'], data_dir=data['data_dir'])


def run_pdlp(args):
    if args.offline:
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from learning.baselines import pdlp_lp_cache as lpc
    from learning.tv_inpainting_test import MISSING_FRACTION

    eta_def = args.t_default or dl.PDLP_TAU0_ARCHIVED
    cache_dir = Path(args.out_dir) / 'pdlp_lp_cache'
    out, info = {}, dict(t_default=eta_def, M_archived=dl.PDLP_M_ARCHIVED,
                         missing_fraction=MISSING_FRACTION, mask_seed=lpc.MASK_SEED)
    for split in args.splits:
        if split == 'test':
            images, kind = lpc.load_olivetti_images(), 'gray'
        elif split == 'ood':
            images, kind = lpc.load_tiny_imagenet_images(), 'color'
        else:
            raise SystemExit(f'pdlp: unknown split {split}')
        t0 = time.time()
        lpc.ensure_lp_cache(split, images, kind, cache_dir, workers=args.workers,
                            recompute=args.recompute_lp)
        print(f'  [{split}] LP solutions ready ({len(images)} instances, '
              f'{time.time() - t0:.1f}s incl. cache hits)')
        batch = lpc.assemble_split(split, kind, len(images), cache_dir)
        info[f'normK_{split}'] = batch['normK']
        print(f'  [{split}] ||K||_2 = {batch["normK"]:.5f}; 1.2*||K|| = {1.2 * batch["normK"]:.5f}'
              f' vs archived M = {dl.PDLP_M_ARCHIVED:.5f}; eta_default = {eta_def:.6f}')
        _crosscheck_f_opt(split, batch['f_opt'])
        t0 = time.time()
        res = cls.pdhg_lp(batch['K'], batch['c'], batch['Qmat'], batch['l'], batch['u'],
                          batch['m1'], batch['X0'], batch['Y0'], batch['Xstar'],
                          batch['Ystar'], eta_default=eta_def, K_max=args.K_max,
                          growth=args.growth, compound=args.compound)
        print(f'  [{split}] batched PDHG done in {time.time() - t0:.1f}s')
        out[split] = res, batch['f_opt']
    return out, info


def _crosscheck_f_opt(split, f_opt):
    """Compare cached optima with the archived paper gap cache, if present."""
    d = dl.ARCHIVE_DIR / 'pdlp' / 'paper_plots'
    for p in sorted(d.glob(f'pdlp_{split}_gaps_K*_reps*.npz')) + \
             sorted(d.glob(f'pdlp_{"in" if split == "test" else split}_gaps_K*_reps*.npz')):
        ref = np.load(p)['f_opts']
        if ref.shape == f_opt.shape:
            print(f'  [{split}] f_opt vs archived {p.name}: max |diff| = '
                  f'{np.max(np.abs(ref - f_opt)):.3e}')
        else:
            print(f'  [{split}] archived {p.name} has {ref.shape[0]} instances, '
                  f'we have {f_opt.shape[0]}; no cross-check')
        return
    print(f'  [{split}] no archived gap cache found for cross-check')


# ---------------------------------------------------------------------------

def summarize(out, K_max):
    rows = []
    for split, (res, _f_opt) in out.items():
        for K in range(1, K_max + 1):
            lo = res.losses[:, K]
            rows.append(dict(split=split, K=K, mean=lo.mean(),
                             q10=np.quantile(lo, 0.1), median=np.median(lo),
                             q90=np.quantile(lo, 0.9),
                             accept_rate=res.accepted[:, K - 1].mean()))
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'coarse line search: {args.exp} / {args.alg}  growth={args.growth} '
          f'c={args.armijo_c} K_max={args.K_max} compound={args.compound}')

    runner = {'quad': run_quad, 'lasso': run_lasso, 'logreg': run_logreg, 'pdlp': run_pdlp}[args.exp]
    t0 = time.time()
    out, info = runner(args)
    print(f'  done in {time.time() - t0:.2f}s; t_default = {info["t_default"]:.6g}')

    df = summarize(out, args.K_max)
    stem = f'{args.exp}_{args.alg}_coarse_ls' + ('' if args.compound else '_nocompound')
    csv_path = out_dir / f'{stem}_summary.csv'
    df.to_csv(csv_path, index=False)

    payload = dict(exp=args.exp, alg=args.alg, growth=args.growth, armijo_c=args.armijo_c,
                   compound=args.compound,
                   K_max=args.K_max, **{k: v for k, v in info.items() if not isinstance(v, str)},
                   data_dir=str(info.get('data_dir', '')))
    for split, (res, f_opt) in out.items():
        payload.update({f'losses_{split}': res.losses, f'steps_{split}': res.steps,
                        f'accepted_{split}': res.accepted,
                        f'fallback_would_fail_{split}': res.fallback_would_fail,
                        f'f_opt_{split}': np.asarray(f_opt),
                        f'n_extra_matvec_{split}': res.n_extra_matvec})
    npz_path = out_dir / f'{stem}.npz'
    np.savez_compressed(npz_path, **payload)

    with pd.option_context('display.width', 160, 'display.float_format', '{:.4g}'.format):
        print(df.to_string(index=False))
    for split, (res, _) in out.items():
        print(f'  [{split}] N={res.losses.shape[0]}  overall accept rate '
              f'{res.accepted.mean():.3f}  fallback_would_fail total '
              f'{int(res.fallback_would_fail.sum())}  extra matvecs {res.n_extra_matvec}')
    print(f'  wrote {csv_path}\n  wrote {npz_path}')


if __name__ == '__main__':
    main()
