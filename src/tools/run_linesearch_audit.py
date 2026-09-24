#!/usr/bin/env python3
"""Replay current ICLR schedules and compare a declared grid of line searches.

Run ONLY as a Slurm job. No training, test-based tuning, or new instance sampling.
Outputs checkpoint per method/split, with CSV summaries and provenance checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'logreg_rebuttal')]
from learning.baselines import coarse_line_search as old
from learning.baselines import data_loaders as dl
from learning.acceleration_stepsizes import get_nesterov_fgm_beta_sequence
import build_logreg_table as blt


@dataclass(frozen=True)
class Setting:
    name: str
    rule: str = 'backtrack'
    c: float = 0.5
    shrink: float = 0.5
    growth: float = 1.0
    initial: str = 'training'
    compound: bool = True


def settings(problem):
    # Declared before looking at any new result. All configurations are exported.
    out = [Setting('fixed_1L', rule='fixed'),
           Setting('vinit_coarse', rule='coarse', growth=2, c=1e-4),
           Setting('vinit_coarse_reset', rule='coarse', growth=2, c=1e-4, compound=False)]
    rules = [('majorization', 0.5)]
    if problem.startswith('logreg'):
        rules = [('armijo', 1e-4), ('majorization', 0.5)]
    for label, c in rules:
        for shrink in [0.5, 0.8]:
            for initial in ['training', 'unit']:
                for growth in [1.0, 2.0]:
                    name = f'{label}_s{shrink:g}_{initial}_g{growth:g}'
                    out.append(Setting(name, c=c, shrink=shrink, initial=initial, growth=growth))
    return out


def simulate(problem, s, lambd, t0, K, cfg, schedule=None, beta=None, monitor=True):
    """Float64 cached operator implementation. Counts are PER INSTANCE.

    Matrix counts include the final objective's forward product for every method.
    Function counts include only values used by line search. With monitor=False,
    fixed schedules evaluate the objective only at the final point (timing mode).
    """
    t0 = float(t0)
    A, B, fopt = s['A'], s['b'], s['f_opt']
    lasso = problem == 'lasso'
    N, n = B.shape[0], A.shape[-1]
    X = np.zeros((N, n))
    Y = X.copy()
    AX = np.zeros_like(B)
    AY = AX.copy()
    losses = np.empty((N, K + 1))
    counts = {key: np.zeros((N, K + 1), dtype=np.int64)
              for key in ['matvec', 'gradient', 'function', 'prox', 'trials']}
    steps = np.empty((N, K))
    accepted = np.zeros((N, K), dtype=bool)
    fallback_fail = np.zeros((N, K), dtype=bool)
    violations = np.zeros((N, K))

    def value(Z, ids=None):
        b = B if ids is None else B[ids]
        if lasso:
            return 0.5 * np.sum((Z - b) ** 2, axis=1)
        return np.mean(np.logaddexp(0.0, Z) - b * Z, axis=1)

    fy = value(AY)
    losses[:, 0] = fy - fopt
    previous = np.full(N, t0)
    for k in range(K):
        for v in counts.values():
            v[:, k + 1] = v[:, k]
        if lasso:
            G = (AY - B) @ A
        else:
            G = np.einsum('nmi,nm->ni', A, expit(AY) - B) / A.shape[1]
        counts['gradient'][:, k + 1] += 1
        counts['matvec'][:, k + 1] += 1
        if not lasso:
            AG = np.einsum('nmi,ni->nm', A, G)
            counts['matvec'][:, k + 1] += 1
        adaptive = cfg.rule != 'fixed'
        if adaptive:
            # GD/ISTA reuses the accepted point's value. FGM needs f(y) when
            # momentum moves the gradient point away from the accepted point.
            if k == 0 or (beta is not None and beta[k - 1] != 0):
                fy = value(AY)
                counts['function'][:, k + 1] += 1
        if cfg.rule == 'fixed':
            t = np.full(N, schedule[k] if schedule is not None else t0)
        elif cfg.rule == 'coarse':
            t = cfg.growth * (previous if cfg.compound else np.full(N, t0))
        else:
            t = np.full(N, t0) if k == 0 else cfg.growth * previous
        Xnew = np.empty_like(X)
        AXnew = np.empty_like(AX)
        fnew = np.empty(N)
        active = np.arange(N)
        trial_number = 0
        while active.size:
            trial_number += 1
            tt = t[active]
            XX = Y[active] - tt[:, None] * G[active]
            if lasso:
                XX = old._soft(XX, tt[:, None] * lambd)
                ZZ = XX @ A.T
                counts['matvec'][active, k + 1] += 1
                counts['prox'][active, k + 1] += 1
            else:
                ZZ = AY[active] - tt[:, None] * AG[active]
            ff = value(ZZ, active) if adaptive or monitor or k == K - 1 else np.zeros(active.size)
            counts['trials'][active, k + 1] += 1
            if adaptive:
                counts['function'][active, k + 1] += 1
                if lasso:
                    D = XX - Y[active]
                    rhs = fy[active] + np.sum(G[active] * D, axis=1) + np.sum(D * D, axis=1) / (2 * tt)
                else:
                    rhs = fy[active] - cfg.c * tt * np.sum(G[active] ** 2, axis=1)
                violation = ff - rhs
                # Match Vinit's exact comparison for reproduction; permit only
                # roundoff-level slack in standard backtracking.
                slack = 0 if cfg.rule == 'coarse' else 1e-13 * (1 + np.abs(fy[active]))
                ok = violation <= slack
            else:
                violation = np.zeros(active.size)
                ok = np.ones(active.size, dtype=bool)
            if cfg.rule == 'coarse' and trial_number == 2:
                fallback_fail[active, k] = ~ok
                take = np.ones(active.size, dtype=bool)
            else:
                take = ok
            ids = active[take]
            Xnew[ids], AXnew[ids], fnew[ids] = XX[take], ZZ[take], ff[take]
            violations[ids, k] = violation[take]
            if trial_number == 1:
                accepted[ids, k] = True
            active = active[~take]
            if active.size:
                if cfg.rule == 'coarse':
                    t[active] = t0
                else:
                    t[active] *= cfg.shrink
                if trial_number >= 100:
                    raise RuntimeError(f'Line search failed at k={k}, {cfg.name}')
        losses[:, k + 1] = fnew - fopt
        if lasso:
            losses[:, k + 1] += lambd * np.sum(np.abs(Xnew), axis=1)
        if beta is None:
            Y, AY = Xnew, AXnew
        else:
            Y = Xnew + beta[k] * (Xnew - X)
            AY = AXnew + beta[k] * (AXnew - AX)
        X, AX, fy = Xnew, AXnew, fnew
        previous = t.copy()
        steps[:, k] = t
    if not np.all(np.isfinite(losses)):
        raise RuntimeError(f'Nonfinite losses: {cfg.name}')
    return dict(losses=losses, steps=steps, accepted=accepted,
                fallback_fail=fallback_fail, violations=violations, **counts)


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_data(problem):
    data = dl.load_lasso() if problem == 'lasso' else dl.load_logreg()
    data['lambd'] = data.get('lambd', 0.0)
    return data


def schedules(problem):
    for K in range(1, 16):
        for label, tag in [('L2O', 'l2o'), ('DR-L2O', 'ldro_pep'), ('OPT-PEP', 'lpep')]:
            if problem == 'lasso':
                p = dl.ARCHIVE_DIR / 'lasso' / 'plots' / f'K_{K}' / f'{tag}_best_stepsize_schedule.csv'
                row = pd.read_csv(p).iloc[0]
                t = row[[f'gamma_{k}' for k in range(K)]].to_numpy(float)
                beta = None
                meta = {'csv': str(p.relative_to(ROOT)), 'source': row['source_dir'], 'sha256': file_hash(p)}
            else:
                alg = 'vanilla_gd' if problem.endswith('_gd') else 'nesterov_fgm'
                selected = blt.select_schedule(dl.ICLR_DIR, label, alg, K)
                if selected is None:
                    continue
                _, t, beta, meta = selected
                if problem.endswith('_gd'):
                    beta = None
            yield label, tag, K, t, beta, meta


def independent_fixed(problem, s, lambd, t, beta):
    if problem != 'lasso':
        d = dict(A_batch=s['A'], b_batch=s['b'], f_opt_batch=s['f_opt'])
        return blt.gd_losses(t, d) if beta is None else blt.fgm_losses(t, beta, d)
    X = np.zeros((len(s['b']), s['A'].shape[1]))
    for step in t:
        G = (X @ s['A'].T - s['b']) @ s['A']
        X = old._soft(X - step * G, step * lambd)
    return 0.5 * np.sum((X @ s['A'].T - s['b']) ** 2, axis=1) + lambd * np.sum(np.abs(X), axis=1) - s['f_opt']


def verify(problem, data):
    checks = []
    K = 15
    beta = get_nesterov_fgm_beta_sequence(0, data['L'], K) if problem.endswith('_fgm') else None
    for cfg in settings(problem)[1:3]:
        suffix = '_nocompound' if not cfg.compound else ''
        exp_alg = 'lasso_ista' if problem == 'lasso' else problem
        reference_path = dl.ICLR_DIR / 'baselines' / 'coarse_ls' / f'{exp_alg}_coarse_ls{suffix}.npz'
        if not reference_path.exists():
            # GD reset was not included in Vinit's minimal archive. The other
            # five archived configurations are checked without substitutions.
            continue
        ref = np.load(reference_path)
        for split, s in data['splits'].items():
            got = simulate(problem, s, data['lambd'], float(ref['t_default']), K, cfg, beta=beta)
            mask = np.ones(len(s['b']), dtype=bool)
            if problem == 'lasso' and split == 'test' and len(ref['f_opt_test']) == 248:
                mask[[111, 189]] = False
            delta = np.max(np.abs(got['losses'][mask] - ref[f'losses_{split}']))
            np.testing.assert_allclose(s['f_opt'][mask], ref[f'f_opt_{split}'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(got['losses'][mask], ref[f'losses_{split}'], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(got['steps'][mask], ref[f'steps_{split}'], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(got['accepted'][mask], ref[f'accepted_{split}'])
            checks.append(dict(check='current_committed_coarse_cache', setting=cfg.name, split=split, max_abs_error=float(delta), n=int(mask.sum()), status='PASS'))
    # A scalar lasso has a closed-form optimum; a huge initial step must
    # backtrack to a passing point. This also checks exact operator accounting.
    tiny = dict(A=np.eye(2), b=np.array([[3., -2.], [1., 0.2]]), f_opt=np.zeros(2))
    cfg = Setting('scalar_check', initial='unit')
    r = simulate('lasso', tiny, 0.4, 16, 1, cfg)
    expected = old._soft(tiny['b'], 0.4)
    expected_f = 0.5 * np.sum((expected - tiny['b']) ** 2, axis=1) + 0.4 * np.abs(expected).sum(axis=1)
    np.testing.assert_allclose(r['losses'][:, 1], expected_f)
    np.testing.assert_array_equal(r['trials'][:, 1], 5)
    np.testing.assert_array_equal(r['matvec'][:, 1], 6)
    assert np.max(r['violations']) < 1e-10
    checks.append(dict(check='analytic_lasso_backtracking_and_counts', status='PASS'))
    return checks


def summary_rows(problem, method, label, split, r, fopt, only_K=None, timing=np.nan):
    masks = [('all', np.ones(len(fopt), dtype=bool))]
    if problem == 'lasso' and split == 'test':
        mask = np.ones(len(fopt), dtype=bool)
        mask[[111, 189]] = False
        masks.append(('paper248', mask))
    Ks = [only_K] if only_K is not None else range(1, r['losses'].shape[1])
    for cohort, mask in masks:
        for K in Ks:
            losses = r['losses'][mask, K]
            d = dict(problem=problem, method=method, label=label, split=split,
                     cohort=cohort, K=K, n=int(mask.sum()), mean=float(losses.mean()),
                     median=float(np.median(losses)), q10=float(np.quantile(losses, .1)),
                     q90=float(np.quantile(losses, .9)), maximum=float(losses.max()),
                     minimum=float(losses.min()),
                     n_not_improved=int(np.sum(losses >= r['losses'][mask, 0])),
                     batch_seconds=timing if K == 15 else np.nan,
                     fallback_fail=int(r['fallback_fail'][mask, :K].sum()))
            for key in ['matvec', 'gradient', 'function', 'prox', 'trials']:
                d[key+'_mean'] = float(r[key][mask, K].mean())
                d[key+'_q90'] = float(np.quantile(r[key][mask, K], .9))
            for tol in ([1e-3, 1e-2, 1e-1] if problem == 'lasso' else [1e-4, 1e-3, 1e-2]):
                d[f'solved_{tol:g}'] = float(np.mean(losses <= tol * (1 + np.abs(fopt[mask]))))
            # Equal matrix-product budget: last fully accepted iterate whose
            # cumulative work fits the fixed schedule's 2*K products. An
            # unsuccessful partial search does not advance the returned point.
            admissible = r['matvec'][mask] <= 2 * K
            indices = np.maximum.accumulate(np.where(admissible, np.arange(r['losses'].shape[1]), 0), axis=1)[:, -1]
            budget_loss = r['losses'][mask][np.arange(mask.sum()), indices]
            d['equal_matvec_mean'] = float(budget_loss.mean())
            d['equal_matvec_median'] = float(np.median(budget_loss))
            d['equal_matvec_q90'] = float(np.quantile(budget_loss, .9))
            d['equal_matvec_steps_mean'] = float(indices.mean())
            yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', choices=['lasso', 'logreg_gd', 'logreg_fgm'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'full'], default='full')
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Experiments are restricted to Slurm. Use slurm/job.slurm.')
    out = args.results_dir / args.problem
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)
    data = load_data(args.problem)
    checks = verify(args.problem, data)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2), flush=True)
    inputs = {str(p.relative_to(ROOT)): file_hash(p) for p in Path(data['data_dir']).glob('*.npz')}
    signature = hashlib.sha256((file_hash(__file__) + json.dumps(inputs, sort_keys=True)).encode()).hexdigest()
    manifest = dict(problem=args.problem, settings=[asdict(c) for c in settings(args.problem)],
                    baseline_sha='40398f2', script_sha256=file_hash(__file__), input_sha256=inputs,
                    job_id=os.environ['SLURM_JOB_ID'], host=platform.node(),
                    python=platform.python_version(), numpy=np.__version__,
                    L=data['L'], lambd=data['lambd'], source_dir=data['data_dir'],
                    repeats=args.repeats, timing='median of 5 CPU batch runs; counters retained, intermediate diagnostic-only objective evaluations omitted; one warmup',
                    schedule_selection='unchanged current iclr selectors; no new tuning',
                    test_exclusions='none in primary results; additionally show historical paper248 mask',
                    schedules=[])
    rows = []
    beta_std = get_nesterov_fgm_beta_sequence(0, data['L'], 15) if args.problem.endswith('_fgm') else None

    def run(method, label, cfg, K=15, schedule=None, beta=None, only_K=None):
        for split, s in data['splits'].items():
            p = raw / f'{method}_{split}.npz'
            t0 = data['t_default'] if cfg.rule == 'coarse' else (1.0 if cfg.initial == 'unit' else 1/data['L'])
            if p.exists():
                stored = np.load(p)
                if str(stored['signature']) != signature:
                    raise RuntimeError(f'Stale checkpoint: choose a new EXPERIMENT directory: {p}')
                r = {k: stored[k] for k in stored.files if k not in ['signature', 'timings']}
                timings = stored['timings'].tolist()
            else:
                fn = lambda: simulate(args.problem, s, data['lambd'], t0, K, cfg, schedule, beta)
                r = fn()
                if cfg.rule == 'backtrack':
                    assert np.max(r['violations']) < 1e-8, (method, np.max(r['violations']))
                if cfg.rule == 'fixed':
                    t = schedule if schedule is not None else np.full(K, t0)
                    ref = independent_fixed(args.problem, s, data['lambd'], t, beta)
                    np.testing.assert_allclose(r['losses'][:, -1], ref, rtol=1e-9, atol=1e-9)
                timings = []
                if K == 15:
                    timed_fn = lambda: simulate(args.problem, s, data['lambd'], t0, K, cfg, schedule, beta, monitor=False)
                    timed_result = timed_fn()
                    np.testing.assert_allclose(timed_result['losses'][:, -1], r['losses'][:, -1], rtol=1e-12, atol=1e-12)
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        timed_fn()
                        timings.append(time.perf_counter() - start)
                tmp = p.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **r, signature=signature, timings=timings)
                tmp.replace(p)
            rows.extend(summary_rows(args.problem, method, label, split, r, s['f_opt'], only_K,
                                     np.median(timings) if timings else np.nan))
            pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
            print(f'{method} {split}: K={K} mean={r["losses"][:, -1].mean():.6g}', flush=True)

    for label, tag, K, t, beta, meta in schedules(args.problem):
        if args.mode == 'smoke' and K != 15:
            continue
        manifest['schedules'].append(dict(label=label, K=K, steps=t.tolist(),
                                         beta=None if beta is None else beta.tolist(), **meta))
        run(f'{tag}_K{K}', label, Setting(label, rule='fixed'), K, t, beta, only_K=K)
    for cfg in settings(args.problem):
        if args.mode == 'smoke' and cfg.name not in ['vinit_coarse', 'majorization_s0.5_training_g2']:
            continue
        run(cfg.name, cfg.name, cfg, beta=beta_std)
    if args.problem == 'lasso':
        paper = pd.read_csv(dl.ARCHIVE_DIR / 'lasso' / 'paper_plots' / 'lasso_losses.csv')
        frame = pd.DataFrame(rows)
        for label, tag in [('L2O', 'l2o'), ('DR-L2O', 'ldro_pep'), ('OPT-PEP', 'lpep')]:
            for split in ['test', 'ood']:
                cohort = 'paper248' if split == 'test' else 'all'
                sub = frame[(frame.label == label) & (frame.split == split) & (frame.cohort == cohort)]
                ref = paper[(paper.arch == tag) & (paper.row == split)]
                merged = sub.merge(ref, on='K')
                np.testing.assert_allclose(merged['mean'], merged.final_loss_mean, rtol=1e-8, atol=1e-8)
                checks.append(dict(check='current_committed_learned_figure', label=label, split=split,
                                   max_abs_error=float(np.max(np.abs(merged['mean']-merged.final_loss_mean))),
                                   horizons=len(merged), status='PASS'))
    (out/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Completed {args.problem}; {len(rows)} summary rows -> {out}', flush=True)


if __name__ == '__main__':
    main()
