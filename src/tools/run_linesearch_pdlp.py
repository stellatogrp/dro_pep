#!/usr/bin/env python3
"""Standard (unit-reset) backtracking PDHG on the paper's TV-inpainting LPs.

PDLP counterpart of ``run_linesearch_audit.py --mode backtracking``. No
training, no new instances: the in-distribution split is the 400 Olivetti
faces and the OOD split the 40 tiny-imagenet images, read from the committed
LP solution cache (``baselines/coarse_ls/pdlp_lp_cache``). The learned
schedules are compared through the archived per-instance Lagrangian gaps of
the paper figure (``archive/pdlp/paper_plots/pdlp_*_gaps_K10_reps1.npz``).

Iteration and metric follow ``learning.tv_averages.run_pdhg_capture_gaps``:

    x+ = clip(x - tau (c - K^T y), l, u)
    y+ = partial_relu(y + sigma (q - K (x+ + theta (x+ - x))))
    gap_k = L(x_k, y*) - L(x*, y_k),  L(x, y) = c^T x - y^T K x + q^T y

Backtracking: tau = sigma = eta (primal weight 1), theta = 1. Each iteration
restarts at eta = 1 and halves eta until the PDLP adaptive-step condition
(Applegate et al. 2021) holds:
``eta <= (||dx||^2 + ||dy||^2) / (2 |(K dx)^T dy|)``.
This is a labeled adaptation of Boyd and Vandenberghe, Algorithm 9.2.
Cost per iteration: one K^T y plus one K x per trial. A fixed schedule costs
two products per iteration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from learning.baselines import data_loaders as dl
from learning.baselines.pdlp_lp_cache import assemble_split

CACHE_DIR = dl.ICLR_DIR / 'baselines' / 'coarse_ls' / 'pdlp_lp_cache'
PDLP_DIR = dl.ARCHIVE_DIR / 'pdlp'
PAPER_DIR = PDLP_DIR / 'paper_plots'
SPLITS = {'test': ('gray', 400, 'in'), 'ood': ('color', 40, 'ood')}
K_MAX = 10
LEARNED = [('L2O', 'l2o'), ('DR-L2O', 'ldro_pep'), ('OPT-PEP', 'lpep')]


def simulate(d, K, schedule=None, eta0=1.0, shrink=0.5, max_trials=60):
    """Batched PDHG (instances are columns). Counts are per instance, cumulative.

    ``schedule`` = (tau, sigma, theta) arrays replays a fixed schedule;
    otherwise unit-reset backtracking with theta = 1.
    """
    Kop = sp.csr_matrix(d['K'])
    KT = Kop.T.tocsr()
    c, m1, Q = d['c'], d['m1'], d['Qmat']
    lb, ub = d['l'][:, None], d['u'][:, None]
    Xs, Ys = d['Xstar'], d['Ystar']
    X, Y = d['X0'].copy(), d['Y0'].copy()
    N = X.shape[1]
    KX = Kop @ X
    KXs = Kop @ Xs
    cXs = c @ Xs
    qYs = np.sum(Q * Ys, axis=0)

    def gap(X, Y, KX):
        prim = c @ X - np.sum(Ys * KX, axis=0) + qYs
        dual = cXs - np.sum(Y * KXs, axis=0) + np.sum(Q * Y, axis=0)
        return prim - dual

    losses = np.empty((N, K + 1))
    matvec = np.zeros((N, K + 1), dtype=np.int64)
    trials = np.zeros((N, K + 1), dtype=np.int64)
    steps = np.full((N, K), np.nan)
    margin = np.full((N, K), np.nan)   # eta_hat / eta at the accepted step
    losses[:, 0] = gap(X, Y, KX)
    for k in range(K):
        matvec[:, k + 1] = matvec[:, k] + 1   # K^T y
        trials[:, k + 1] = trials[:, k]
        KTY = KT @ Y
        if schedule is not None:
            tau, sigma, theta = (float(a[k]) for a in schedule)
            eta = np.full(N, np.nan)
        else:
            tau = sigma = eta = np.full(N, float(eta0))
            theta = 1.0
        Xn, Yn, KXn = np.empty_like(X), np.empty_like(Y), np.empty_like(KX)
        active = np.arange(N)
        for trial in range(1, max_trials + 1):
            t = tau if np.isscalar(tau) else tau[active]
            s = sigma if np.isscalar(sigma) else sigma[active]
            Xp = np.clip(X[:, active] - t * (c[:, None] - KTY[:, active]), lb, ub)
            KXp = Kop @ Xp
            V = Y[:, active] + s * (Q[:, active] - (KXp + theta * (KXp - KX[:, active])))
            V[:m1] = np.maximum(V[:m1], 0.0)
            matvec[active, k + 1] += 1
            trials[active, k + 1] += 1
            if schedule is None:
                dX, dY = Xp - X[:, active], V - Y[:, active]
                num = np.sum(dX * dX, axis=0) + np.sum(dY * dY, axis=0)
                den = 2.0 * np.abs(np.sum((KXp - KX[:, active]) * dY, axis=0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    eta_hat = np.where(den > 0, num / den, np.inf)
                ok = eta[active] <= eta_hat
            else:
                ok = np.ones(active.size, dtype=bool)
            ids = active[ok]
            Xn[:, ids], Yn[:, ids], KXn[:, ids] = Xp[:, ok], V[:, ok], KXp[:, ok]
            if schedule is None:
                steps[ids, k] = eta[ids]
                margin[ids, k] = eta_hat[ok] / eta[ids]
            active = active[~ok]
            if not active.size:
                break
            eta = eta.copy()
            eta[active] *= shrink
            tau = sigma = eta
        else:
            raise RuntimeError(f'Backtracking failed at k={k} for {active.size} instances')
        X, Y, KX = Xn, Yn, KXn
        losses[:, k + 1] = gap(X, Y, KX)
    if not np.all(np.isfinite(losses)):
        raise RuntimeError('Nonfinite gaps')
    return dict(losses=losses, matvec=matvec, trials=trials, steps=steps, margin=margin)


def load_schedule(tag):
    arr = np.loadtxt(PDLP_DIR / tag.replace('_', '-') / f'learned_pdhg_stepsizes_K{K_MAX}.csv',
                     delimiter=',', skiprows=1)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def summary_rows(method, label, split, losses, matvec, per_iter_cost=None):
    """``per_iter_cost``: fixed schedules cost 2 products per iteration."""
    N = losses.shape[0]
    for K in range(1, losses.shape[1]):
        L = losses[:, K]
        row = dict(problem='pdlp', method=method, label=label, split=split, cohort='all',
                   K=K, n=N, mean=L.mean(), median=np.median(L), q10=np.quantile(L, .1),
                   q90=np.quantile(L, .9), maximum=L.max(), minimum=L.min(),
                   n_not_improved=int(np.sum(L >= losses[:, 0])))
        if matvec is None:
            row.update(matvec_mean=per_iter_cost * K, equal_matvec_mean=L.mean(),
                       equal_matvec_median=np.median(L), equal_matvec_q90=np.quantile(L, .9),
                       equal_matvec_steps_mean=float(K))
        else:
            # Last fully accepted iterate whose cumulative cost fits 2K products.
            ok = matvec <= 2 * K
            idx = np.maximum.accumulate(np.where(ok, np.arange(matvec.shape[1]), 0), axis=1)[:, -1]
            B = losses[np.arange(N), idx]
            row.update(matvec_mean=matvec[:, K].mean(), equal_matvec_mean=B.mean(),
                       equal_matvec_median=np.median(B), equal_matvec_q90=np.quantile(B, .9),
                       equal_matvec_steps_mean=idx.mean())
        yield row


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', type=Path, default=ROOT / 'results' / 'linesearch-pdlp')
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    out = args.results_dir / 'pdlp'
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    rows, checks, timings = [], [], {}
    for split, (kind, n, tag) in SPLITS.items():
        d = assemble_split(split, kind, n, CACHE_DIR)
        paper = np.load(PAPER_DIR / f'pdlp_{tag}_gaps_K{K_MAX}_reps1.npz')
        np.testing.assert_allclose(d['f_opt'], paper['f_opts'], rtol=1e-9, atol=1e-9)
        # The replayed archived schedules must reproduce the paper's per-instance gaps.
        for label, arch in LEARNED[1:]:
            r = simulate(d, K_MAX, schedule=load_schedule(arch))
            err = np.max(np.abs(r['losses'] - paper[arch]) / (1 + np.abs(paper[arch])))
            assert err < 1e-6, (split, arch, err)
            checks.append(dict(check='replay_archived_schedule_vs_paper_gaps', label=label,
                               split=split, max_rel_error=float(err), status='PASS'))
        for label, arch in LEARNED:
            rows.extend(summary_rows(arch, label, split, paper[arch], None, per_iter_cost=2))

        r = simulate(d, K_MAX)
        timed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            simulate(d, K_MAX)
            timed.append(time.perf_counter() - start)
        timings[split] = float(np.median(timed))
        np.savez_compressed(out / 'raw' / f'backtracking_{split}.npz', **r, f_opt=d['f_opt'])
        rows.extend(summary_rows('backtracking', 'backtracking', split, r['losses'], r['matvec']))
        big = r['steps'] > 1.0 / d['normK']
        checks.append(dict(check='accepted_steps_satisfy_pdlp_condition', split=split,
                           min_margin=float(np.nanmin(r['margin'])), status='PASS'))
        assert np.nanmin(r['margin']) >= 1.0
        print(f'{split}: normK={d["normK"]:.4f}, BT K={K_MAX} mean gap={r["losses"][:, -1].mean():.4g}, '
              f'DR-L2O {paper["ldro_pep"][:, -1].mean():.4g}, products={r["matvec"][:, -1].mean():.2f}, '
              f'steps>1/||K||: {big.mean():.3f}, median batch {timings[split]:.3f}s', flush=True)
        step_stats = dict(normK=d['normK'], step_mean=float(np.mean(r['steps'])),
                          step_min=float(np.min(r['steps'])), step_max=float(np.max(r['steps'])),
                          frac_steps_above_1_over_normK=float(big.mean()),
                          trials_mean_per_iter=float(r['trials'][:, -1].mean() / K_MAX))
        checks.append(dict(check='step_diagnostics', split=split, **step_stats))
    frame = pd.DataFrame(rows)
    # The learned rows must match the committed paper table exactly.
    paper = pd.read_csv(PAPER_DIR / 'pdlp_losses.csv')
    merged = frame.merge(paper, left_on=['method', 'split', 'K'], right_on=['arch', 'row', 'K'])
    assert len(merged) == 60, len(merged)
    for a, b in [('mean', 'gap_mean'), ('q10', 'gap_q10'), ('q90', 'gap_q90'), ('median', 'gap_median')]:
        np.testing.assert_allclose(merged[a], merged[b], rtol=1e-10)
    checks.append(dict(check='learned_rows_match_pdlp_losses_csv', rows=len(merged), status='PASS'))
    frame.to_csv(out / 'summary.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    manifest = dict(problem='pdlp', method='backtracking', K_max=K_MAX,
                    line_search=dict(reference='Boyd and Vandenberghe, Algorithm 9.2 (adapted)',
                                     initial_trial_each_iteration=1.0, shrink=0.5, theta=1.0,
                                     primal_weight=1.0,
                                     acceptance='PDLP adaptive-step condition (Applegate et al. 2021)'),
                    learned_source='archive/pdlp/paper_plots/pdlp_{in,ood}_gaps_K10_reps1.npz',
                    batch_seconds_median=timings, script_sha256=file_hash(__file__),
                    host=platform.node(), python=platform.python_version(), numpy=np.__version__)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Completed pdlp; {len(frame)} summary rows -> {out}')


if __name__ == '__main__':
    main()
