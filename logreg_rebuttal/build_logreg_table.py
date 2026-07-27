"""Build the OpenReview rebuttal table for the LogReg GD-vs-FGM experiment.

Mirrors the selection conventions of lasso_intro_repro/reconstruct_lasso_intro.py:
  - per (framework, alg, K): glob every progress.csv, filter runs via the
    sibling .hydra/config.yaml, take the min-validation_loss row within each
    run, then the min across runs;
  - DR-L2O uses the fixed robust choice eps=10, eta_t=1e-3 (not CV'd);
    L2O and OPT-PEP validation-select the learning rate.

Learned schedules and handcrafted baselines (GD 1/L, Silver GD, Nesterov FGM)
are re-simulated with plain NumPy on the archived test/OOD sets, and
mean/median/q10/q90/fraction-solved statistics are written to results.csv,
table.md (OpenReview paste), and table.tex.

Usage (from repo root):
    python logreg_rebuttal/build_logreg_table.py \
        --runs-root logreg_rebuttal/runs \
        --data-dir  logreg_rebuttal/data \
        --ks 5 10 15

`runs_root` must contain learn_dro_outputs/LogReg/..., learn_l2o_outputs/...,
learn_lpep_outputs/... as rsynced from the cluster; `data_dir` must contain
training_set.npz, test_set.npz, ood_set.npz from the sample-creation run.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from learning.acceleration_stepsizes import get_nesterov_fgm_beta_sequence  # noqa: E402
from learning.silver_stepsizes import get_nonstrongly_convex_silver_stepsizes  # noqa: E402


# ---------------------------------------------------------------------------
# NumPy simulators (original coordinates, x0 = 0)
# ---------------------------------------------------------------------------

def logreg_f(X, A_batch, b_batch):
    """f(x_i) for every instance i; X has shape (S, n)."""
    AX = np.einsum('smn,sn->sm', A_batch, X)
    # log(1 + exp(AX)) computed stably
    softplus = np.logaddexp(0.0, AX)
    return -np.mean(b_batch * AX - softplus, axis=1)


def logreg_grad(X, A_batch, b_batch):
    m = A_batch.shape[1]
    AX = np.einsum('smn,sn->sm', A_batch, X)
    sig = 1.0 / (1.0 + np.exp(-AX))
    return np.einsum('smn,sm->sn', A_batch, sig - b_batch) / m


def gd_losses(t_vec, data):
    """f(x_K) - f_opt for GD with per-iteration step sizes t_vec."""
    A, b, f_opt = data['A_batch'], data['b_batch'], data['f_opt_batch']
    S, n = A.shape[0], A.shape[2]
    X = np.zeros((S, n))
    for t in t_vec:
        X = X - t * logreg_grad(X, A, b)
    return logreg_f(X, A, b) - f_opt


def fgm_losses(t_vec, beta_vec, data):
    """f(x_K) - f_opt for FGM: x+ = y - t g(y); y+ = x+ + beta (x+ - x).

    Matches logreg_fgm_trajectories: beta_vec[K-1] is unused (y_K is never
    formed), and the reported loss is at x_K.
    """
    A, b, f_opt = data['A_batch'], data['b_batch'], data['f_opt_batch']
    S, n = A.shape[0], A.shape[2]
    Y = np.zeros((S, n))
    X_curr = np.zeros((S, n))
    for k in range(len(t_vec)):
        X_new = Y - t_vec[k] * logreg_grad(Y, A, b)
        Y = X_new + beta_vec[k] * (X_new - X_curr)
        X_curr = X_new
    return logreg_f(X_curr, A, b) - f_opt


# ---------------------------------------------------------------------------
# Schedule selection from progress.csv sweeps
# ---------------------------------------------------------------------------

FRAMEWORK_DIRS = {
    'DR-L2O': 'learn_dro_outputs',
    'L2O': 'learn_l2o_outputs',
    'OPT-PEP': 'learn_lpep_outputs',
}
FRAMEWORK_CFG = {
    'DR-L2O': 'ldro-pep',
    'L2O': 'l2o',
    'OPT-PEP': 'lpep',
}
# DR-L2O hyperparameter selection. The paper's methodology cross-validates
# the Wasserstein radius, so by default eps is validation-selected across the
# sweep (like the learning rate for L2O/OPT-PEP). Set to a float (e.g. 10.0)
# to reproduce the intro-figure fixed-eps convention instead.
DRL2O_EPS = None
DRL2O_ETA = 1e-3


def load_candidates(runs_root, series, alg, K):
    """All (val_loss, t, beta, meta) candidates for one (series, alg, K)."""
    pattern = os.path.join(
        runs_root, FRAMEWORK_DIRS[series], 'LogReg', '*', '*',
        'learn_dro_outputs', f'K_{K}', 'progress.csv',
    )
    out = []
    for csv_path in sorted(glob.glob(pattern)):
        # csv_path = <run_dir>/learn_dro_outputs/K_<K>/progress.csv; the outer
        # series dir may ALSO be named learn_dro_outputs, so use dirname
        # arithmetic rather than a string split.
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(csv_path)))
        cfg_path = os.path.join(run_dir, '.hydra', 'config.yaml')
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get('learning_framework') != FRAMEWORK_CFG[series]:
            continue
        if cfg.get('alg') != alg:
            continue
        if series == 'DR-L2O':
            if DRL2O_EPS is not None and float(cfg.get('eps', -1)) != DRL2O_EPS:
                continue
            if float(cfg.get('eta_t', -1)) != DRL2O_ETA:
                continue
        df = pd.read_csv(csv_path)
        df = df[np.isfinite(df['validation_loss'])]
        if len(df) == 0:
            continue
        i = int(df['validation_loss'].idxmin())
        t = df.loc[i, [f't{k}' for k in range(K)]].to_numpy(float)
        beta = None
        if f'beta{K - 1}' in df.columns:
            beta = df.loc[i, [f'beta{k}' for k in range(K)]].to_numpy(float)
        out.append((
            float(df.loc[i, 'validation_loss']), t, beta,
            {'csv': os.path.relpath(csv_path, runs_root), 'row': i,
             'eps': cfg.get('eps'), 'eta_t': cfg.get('eta_t')},
        ))
    return out


def select_schedule(runs_root, series, alg, K):
    cands = load_candidates(runs_root, series, alg, K)
    if not cands:
        return None
    return min(cands, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(losses, f_opt, initial_losses, tols=(1e-2, 1e-3)):
    """Summary stats; 'solved' uses the paper's relative criterion
    loss <= eta * (1 + |f_opt|). Divergence = loss >= initial loss."""
    s = {
        'mean': float(np.mean(losses)),
        'median': float(np.median(losses)),
        'q10': float(np.quantile(losses, 0.1)),
        'q90': float(np.quantile(losses, 0.9)),
        'max': float(np.max(losses)),
        'n_diverged': int(np.sum(losses >= initial_losses)),
    }
    for tol in tols:
        s[f'solved_{tol:g}'] = float(np.mean(losses <= tol * (1 + np.abs(f_opt))))
    return s


def fmt_sci(x):
    if not np.isfinite(x):
        return 'inf'
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return f'{mant:.2f}e{exp:+03d}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 15])
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--solved-tol', type=float, default=1e-2,
                    help='tolerance shown in the markdown/latex tables')
    args = ap.parse_args()

    train = dict(np.load(os.path.join(args.data_dir, 'training_set.npz')))
    datasets = {
        'test': dict(np.load(os.path.join(args.data_dir, 'test_set.npz'))),
        'ood': dict(np.load(os.path.join(args.data_dir, 'ood_set.npz'))),
    }

    # L exactly as used in training: max per-instance smoothness on the
    # training set (delta = 0).
    m = train['A_batch'].shape[1]
    L_vals = [np.linalg.eigvalsh(A.T @ A).max() / (4 * m) for A in train['A_batch']]
    L = float(np.max(L_vals))
    print(f'L (training-set max) = {L:.6f}')

    init_losses = {
        name: logreg_f(np.zeros((d['A_batch'].shape[0], d['A_batch'].shape[2])),
                       d['A_batch'], d['b_batch']) - d['f_opt_batch']
        for name, d in datasets.items()
    }

    rows = []
    manifest = []

    def add_method(method, K, t_vec, beta_vec, meta=None):
        for ds_name, ds in datasets.items():
            if beta_vec is None:
                losses = gd_losses(t_vec, ds)
            else:
                losses = fgm_losses(t_vec, beta_vec, ds)
            s = stats(losses, ds['f_opt_batch'], init_losses[ds_name])
            rows.append({'method': method, 'K': K, 'dataset': ds_name, **s})
        manifest.append({
            'method': method, 'K': K,
            't': list(np.round(t_vec, 6)),
            'beta': list(np.round(beta_vec, 6)) if beta_vec is not None else None,
            'min_beta': float(np.min(beta_vec[:-1])) if beta_vec is not None and K > 1 else None,
            **(meta or {}),
        })

    missing = []
    for K in args.ks:
        # Handcrafted baselines
        add_method('GD (1/L)', K, np.full(K, 1.0 / L), None)
        add_method('Silver GD', K,
                   np.array(get_nonstrongly_convex_silver_stepsizes(K, L=L)), None)
        beta_std = get_nesterov_fgm_beta_sequence(0.0, L, K)
        add_method('Nesterov FGM', K, np.full(K, 1.0 / L), np.asarray(beta_std))

        # Learned schedules
        for series in ['L2O', 'OPT-PEP', 'DR-L2O']:
            for alg, alg_label in [('vanilla_gd', 'GD'), ('nesterov_fgm', 'FGM')]:
                sel = select_schedule(args.runs_root, series, alg, K)
                label = f'{series} {alg_label}'
                if sel is None:
                    missing.append((label, K))
                    continue
                val_loss, t_vec, beta_vec, meta = sel
                add_method(label, K, t_vec,
                           beta_vec if alg == 'nesterov_fgm' else None,
                           {'val_loss': val_loss, **meta})

    if missing:
        print('WARNING: no runs found for:', missing)

    os.makedirs(args.out_dir, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)
    pd.DataFrame(manifest).to_csv(
        os.path.join(args.out_dir, 'selected_schedules.csv'), index=False)

    # ------------------------------------------------------------------
    # Markdown table (one block per dataset)
    # ------------------------------------------------------------------
    tol = args.solved_tol
    method_order = [m for m in [
        'GD (1/L)', 'Silver GD', 'Nesterov FGM',
        'L2O GD', 'L2O FGM', 'OPT-PEP GD', 'OPT-PEP FGM',
        'DR-L2O GD', 'DR-L2O FGM',
    ] if m in set(results['method'])]

    md = []
    tex = []
    for ds_name, ds_title in [('test', 'In-distribution test set'),
                              ('ood', 'Out-of-distribution set')]:
        md.append(f'**{ds_title}** (loss $f(x^K)-f^\\star$; '
                  f'solved = fraction with loss $\\le {tol:g}(1+|f^\\star|)$)\n')
        header = '| Method |' + ''.join(
            f' K={K} mean [q10, q90] | K={K} solved |' for K in args.ks)
        sep = '|---|' + '---|' * (2 * len(args.ks))
        md += [header, sep]
        for method in method_order:
            cells = [f'| {method} |']
            for K in args.ks:
                r = results[(results['method'] == method) & (results['K'] == K)
                            & (results['dataset'] == ds_name)]
                if len(r) == 0:
                    cells.append(' -- | -- |')
                    continue
                r = r.iloc[0]
                cells.append(
                    f" {fmt_sci(r['mean'])} [{fmt_sci(r['q10'])}, {fmt_sci(r['q90'])}] "
                    f"| {r[f'solved_{tol:g}'] * 100:.0f}% |")
            md.append(''.join(cells))
        md.append('')

        # LaTeX block
        tex.append(f'% {ds_title}')
        tex.append('\\begin{tabular}{l' + 'cc' * len(args.ks) + '}')
        tex.append('\\toprule')
        tex.append('Method & ' + ' & '.join(
            f'$K={K}$ mean & solved' for K in args.ks) + ' \\\\')
        tex.append('\\midrule')
        for method in method_order:
            cells = [method]
            for K in args.ks:
                r = results[(results['method'] == method) & (results['K'] == K)
                            & (results['dataset'] == ds_name)]
                if len(r) == 0:
                    cells += ['--', '--']
                    continue
                r = r.iloc[0]
                cells += [f"{fmt_sci(r['mean'])}",
                          f"{r[f'solved_{tol:g}'] * 100:.0f}\\%"]
            tex.append(' & '.join(cells) + ' \\\\')
        tex.append('\\bottomrule')
        tex.append('\\end{tabular}')
        tex.append('')

    with open(os.path.join(args.out_dir, 'table.md'), 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out_dir, 'table.tex'), 'w') as f:
        f.write('\n'.join(tex))

    print('\n'.join(md))
    neg = [m for m in manifest if m.get('min_beta') is not None and m['min_beta'] < 0]
    if neg:
        print('Learned schedules with NEGATIVE momentum coefficients '
              '(signed span evidence):')
        for m_ in neg:
            print(f"  {m_['method']} K={m_['K']}: min beta = {m_['min_beta']:.4f}")
    print(f"Wrote results.csv, selected_schedules.csv, table.md, table.tex "
          f"to {args.out_dir}")


if __name__ == '__main__':
    main()
