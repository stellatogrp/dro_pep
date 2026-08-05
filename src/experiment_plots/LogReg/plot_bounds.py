"""Paper figures for the real-data logistic regression certification experiment.

Data layout (symlink or copy run outputs here, relative to this script):
    data/samples/{GD,FGM}_1_50/{samples.csv, sample_summary_dist.csv}
    data/pep/{GD,FGM}_1_50/pep.csv
    data/dro/{GD,FGM}_{exp,cvar}_1_50/dro.csv

Outputs: logreg_obj_val.pdf (+ .csv) and logreg_eps_sweep.pdf (+ .csv).

Cross-validated eps: per K, the smallest DRO bound (over eps) that covers the
across-repeat 95th quantile of the matching empirical metric, exactly as in
Lasso/plot_bounds.py. No preconditioning anywhere.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, NullLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 17,
    "figure.figsize": (12, 5),
})

K_MAX = 30
ALPHA_VALS = [0.01, 0.05, 0.10]
DEFAULT_ALPHA = 0.01
COVERAGE_QUANTILE = 0.95
X_TICKS = [1, 3, 6, 12, 30]

WORST_COLOR = '#FFAA1C'
EXP_COLOR = '#D81B60'
CVAR_COLOR = 'tab:blue'
MARKERS = {'worst': 'o', 'cvar': 's', 'exp': '^'}
MARKEVERY = [0, 1, 2, 4, 7, 11, 17, 23, 29]   # log-spaced marker positions



# A selected radius is trustworthy when the grid resolves it from below: the
# next smaller tested radius must be within this factor, otherwise the chosen
# bound may be loose by roughly that much.
GRID_GAP_FACTOR = 3.0

def set_log_xaxis(axi):
    axi.set_xscale('log')
    axi.set_xticks(X_TICKS)
    axi.xaxis.set_major_formatter(ScalarFormatter())
    axi.xaxis.set_minor_formatter(NullFormatter())
    axi.xaxis.set_minor_locator(NullLocator())


def quantile_threshold_per_k(dist, metric_col, q=COVERAGE_QUANTILE):
    return dist.groupby('K')[metric_col].quantile(q)


def cross_val_bound(dro, dist, metric_col, K_max, alpha=None, label=''):
    """Per-K: smallest dro_feas_sol (over eps) covering the across-repeat
    95th-quantile threshold; falls back to the largest bound if none does."""
    if alpha is not None and 'alpha' in dro.columns:
        dro = dro[np.isclose(dro['alpha'], alpha)]
    thr = quantile_threshold_per_k(dist, metric_col)
    bounds, chosen_eps = [], []
    edge, coarse, uncovered = [], [], []
    for k in range(1, K_max + 1):
        rows = dro[dro['K'] == k]
        if rows.empty:   # K not yet computed (partial pull): break the line
            bounds.append(np.nan)
            chosen_eps.append(np.nan)
            continue
        feas = rows[rows['dro_feas_sol'] >= thr.loc[k]]
        if not len(feas):
            # No eps in the grid certifies the empirical threshold. The
            # fallback below plots a bound that does NOT cover it -- almost
            # always a truncated eps grid at this K, not a real result.
            uncovered.append((k, float(rows['dro_feas_sol'].max()), float(thr.loc[k])))
        pick = (rows.loc[feas['dro_feas_sol'].idxmin()] if len(feas)
                else rows.loc[rows['dro_feas_sol'].idxmax()])
        bounds.append(float(pick['dro_feas_sol']))
        chosen_eps.append(float(pick['eps']))
        # Only candidates BELOW the selection could tighten it (the bound grows
        # with eps), so the grid is adequate at this K unless the selection sits
        # at the smallest tested radius, or the next smaller one is far away.
        tested = np.sort(rows['eps'].to_numpy())
        sel = float(pick['eps'])
        below = tested[tested < sel]
        if not len(below):
            edge.append(k)
        elif sel / below[-1] > GRID_GAP_FACTOR:
            coarse.append((k, float(below[-1]), sel))
    tag = label or metric_col
    if edge:
        print(f"[cross_val WARNING] {tag}: selection sits at the smallest tested "
              f"eps at K={edge}; extend the grid downward before trusting the bound")
    if coarse:
        spans = ", ".join(f"K={k}: {lo:.3g}->{hi:.3g}" for k, lo, hi in coarse)
        print(f"[cross_val WARNING] {tag}: coarse grid below the selection "
              f"({spans}); the bound may be loose, densify those radii")
    for k, got, want in uncovered:
        print(f"[cross_val WARNING] {tag}: K={k} has no covering eps -- plotted "
              f"bound {got:.4e} < threshold {want:.4e} (fallback, not a certificate)")
    print(f"[cross_val eps] {label or metric_col}: {[round(e, 6) for e in chosen_eps]}")
    return bounds, chosen_eps


def compute_empirical_avg(samples, k):
    return samples[samples['K'] == k]['obj_val'].mean()


def compute_empirical_cvar(samples, k, alpha=DEFAULT_ALPHA):
    """Mean of the worst (largest) alpha-fraction.

    Must match experiment_classes/lasso.py:compute_empirical_cvar, which
    produces the cvar_<alpha> columns of sample_summary_dist.csv used as the
    cross-validation threshold. A quantile-based variant disagrees with it
    whenever alpha*n is not an integer (e.g. top-10 vs top-9 at alpha=0.05,
    n=200), which would make the plotted sample curve and the threshold
    driving the bound two different quantities.
    """
    vals = samples.loc[samples['K'] == k, 'obj_val'].to_numpy()
    n_tail = max(1, int(np.ceil(alpha * len(vals))))
    return float(np.mean(np.sort(vals)[-n_tail:]))


def _load(alg):
    """alg in {'GD', 'FGM'} -> dict of dataframes."""
    return {
        'samples': pd.read_csv(f'data/samples/{alg}_1_50/samples.csv'),
        'dist': pd.read_csv(f'data/samples/{alg}_1_50/sample_summary_dist.csv'),
        'pep': pd.read_csv(f'data/pep/{alg}_1_50/pep.csv'),
        'exp_dro': pd.read_csv(f'data/dro/{alg}_exp_1_50/dro.csv'),
        'cvar_dro': pd.read_csv(f'data/dro/{alg}_cvar_1_50/dro.csv'),
    }


def _plot_panel(axi, title, d, exp_bound, cvar_bound):
    Ks = range(1, K_MAX + 1)
    pep_vals = np.asarray(d['pep']['val'])[:K_MAX]  # single metric per run
    axi.set_title(title)
    axi.plot(Ks, pep_vals, label='Worst-case', color=WORST_COLOR,
             marker=MARKERS['worst'], markevery=MARKEVERY, markersize=5)
    axi.plot(Ks, cvar_bound, label='CVaR', color=CVAR_COLOR,
             marker=MARKERS['cvar'], markevery=MARKEVERY, markersize=5)
    axi.plot(Ks, exp_bound, label='Expectation', color=EXP_COLOR,
             marker=MARKERS['exp'], markevery=MARKEVERY, markersize=5)
    return pep_vals


def main_bounds_alg(alpha=DEFAULT_ALPHA, out_path='logreg_obj_val.pdf'):
    gd, fgm = _load('GD'), _load('FGM')

    gd_exp_bound, gd_exp_eps = cross_val_bound(gd['exp_dro'], gd['dist'], 'mean', K_MAX, label='GD exp')
    fgm_exp_bound, fgm_exp_eps = cross_val_bound(fgm['exp_dro'], fgm['dist'], 'mean', K_MAX, label='FGM exp')
    gd_cvar_bound, gd_cvar_eps = cross_val_bound(gd['cvar_dro'], gd['dist'], f'cvar_{alpha}', K_MAX, alpha, label=f'GD cvar a={alpha}')
    fgm_cvar_bound, fgm_cvar_eps = cross_val_bound(fgm['cvar_dro'], fgm['dist'], f'cvar_{alpha}', K_MAX, alpha, label=f'FGM cvar a={alpha}')

    gd_exp_k = [compute_empirical_avg(gd['samples'], k) for k in range(1, K_MAX + 1)]
    fgm_exp_k = [compute_empirical_avg(fgm['samples'], k) for k in range(1, K_MAX + 1)]
    gd_cvar_k = [compute_empirical_cvar(gd['samples'], k, alpha) for k in range(1, K_MAX + 1)]
    fgm_cvar_k = [compute_empirical_cvar(fgm['samples'], k, alpha) for k in range(1, K_MAX + 1)]

    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylabel(r'$\|\nabla f(x^K)\|^2$')
    for axi in ax:
        axi.set_xlabel(r'$K$')
        axi.set_yscale('log')
        axi.grid(color='lightgray', alpha=0.3)
        set_log_xaxis(axi)
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])

    gd_pep_vals = _plot_panel(ax[0], 'Gradient Descent (GD)', gd, gd_exp_bound, gd_cvar_bound)
    fgm_pep_vals = _plot_panel(ax[1], 'Fast Gradient Method (FGM)', fgm, fgm_exp_bound, fgm_cvar_bound)

    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.16, box.width, box.height - 0.2])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3)
    plt.suptitle('Logistic Regression (german.numer)')

    Ks = list(range(1, K_MAX + 1))
    gd_worst = gd['samples'][['K', 'obj_val']].groupby(['K']).max()
    fgm_worst = fgm['samples'][['K', 'obj_val']].groupby(['K']).max()
    pd.DataFrame({
        'K': Ks, 'alpha': alpha,
        'GD_worst_bound': gd_pep_vals, 'GD_worst_sample': gd_worst['obj_val'].to_numpy()[:K_MAX],
        'GD_exp_bound': gd_exp_bound, 'GD_exp_eps': gd_exp_eps, 'GD_exp_sample': gd_exp_k,
        'GD_cvar_bound': gd_cvar_bound, 'GD_cvar_eps': gd_cvar_eps, 'GD_cvar_sample': gd_cvar_k,
        'FGM_worst_bound': fgm_pep_vals, 'FGM_worst_sample': fgm_worst['obj_val'].to_numpy()[:K_MAX],
        'FGM_exp_bound': fgm_exp_bound, 'FGM_exp_eps': fgm_exp_eps, 'FGM_exp_sample': fgm_exp_k,
        'FGM_cvar_bound': fgm_cvar_bound, 'FGM_cvar_eps': fgm_cvar_eps, 'FGM_cvar_sample': fgm_cvar_k,
    }).to_csv(out_path.rsplit('.', 1)[0] + '.csv', index=False)

    plt.savefig(out_path)
    plt.close(fig)


def main_eps_sweep(k_fixed=12, alpha=DEFAULT_ALPHA, out_path='logreg_eps_sweep.pdf'):
    """Certified bound vs Wasserstein radius eps at fixed K: the interpolation
    from in-sample behavior up to the worst-case PEP bound, on real data.
    Requires a wide-eps dro.csv (logspace) in data/dro/{alg}_sweep_K{k}/."""
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylabel(f'bound at $K={k_fixed}$')
    for axi, alg, title in [(ax[0], 'GD', 'Gradient Descent (GD)'),
                            (ax[1], 'FGM', 'Fast Gradient Method (FGM)')]:
        dro = pd.read_csv(f'data/dro/{alg}_sweep_K{k_fixed}/dro.csv')
        pep = pd.read_csv(f'data/pep/{alg}_1_50/pep.csv')
        pep_val = float(pep[pep['K'] == k_fixed]['val'].iloc[0])
        rows = dro[(dro['K'] == k_fixed) & np.isclose(dro.get('alpha', alpha), alpha)] \
            if 'alpha' in dro.columns else dro[dro['K'] == k_fixed]
        # In-sample mean from the sweep run's own in-sample objectives (falls
        # back gracefully when the out-of-sample dist file is not present).
        ins = pd.read_csv(f'data/dro/{alg}_sweep_K{k_fixed}/samples.csv')
        emp_mean = float(ins[ins['K'] == k_fixed]['obj_val'].mean())

        axi.plot(rows['eps'], rows['dro_feas_sol'], color=CVAR_COLOR, marker='s', markersize=5,
                 label='DRO-PEP')
        axi.axhline(pep_val, color=WORST_COLOR, linestyle='--', label='Worst-case (PEP)')
        axi.axhline(emp_mean, color=EXP_COLOR, linestyle=':', label='In-sample mean')
        axi.set_xscale('log')
        axi.set_yscale('log')
        axi.set_xlabel(r'$\varepsilon$')
        axi.set_title(title)
        axi.grid(color='lightgray', alpha=0.3)

    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.16, box.width, box.height - 0.2])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3)
    plt.suptitle(f'Interpolation between in-sample and worst-case ($K={k_fixed}$)')
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == '__main__':
    main_bounds_alg()
    try:
        main_eps_sweep()
    except FileNotFoundError:
        print('eps sweep data not present; skipping logreg_eps_sweep.pdf')
