import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, NullLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 17,
    "figure.figsize": (12, 5),
})

exp_K_max = 25   # eps-extension coverage; base grid only beyond
cvar_K_max = 25

# num_eps_vals = 7

num_per_group = 100
groups = 100

# CVaR confidence levels (must match cfg.alpha_vals used to generate dro.csv).
ALPHA_VALS = [0.01, 0.05, 0.10]
DEFAULT_ALPHA = 0.05  # matches the alpha stated in the paper for quad and Lasso

# Across-repeat coverage level used to cross-validate the eps choice. Independent of alpha
# (the within-experiment CVaR tail level) -- it is NOT 1 - alpha.
COVERAGE_QUANTILE = 0.95

# Log-scale x-axis ticks (plain integer labels) for the iteration axis.
# Ticks beyond exp_K_max are never rendered (set_xticks does not widen xlim).
X_TICKS = [1, 3, 6, 12, 24]


def set_log_xaxis(axi):
    """Log-scale the iteration (K) axis with readable integer ticks."""
    axi.set_xscale('log')
    axi.set_xticks(X_TICKS)
    axi.xaxis.set_major_formatter(ScalarFormatter())
    axi.xaxis.set_minor_formatter(NullFormatter())
    axi.xaxis.set_minor_locator(NullLocator())  # no stray log minor ticks


def quantile_threshold_per_k(dist, metric_col, q=COVERAGE_QUANTILE):
    """Per-K q-quantile of metric_col taken ACROSS repeats. Series indexed by K."""
    return dist.groupby('K')[metric_col].quantile(q)


def cross_val_bound(dro, dist, metric_col, K_max, alpha=None, label=''):
    """Per-K: smallest dro_feas_sol (over eps) that is >= the across-repeat 95th-quantile
    threshold for metric_col; fall back to the largest available bound if none qualifies.
    Prints the chosen eps per K to stdout."""
    if alpha is not None and 'alpha' in dro.columns:
        dro = dro[np.isclose(dro['alpha'], alpha)]
    thr = quantile_threshold_per_k(dist, metric_col)
    bounds = []
    chosen_eps = []
    grid_sizes, uncovered = {}, []
    for k in range(1, K_max + 1):
        rows = dro[dro['K'] == k]
        if rows.empty:
            bounds.append(np.nan)
            chosen_eps.append(np.nan)
            continue
        grid_sizes[k] = len(rows)
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
    tag = label or metric_col
    if grid_sizes:
        full = max(grid_sizes.values())
        partial = {k: n for k, n in grid_sizes.items() if n < full}
        if partial:
            print(f"[cross_val WARNING] {tag}: incomplete eps grid (full={full}) "
                  f"at K={partial}; rerun those chunks before trusting the curve")
    for k, got, want in uncovered:
        print(f"[cross_val WARNING] {tag}: K={k} has no covering eps -- plotted "
              f"bound {got:.4e} < threshold {want:.4e} (fallback, not a certificate)")
    print(f"[cross_val eps] {label or metric_col}: {[round(e, 6) for e in chosen_eps]}")
    return bounds, chosen_eps


def compute_exp_prob(samples, pep, dro, k):
    dro_bound = dro[dro['K'] == k]['dro_feas_sol'].iloc[0]
    count = 0
    for g in range(groups):
        idx_low = g * num_per_group
        idx_high = g * num_per_group + num_per_group

        samples_g = samples[(idx_low <= samples['i']) & (samples['i'] < idx_high) & (samples['K'] == k)]
        mean = samples_g['obj_val'].mean()

        if mean < dro_bound:
            count += 1

    return count / groups


def compute_empirical_avg(samples, k):
    return samples[samples['K'] == k]['obj_val'].mean()


def compute_cvar_prob(samples, pep, dro, k, alpha=0.05):
    dro_bound = dro[dro['K'] == k]['dro_feas_sol'].iloc[0]
    # dro_bound = dro[dro['K'] == k]['mro_sol'].iloc[0]
    count = 0
    for g in range(groups):
        idx_low = g * num_per_group
        idx_high = g * num_per_group + num_per_group

        samples_g = samples[(idx_low <= samples['i']) & (samples['i'] < idx_high) & (samples['K'] == k)]
        quantile = samples_g['obj_val'].quantile(1-alpha)
        tail_loss = samples_g[samples_g['obj_val'] >= quantile]
        cvar = tail_loss['obj_val'].mean()

        if cvar < dro_bound:
            count += 1

    return count / groups


def compute_empirical_cvar(samples, k, alpha=DEFAULT_ALPHA):
    """Mean of the worst (largest) alpha-fraction.

    Must match experiment_classes/lasso.py:compute_empirical_cvar, which
    produces the cvar_<alpha> columns of sample_summary_dist.csv used as the
    cross-validation threshold. A quantile-based variant disagrees with it
    whenever alpha*n is not an integer (e.g. top-10 vs top-9 at alpha=0.05,
    n=200).
    """
    vals = samples.loc[samples['K'] == k, 'obj_val'].to_numpy()
    n_tail = max(1, int(np.ceil(alpha * len(vals))))
    return float(np.mean(np.sort(vals)[-n_tail:]))


ISTA_samples = pd.read_csv('data/samples/ISTA_1_50/samples.csv')
FISTA_samples = pd.read_csv('data/samples/FISTA_1_50/samples.csv')
ISTA_worst_cases = ISTA_samples[['K', 'obj_val']].groupby(['K']).max()
FISTA_worst_cases = FISTA_samples[['K', 'obj_val']].groupby(['K']).max()
# OptISTA_samples = pd.read_csv('data/samples/OptISTA_1_50/samples.csv')
# ISTA_samples = pd.read_csv('data/dro/ISTA_exp_1_50/samples.csv')
# FISTA_samples = pd.read_csv('data/dro/FISTA_exp_1_50/samples.csv')

# Across-repeat distributions of the per-K empirical summaries (for cross-validated eps choice).
ISTA_dist = pd.read_csv('data/samples/ISTA_1_50/sample_summary_dist.csv')
FISTA_dist = pd.read_csv('data/samples/FISTA_1_50/sample_summary_dist.csv')

ISTA_pep = pd.read_csv('data/pep/ISTA_1_50/pep.csv')
FISTA_pep = pd.read_csv('data/pep/FISTA_1_50/pep.csv')
# OptISTA_pep = pd.read_csv('data/pep/OptISTA_1_50/pep.csv')

ISTA_exp_dro = pd.read_csv('data/dro/ISTA_exp_1_50/dro.csv')
ISTA_cvar_dro = pd.read_csv('data/dro/ISTA_cvar_1_50/dro.csv')
FISTA_exp_dro = pd.read_csv('data/dro/FISTA_exp_1_50/dro.csv')
FISTA_cvar_dro = pd.read_csv('data/dro/FISTA_cvar_1_50/dro.csv')
# OptISTA_exp_dro = pd.read_csv('data/dro/OptISTA_exp_1_50/dro.csv')
# OptISTA_cvar_dro = pd.read_csv('data/dro/OptISTA_cvar_1_50/dro.csv')


def main_bounds_alg(alpha=DEFAULT_ALPHA, out_path='Lasso_all.pdf'):
    """
    Generates and saves a plot comparing ISTA and FGM algorithms
    across worst-case, expectation, and CVaR metrics at a single alpha level.
    """

    # --- Data Preparation ---
    # Cross-validate the eps choice per K: smallest DRO bound that exceeds the across-repeat
    # 95th quantile of the matching empirical metric (mean for expectation, cvar_<alpha> for CVaR).
    ISTA_exp_bound, ISTA_exp_eps = cross_val_bound(ISTA_exp_dro, ISTA_dist, 'mean', exp_K_max, label='ISTA exp')
    FISTA_exp_bound, FISTA_exp_eps = cross_val_bound(FISTA_exp_dro, FISTA_dist, 'mean', exp_K_max, label='FISTA exp')
    ISTA_cvar_bound, ISTA_cvar_eps = cross_val_bound(ISTA_cvar_dro, ISTA_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'ISTA cvar a={alpha}')
    FISTA_cvar_bound, FISTA_cvar_eps = cross_val_bound(FISTA_cvar_dro, FISTA_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'FISTA cvar a={alpha}')

    # Compute empirical (sample) expectation and CVaR values
    ISTA_exp_k = []
    FISTA_exp_k = []
    ISTA_cvar_k = []
    FISTA_cvar_k = []

    for k in range(1, exp_K_max + 1):
        ISTA_exp_k.append(compute_empirical_avg(ISTA_samples, k))
        FISTA_exp_k.append(compute_empirical_avg(FISTA_samples, k))

    for k in range(1, cvar_K_max + 1):
        ISTA_cvar_k.append(compute_empirical_cvar(ISTA_samples, k, alpha))
        FISTA_cvar_k.append(compute_empirical_cvar(FISTA_samples, k, alpha))

    # Compute empirical (sample) worst-case values
    # ISTA_worst_cases = ISTA_samples[['K', 'obj_val']].groupby(['K']).max()
    # FISTA_worst_cases = FISTA_samples[['K', 'obj_val']].groupby(['K']).max()

    # Define colors for metrics
    worst_case_color = '#FFAA1C'
    exp_color = '#D81B60'
    cvar_color = 'tab:blue'

    # --- Plotting ---
    # Create 2 subplots (1 row, 2 columns)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 2)

    # --- Setup General Plot-wide Properties ---
    ax[0].set_ylabel(r'$f(x^K) - f^\star$')
    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    # Apply settings to both axes
    for axi in ax:
        axi.set_yscale('log')
        axi.grid(color='lightgray', alpha=0.3)
        set_log_xaxis(axi)

    # Share Y and X axes
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])

    # --- Subplot 0: (ISTA) ---
    ax[0].set_title('ISTA')

    # Worst-case
    ax[0].plot(range(1, exp_K_max + 1), ISTA_pep[ISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='Worst-case', marker='o', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=worst_case_color)
    # ax[0].plot(range(1, exp_K_max + 1), ISTA_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)
        
    # CVaR
    ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_bound, label='CVaR', marker='s', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=cvar_color)
    # ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)

    # Expectation
    ax[0].plot(range(1, exp_K_max + 1), ISTA_exp_bound, label='Expectation', marker='^', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=exp_color)
    # ax[0].plot(range(1, exp_K_max + 1), ISTA_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # --- Subplot 1: FISTA ---
    ax[1].set_title('FISTA')

    # Worst-case
    ax[1].plot(range(1, exp_K_max + 1), FISTA_pep[FISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='Worst-case', marker='o', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=worst_case_color)
    # ax[1].plot(range(1, exp_K_max + 1), FISTA_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_bound, label='Expectation', marker='^', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=exp_color)
    # ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # CVaR
    ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_bound, label='CVaR', marker='s', markevery=[0, 1, 3, 6, 10, 15, 19, 24], markersize=5, color=cvar_color)
    # ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)


    # --- Legend and Final Touches ---
    # Adjust subplot positions to make room for legend
    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.16, box.width, box.height - 0.2])

    # Get handles and labels from the first plot (they are identical for both)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncols=3)

    plt.suptitle('Lasso Minimization')

    # Save the data behind the plot next to the PDF.
    Ks = list(range(1, exp_K_max + 1))
    plot_data = pd.DataFrame({
        'K': Ks,
        'alpha': alpha,
        'ISTA_worst_bound': np.asarray(ISTA_pep[ISTA_pep['obj'] == 'obj_val']['val'])[:exp_K_max],
        'ISTA_worst_sample': ISTA_worst_cases['obj_val'].to_numpy()[:exp_K_max],
        'ISTA_exp_bound': ISTA_exp_bound, 'ISTA_exp_eps': ISTA_exp_eps, 'ISTA_exp_sample': ISTA_exp_k,
        'ISTA_cvar_bound': ISTA_cvar_bound, 'ISTA_cvar_eps': ISTA_cvar_eps, 'ISTA_cvar_sample': ISTA_cvar_k,
        'FISTA_worst_bound': np.asarray(FISTA_pep[FISTA_pep['obj'] == 'obj_val']['val'])[:exp_K_max],
        'FISTA_worst_sample': FISTA_worst_cases['obj_val'].to_numpy()[:exp_K_max],
        'FISTA_exp_bound': FISTA_exp_bound, 'FISTA_exp_eps': FISTA_exp_eps, 'FISTA_exp_sample': FISTA_exp_k,
        'FISTA_cvar_bound': FISTA_cvar_bound, 'FISTA_cvar_eps': FISTA_cvar_eps, 'FISTA_cvar_sample': FISTA_cvar_k,
    })
    plot_data.to_csv(out_path.rsplit('.', 1)[0] + '.csv', index=False)

    # Save the figure
    # plt.show()
    plt.savefig(out_path)
    plt.close(fig)


def main_bounds_all_alpha(out_path='Lasso_all_alpha.pdf'):
    """Overlay CVaR bound + sample curves for every alpha in ALPHA_VALS (ISTA and FISTA)."""
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylabel(r'$f(x^K) - f^\star$')
    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')
    for axi in ax:
        axi.set_yscale('log')
        axi.grid(color='lightgray', alpha=0.3)
        set_log_xaxis(axi)
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])
    ax[0].set_title('ISTA')
    ax[1].set_title('FISTA')

    cmap = plt.get_cmap('viridis')
    records = []
    for j, alpha in enumerate(ALPHA_VALS):
        color = cmap(j / max(1, len(ALPHA_VALS) - 1))

        ISTA_cvar_bound, ISTA_cvar_eps = cross_val_bound(ISTA_cvar_dro, ISTA_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'ISTA cvar a={alpha}')
        FISTA_cvar_bound, FISTA_cvar_eps = cross_val_bound(FISTA_cvar_dro, FISTA_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'FISTA cvar a={alpha}')
        ISTA_cvar_k = [compute_empirical_cvar(ISTA_samples, k, alpha) for k in range(1, cvar_K_max + 1)]
        FISTA_cvar_k = [compute_empirical_cvar(FISTA_samples, k, alpha) for k in range(1, cvar_K_max + 1)]

        for idx, k in enumerate(range(1, cvar_K_max + 1)):
            records.append({'alg': 'ISTA', 'alpha': alpha, 'K': k, 'cvar_bound': ISTA_cvar_bound[idx],
                            'cvar_eps': ISTA_cvar_eps[idx], 'cvar_sample': ISTA_cvar_k[idx]})
            records.append({'alg': 'FISTA', 'alpha': alpha, 'K': k, 'cvar_bound': FISTA_cvar_bound[idx],
                            'cvar_eps': FISTA_cvar_eps[idx], 'cvar_sample': FISTA_cvar_k[idx]})

        ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_bound,
                   label=rf'CVaR $\alpha={alpha}$ (Bound)', color=color)
        ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_k,
                   label=rf'CVaR $\alpha={alpha}$ (Sample)', linestyle='--', color=color)
        ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_bound,
                   label=rf'CVaR $\alpha={alpha}$ (Bound)', color=color)
        ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_k,
                   label=rf'CVaR $\alpha={alpha}$ (Sample)', linestyle='--', color=color)

    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.2, box.width, box.height - 0.2])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3)
    plt.suptitle('Lasso Minimization (CVaR across alpha)')
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == '__main__':
    # main()
    main_bounds_alg()
    main_bounds_all_alpha()
