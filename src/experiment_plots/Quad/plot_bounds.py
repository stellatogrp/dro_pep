import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, NullLocator

# Log-scale x-axis ticks (plain integer labels) for the iteration axis.
X_LOG_TICKS = [1, 2, 5, 10, 20, 40]


def set_log_xaxis(axi):
    """Log-scale the iteration (K) axis with readable integer ticks."""
    axi.set_xscale('log')
    axi.set_xticks(X_LOG_TICKS)
    axi.xaxis.set_major_formatter(ScalarFormatter())
    axi.xaxis.set_minor_formatter(NullFormatter())
    axi.xaxis.set_minor_locator(NullLocator())  # no stray log minor ticks


# Theory slope guide lines: drawn only over the asymptotic tail [K_THEORY_START, K_max],
# and floated THEORY_OFFSET x above the bound so they read as over-approximating slopes.
K_THEORY_START = 20
THEORY_OFFSET = 1.3
# Horizontal gap (in points) between the end of a guide line and its inline label.
# Increase to push the labels further into the whitespace when the axes are narrowed.
LABEL_DX_PTS = 4


def theory_line(bound, p, K_max, q=0, k_start=K_THEORY_START, offset=THEORY_OFFSET):
    """Guide y = C*K^{-p}*(log K)^q over [k_start, K_max], sitting `offset`x above `bound`.

    C is chosen so the line is tangent to the bound from above on this tail: it touches
    at the tightest K and lies >= offset*bound elsewhere. Read as the asymptotic rate
    over-approximating the bound (q>0 adds the log factor; q=0 is a pure power law).
    """
    Ks = np.arange(k_start, K_max + 1)
    bound = np.asarray(bound, dtype=float)[k_start - 1:K_max]
    shape = Ks ** (-p) * np.log(Ks) ** q
    C = offset * np.max(bound / shape)
    return Ks, C * shape


def rate_label(p, q):
    """LaTeX O(...) label for the K^{-p}(log K)^q rate."""
    s = rf'K^{{-{p:g}}}'
    if q == 1:
        s += r'\log K'
    elif q:
        s += rf'(\log K)^{{{q:g}}}'
    return rf'$\mathcal{{O}}({s})$'


def add_theory_line(axi, bound, p, K_max, color, q=0, offset=THEORY_OFFSET):
    """Plot a dotted theory slope guide over the tail and label it inline at the right end."""
    Ks, yt = theory_line(bound, p, K_max, q, offset=offset)
    axi.plot(Ks, yt, linestyle=':', color=color, linewidth=1.5)
    axi.annotate(rate_label(p, q), xy=(Ks[-1], yt[-1]),
                 xytext=(LABEL_DX_PTS, 0), textcoords='offset points',
                 color=color, fontsize=12, va='center', ha='left')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 17,
    "figure.figsize": (12, 5),
})

exp_K_max = 40  # eps-extension coverage; base grid only beyond
cvar_K_max = 40

num_eps_vals = 5

num_per_group = 100
groups = 100

METRIC = 'obj_val'

# CVaR confidence levels (must match cfg.alpha_vals used to generate dro.csv).
ALPHA_VALS = [0.01, 0.05, 0.10]
DEFAULT_ALPHA = 0.05  # matches the alpha stated in the paper for quad and Lasso

# Across-repeat coverage level used to cross-validate the eps choice. Independent of alpha
# (the within-experiment CVaR tail level) -- it is NOT 1 - alpha.
COVERAGE_QUANTILE = 0.95


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
    for k in range(1, K_max + 1):
        rows = dro[dro['K'] == k]
        if rows.empty:   # K not yet computed (partial pull): break the line
            bounds.append(np.nan)
            chosen_eps.append(np.nan)
            continue
        feas = rows[rows['dro_feas_sol'] >= thr.loc[k]]
        pick = (rows.loc[feas['dro_feas_sol'].idxmin()] if len(feas)
                else rows.loc[rows['dro_feas_sol'].idxmax()])
        bounds.append(float(pick['dro_feas_sol']))
        chosen_eps.append(float(pick['eps']))
    print(f"[cross_val eps] {label or metric_col}: {[round(e, 6) for e in chosen_eps]}")
    return bounds, chosen_eps


def compute_exp_prob(samples, pep, dro, k):
    dro_bound = dro[dro['K'] == k]['dro_feas_sol'].iloc[0]
    count = 0 
    for g in range(groups):
        idx_low = g * num_per_group
        idx_high = g * num_per_group + num_per_group

        samples_g = samples[(idx_low <= samples['i']) & (samples['i'] < idx_high) & (samples['K'] == k)]
        mean = samples_g[METRIC].mean()

        if mean < dro_bound:
            count += 1

    return count / groups


def compute_empirical_avg(samples, k):
    return samples[samples['K'] == k][METRIC].mean()


def compute_cvar_prob(samples, pep, dro, k, alpha=0.1):
    dro_bound = dro[dro['K'] == k]['dro_feas_sol'].iloc[0]
    # dro_bound = dro[dro['K'] == k]['mro_sol'].iloc[0]
    count = 0
    for g in range(groups):
        idx_low = g * num_per_group
        idx_high = g * num_per_group + num_per_group

        samples_g = samples[(idx_low <= samples['i']) & (samples['i'] < idx_high) & (samples['K'] == k)]
        quantile = samples_g[METRIC].quantile(1-alpha)
        tail_loss = samples_g[samples_g[METRIC] >= quantile]
        cvar = tail_loss[METRIC].mean()

        if cvar < dro_bound:
            count += 1

    return count / groups


def compute_empirical_cvar(samples, k, alpha=DEFAULT_ALPHA):
    samples_k = samples[samples['K'] == k]
    quantile = samples_k[METRIC].quantile(1-alpha)
    tail_loss = samples_k[samples_k[METRIC] >= quantile]
    return tail_loss[METRIC].mean()


# precond = 'precond_avg'

GD_samples = pd.read_csv('data/samples/grad_desc_1_50/samples.csv')
NGD_samples = pd.read_csv('data/samples/nesterov_grad_desc_1_50/samples.csv')

# GD_samples = pd.read_csv('data/dro/grad_desc_exp_1_50/samples.csv')
# NGD_samples = pd.read_csv('data/dro/nesterov_grad_desc_exp_1_50/samples.csv')

GD_pep = pd.read_csv('data/pep/grad_desc_1_50/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc_1_50/pep.csv')

# GD_exp_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_exp_1_50/dro.csv')
# GD_cvar_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_cvar_1_50/dro.csv')
# NGD_exp_dro = pd.read_csv(f'data/dro/{precond}/nesterov_grad_desc_exp_1_50/dro.csv')
# NGD_cvar_dro = pd.read_csv(f'data/dro/{precond}/nesterov_grad_desc_cvar_1_50/dro.csv')

GD_exp_dro = pd.read_csv(f'data/dro/grad_desc_exp_1_50/dro.csv')
GD_cvar_dro = pd.read_csv(f'data/dro/grad_desc_cvar_1_50/dro.csv')
NGD_exp_dro = pd.read_csv(f'data/dro/nesterov_grad_desc_exp_1_50/dro.csv')
NGD_cvar_dro = pd.read_csv(f'data/dro/nesterov_grad_desc_cvar_1_50/dro.csv')

# Across-repeat distributions of the per-K empirical summaries (for cross-validated eps choice).
GD_dist = pd.read_csv('data/samples/grad_desc_1_50/sample_summary_dist.csv')
NGD_dist = pd.read_csv('data/samples/nesterov_grad_desc_1_50/sample_summary_dist.csv')

# GD_exp_fit_params = pd.read_csv(f'gd_exp_fit_params.csv')
# GD_cvar_fit_params = pd.read_csv(f'gd_cvar_fit_params.csv')
# NGD_exp_fit_params = pd.read_csv(f'ngd_exp_fit_params.csv')
# NGD_cvar_fit_params = pd.read_csv(f'ngd_cvar_fit_params.csv')


def main_bounds_alg(alpha=DEFAULT_ALPHA, out_path='quad_nonstrongcvx.pdf'):
    """
    Generates and saves a plot comparing GD and FGM algorithms
    across worst-case, expectation, and CVaR metrics at a single alpha level.
    """

    # --- Data Preparation ---
    # Cross-validate the eps choice per K: smallest DRO bound that exceeds the across-repeat
    # 95th quantile of the matching empirical metric (mean for expectation, cvar_<alpha> for CVaR).
    GD_exp_bound, GD_exp_eps = cross_val_bound(GD_exp_dro, GD_dist, 'mean', exp_K_max, label='GD exp')
    NGD_exp_bound, NGD_exp_eps = cross_val_bound(NGD_exp_dro, NGD_dist, 'mean', exp_K_max, label='NGD exp')
    GD_cvar_bound, GD_cvar_eps = cross_val_bound(GD_cvar_dro, GD_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'GD cvar a={alpha}')
    NGD_cvar_bound, NGD_cvar_eps = cross_val_bound(NGD_cvar_dro, NGD_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'NGD cvar a={alpha}')

    # Compute empirical (sample) expectation and CVaR values
    GD_exp_k = []
    NGD_exp_k = []
    GD_cvar_k = []
    NGD_cvar_k = []

    for k in range(1, exp_K_max + 1):
        GD_exp_k.append(compute_empirical_avg(GD_samples, k))
        NGD_exp_k.append(compute_empirical_avg(NGD_samples, k))

    for k in range(1, cvar_K_max + 1):
        GD_cvar_k.append(compute_empirical_cvar(GD_samples, k, alpha))
        NGD_cvar_k.append(compute_empirical_cvar(NGD_samples, k, alpha))

    # Compute empirical (sample) worst-case values
    GD_worst_cases = GD_samples[['K', METRIC]].groupby(['K']).max()
    NGD_worst_cases = NGD_samples[['K', METRIC]].groupby(['K']).max()

    # Define colors for metrics
    worst_case_color = '#FFAA1C'
    exp_color = '#D81B60'
    cvar_color = 'tab:blue'

    # --- Plotting ---
    # Create 2 subplots (1 row, 2 columns)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 2)

    # --- Setup General Plot-wide Properties ---
    ax[0].set_ylabel(r'$f(x^K) - f(x^\star)$')
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

    # --- Subplot 0: Gradient Descent (GD) ---
    ax[0].set_title('Gradient Descent (GD)')

    # Worst-case
    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case', marker='o', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=worst_case_color)
    # ax[0].plot(range(1, exp_K_max + 1), GD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # CVaR
    ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_bound, label='CVaR', marker='s', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=cvar_color)
    # ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)

    # Expectation
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_bound, label='Expectation', marker='^', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=exp_color)
    # ax[0].plot(range(1, exp_K_max + 1), GD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # Theoretical asymptotic slope guides (GD only), over the tail and above each bound.
    GD_worst_bound_vals = np.asarray(GD_pep[GD_pep['obj'] == METRIC]['val'])[:exp_K_max]
    add_theory_line(ax[0], GD_worst_bound_vals, 1.0, exp_K_max, worst_case_color)
    add_theory_line(ax[0], GD_cvar_bound, 1.25, cvar_K_max, cvar_color)
    add_theory_line(ax[0], GD_exp_bound, 1.5, exp_K_max, exp_color, offset=1.12)

    # End the axis exactly at K_max; the inline theory-rate labels sit in the
    # figure margin (point offset) freed by the box.width shrink below.
    ax[0].set_xlim(right=exp_K_max)

    # --- Subplot 1: Fast Gradient Method (FGM) ---
    ax[1].set_title('Fast Gradient Method (FGM)')

    # Worst-case
    ax[1].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case', marker='o', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=worst_case_color)
    # ax[1].plot(range(1, exp_K_max + 1), NGD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_bound, label='Expectation', marker='^', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=exp_color)
    # ax[1].plot(range(1, exp_K_max + 1), NGD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # CVaR
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_bound, label='CVaR', marker='s', markevery=[0, 1, 3, 6, 11, 18, 27, 39], markersize=5, color=cvar_color)
    # ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)

    # Theoretical asymptotic slope guides (FGM): O(K^-2), O(K^-3 log K), O(K^-2.5).
    NGD_worst_bound_vals = np.asarray(NGD_pep[NGD_pep['obj'] == METRIC]['val'])[:exp_K_max]
    add_theory_line(ax[1], NGD_worst_bound_vals, 2.0, exp_K_max, worst_case_color)
    add_theory_line(ax[1], NGD_cvar_bound, 2.5, cvar_K_max, cvar_color)
    add_theory_line(ax[1], NGD_exp_bound, 3.0, exp_K_max, exp_color, q=1, offset=1.05)


    # --- Legend and Final Touches ---
    # Adjust subplot positions to make room for legend
    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.16, box.width * 0.9, box.height - 0.2])

    # Get handles and labels from the first plot (they are identical for both)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncols=3)

    # plt.suptitle('Nonstrongly Convex Quadratic Minimization, Distance to Optimality')
    plt.suptitle('Unconstrained Quadratic Minimization')

    # Save the data behind the plot next to the PDF.
    Ks = list(range(1, exp_K_max + 1))
    plot_data = pd.DataFrame({
        'K': Ks,
        'alpha': alpha,
        'GD_worst_bound': np.asarray(GD_pep[GD_pep['obj'] == METRIC]['val'])[:exp_K_max],
        'GD_worst_sample': GD_worst_cases[METRIC].to_numpy()[:exp_K_max],
        'GD_exp_bound': GD_exp_bound, 'GD_exp_eps': GD_exp_eps, 'GD_exp_sample': GD_exp_k,
        'GD_cvar_bound': GD_cvar_bound, 'GD_cvar_eps': GD_cvar_eps, 'GD_cvar_sample': GD_cvar_k,
        'NGD_worst_bound': np.asarray(NGD_pep[NGD_pep['obj'] == METRIC]['val'])[:exp_K_max],
        'NGD_worst_sample': NGD_worst_cases[METRIC].to_numpy()[:exp_K_max],
        'NGD_exp_bound': NGD_exp_bound, 'NGD_exp_eps': NGD_exp_eps, 'NGD_exp_sample': NGD_exp_k,
        'NGD_cvar_bound': NGD_cvar_bound, 'NGD_cvar_eps': NGD_cvar_eps, 'NGD_cvar_sample': NGD_cvar_k,
    })
    plot_data.to_csv(out_path.rsplit('.', 1)[0] + '.csv', index=False)

    # Save the figure
    # plt.show()
    plt.savefig(out_path)
    plt.close(fig)


def main_bounds_all_alpha(out_path='quad_nonstrongcvx_all_alpha.pdf'):
    """Overlay CVaR bound + sample curves for every alpha in ALPHA_VALS (GD and FGM)."""
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylabel(r'$f(x^K) - f(x^\star)$')
    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')
    for axi in ax:
        axi.set_yscale('log')
        axi.grid(color='lightgray', alpha=0.3)
        set_log_xaxis(axi)
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])
    ax[0].set_title('Gradient Descent (GD)')
    ax[1].set_title('Fast Gradient Method (FGM)')

    cmap = plt.get_cmap('viridis')
    records = []
    for j, alpha in enumerate(ALPHA_VALS):
        color = cmap(j / max(1, len(ALPHA_VALS) - 1))

        GD_cvar_bound, GD_cvar_eps = cross_val_bound(GD_cvar_dro, GD_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'GD cvar a={alpha}')
        NGD_cvar_bound, NGD_cvar_eps = cross_val_bound(NGD_cvar_dro, NGD_dist, f'cvar_{alpha}', cvar_K_max, alpha, label=f'NGD cvar a={alpha}')
        GD_cvar_k = [compute_empirical_cvar(GD_samples, k, alpha) for k in range(1, cvar_K_max + 1)]
        NGD_cvar_k = [compute_empirical_cvar(NGD_samples, k, alpha) for k in range(1, cvar_K_max + 1)]

        for idx, k in enumerate(range(1, cvar_K_max + 1)):
            records.append({'alg': 'GD', 'alpha': alpha, 'K': k, 'cvar_bound': GD_cvar_bound[idx],
                            'cvar_eps': GD_cvar_eps[idx], 'cvar_sample': GD_cvar_k[idx]})
            records.append({'alg': 'NGD', 'alpha': alpha, 'K': k, 'cvar_bound': NGD_cvar_bound[idx],
                            'cvar_eps': NGD_cvar_eps[idx], 'cvar_sample': NGD_cvar_k[idx]})

        ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_bound,
                   label=rf'CVaR $\alpha={alpha}$ (Bound)', color=color)
        ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_k,
                   label=rf'CVaR $\alpha={alpha}$ (Sample)', linestyle='--', color=color)
        ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_bound,
                   label=rf'CVaR $\alpha={alpha}$ (Bound)', color=color)
        ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_k,
                   label=rf'CVaR $\alpha={alpha}$ (Sample)', linestyle='--', color=color)

    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.2, box.width, box.height - 0.2])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3)
    plt.suptitle('Least Squares Minimization (CVaR across alpha)')
    pd.DataFrame(records).to_csv(out_path.rsplit('.', 1)[0] + '.csv', index=False)
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == '__main__':
    # main()
    # main_bounds()
    main_bounds_alg()
    main_bounds_all_alpha()