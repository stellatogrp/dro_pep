import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (12, 7),
})

exp_K_max = 30
cvar_K_max = 30

num_eps_vals = 5

num_per_group = 100
groups = 100

METRIC = 'opt_dist_sq_norm'

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


def compute_empirical_cvar(samples, k, alpha=0.1):
    samples_k = samples[samples['K'] == k]
    quantile = samples_k[METRIC].quantile(1-alpha)
    tail_loss = samples_k[samples_k[METRIC] >= quantile]
    return tail_loss[METRIC].mean()


# precond = 'precond_avg'

GD_samples = pd.read_csv('data/samples/grad_desc_1_30/samples.csv')
NGD_samples = pd.read_csv('data/samples/nesterov_fgm_1_30/samples.csv')

GD_pep = pd.read_csv('data/pep/grad_desc_1_30/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_fgm_1_30/pep.csv')

# GD_exp_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_exp_1_40/dro.csv')
# GD_cvar_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_cvar_1_40/dro.csv')
# NGD_exp_dro = pd.read_csv(f'data/dro/{precond}/nesterov_fgm_exp_1_40/dro.csv')
# NGD_cvar_dro = pd.read_csv(f'data/dro/{precond}/nesterov_fgm_cvar_1_40/dro.csv')

GD_exp_dro = pd.read_csv(f'data/dro/grad_desc_exp_1_30/dro.csv')
GD_cvar_dro = pd.read_csv(f'data/dro/grad_desc_cvar_1_30/dro.csv')
NGD_exp_dro = pd.read_csv(f'data/dro/nesterov_fgm_exp_1_30/dro.csv')
NGD_cvar_dro = pd.read_csv(f'data/dro/nesterov_fgm_cvar_1_30/dro.csv')

GD_exp_fit_params = pd.read_csv(f'gd_exp_fit_params.csv')
GD_cvar_fit_params = pd.read_csv(f'gd_cvar_fit_params.csv')
NGD_exp_fit_params = pd.read_csv(f'ngd_exp_fit_params.csv')
NGD_cvar_fit_params = pd.read_csv(f'ngd_cvar_fit_params.csv')


def main_bounds():

    # ax[1].sharey(ax[0])
    GD_color = 'tab:blue'
    NGD_color = 'tab:orange'

    GD_exp_dro_eps = GD_exp_dro[GD_exp_dro['eps_idx'] == 5]
    GD_cvar_dro_eps = GD_cvar_dro[GD_cvar_dro['eps_idx'] == 3]
    NGD_exp_dro_eps = NGD_exp_dro[NGD_exp_dro['eps_idx'] == 5]
    NGD_cvar_dro_eps = NGD_cvar_dro[NGD_cvar_dro['eps_idx'] == 2]

    GD_exp_gamma = GD_exp_fit_params['gamma'].loc[0]
    GD_exp_C = GD_exp_fit_params['C'].loc[0]

    GD_cvar_gamma = GD_cvar_fit_params['gamma'].loc[0]
    GD_cvar_C = GD_cvar_fit_params['C'].loc[0]

    NGD_exp_gamma = NGD_exp_fit_params['gamma'].loc[0]
    NGD_exp_C = NGD_exp_fit_params['C'].loc[0]

    NGD_cvar_gamma = NGD_cvar_fit_params['gamma'].loc[0]
    NGD_cvar_C = NGD_cvar_fit_params['C'].loc[0]

    GD_worst_k = []
    NGD_worst_k = []

    GD_exp_k = []
    NGD_exp_k = []
    GD_cvar_k = []
    NGD_cvar_k = []

    for k in range(1, exp_K_max + 1):
        GD_exp_k.append(compute_empirical_avg(GD_samples, k))
        NGD_exp_k.append(compute_empirical_avg(NGD_samples, k))
    
    for k in range(1, cvar_K_max + 1):
        GD_cvar_k.append(compute_empirical_cvar(GD_samples, k))
        NGD_cvar_k.append(compute_empirical_cvar(NGD_samples, k))

    GD_worst_cases = GD_samples[['K', METRIC]].groupby(['K']).max()
    NGD_worst_cases = NGD_samples[['K', METRIC]].groupby(['K']).max()

    fig, ax = plt.subplots(1, 3)

    # ax[0].set_ylabel(r'$f(x^K) - f^\star$')
    ax[0].set_ylabel(r'$\|x^K - x^\star \|^2$')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')

    ax[0].grid(color='lightgray', alpha=0.3)
    ax[1].grid(color='lightgray', alpha=0.3)
    ax[2].grid(color='lightgray', alpha=0.3)

    ax[0].set_xticks([10, 20, 30, 40])

    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])

    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    ax[0].set_title('Worst-case')
    ax[2].set_title('Expectation')
    ax[1].set_title('CVaR')

    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == METRIC]['val'][:exp_K_max], label='GD DRO Bound', color=GD_color, linewidth=3)
    ax[0].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == METRIC]['val'][:exp_K_max], label='FGM DRO Bound', color=NGD_color)
    ax[0].plot(range(1, exp_K_max + 1), GD_worst_cases[:exp_K_max], label='GD Sample', linestyle='--', color=GD_color)

    # ax[1].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['mro_sol'][:exp_K_max], label='Exp', color=GD_color)

    exp_K = np.arange(1, exp_K_max + 1)
    cvar_K = np.arange(1, cvar_K_max + 1)
    ax[1].plot(cvar_K, GD_cvar_C * np.power(cvar_K, -GD_cvar_gamma), linestyle='dotted', color=GD_color)

    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_k, label='Sample', linestyle='--', color=GD_color)
    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar', color=GD_color)
    # # ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    ax[0].plot(range(1, exp_K_max + 1), NGD_worst_cases[:exp_K_max], label='FGM Sample', linestyle='--', color=NGD_color)
    # ax[1].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['mro_sol'][:exp_K_max], label='Exp', color=NGD_color)

    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_k, label='Sample', linestyle='--', color=NGD_color)
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar', color=NGD_color)
    # ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    ax[1].plot(cvar_K, NGD_cvar_C * np.power(cvar_K, -NGD_cvar_gamma), linestyle='dotted', color=NGD_color)

    ax[2].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='GD DRO Bound', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='FGM DRO Bound', color=NGD_color)
    ax[2].plot(range(1, exp_K_max + 1), GD_exp_k, label='GD Sample', linestyle='--', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_k, label='FGM Sample', linestyle='--', color=NGD_color)
    ax[2].plot(exp_K, GD_exp_C * np.pow(exp_K, -GD_exp_gamma), label='Fitted GD Curve', linestyle='dotted', color=GD_color)
    ax[2].plot(exp_K, NGD_exp_C * np.pow(exp_K, -NGD_exp_gamma), label='Fitted FGM Curve', linestyle='dotted', color=NGD_color)

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.1, box.width, box.height-.1])

    # ax[0].legend()
    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3)

    plt.suptitle('Nonstrongly Convex Quadratic Minimization, Distance to Optimality')

    plt.show()
    # plt.savefig(f'quad_nonstrongcvx.pdf')


def main_bounds_alg():
    """
    Generates and saves a plot comparing GD and FGM algorithms
    across worst-case, expectation, and CVaR metrics.
    """

    # --- Data Preparation ---
    # Select specific epsilon index for DRO bounds
    GD_exp_dro_eps = GD_exp_dro[GD_exp_dro['eps_idx'] == 5]
    GD_cvar_dro_eps = GD_cvar_dro[GD_cvar_dro['eps_idx'] == 3]
    NGD_exp_dro_eps = NGD_exp_dro[NGD_exp_dro['eps_idx'] == 5]
    NGD_cvar_dro_eps = NGD_cvar_dro[NGD_cvar_dro['eps_idx'] == 2]

    # Compute empirical (sample) expectation and CVaR values
    GD_exp_k = []
    NGD_exp_k = []
    GD_cvar_k = []
    NGD_cvar_k = []

    for k in range(1, exp_K_max + 1):
        GD_exp_k.append(compute_empirical_avg(GD_samples, k))
        NGD_exp_k.append(compute_empirical_avg(NGD_samples, k))
    
    for k in range(1, cvar_K_max + 1):
        GD_cvar_k.append(compute_empirical_cvar(GD_samples, k))
        NGD_cvar_k.append(compute_empirical_cvar(NGD_samples, k))

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
    ax[0].set_ylabel(r'$\|x^K - x^\star \|^2$')
    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    # Apply settings to both axes
    for axi in ax:
        axi.set_yscale('log')
        axi.grid(color='lightgray', alpha=0.3)
        axi.set_xticks([10, 20, 30, 40])
    
    # Share Y and X axes
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])

    # --- Subplot 0: Gradient Descent (GD) ---
    ax[0].set_title('Gradient Descent (GD)')

    # Worst-case
    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    ax[0].plot(range(1, exp_K_max + 1), GD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # CVaR
    ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVaR (Bound)', color=cvar_color)
    ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)
    
    # Expectation
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (Bound)', color=exp_color)
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # --- Subplot 1: Fast Gradient Method (FGM) ---
    ax[1].set_title('Fast Gradient Method (FGM)')

    # Worst-case
    ax[1].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    ax[1].plot(range(1, exp_K_max + 1), NGD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (Bound)', color=exp_color)
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # CVaR
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVaR (Bound)', color=cvar_color)
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)


    # --- Legend and Final Touches ---
    # Adjust subplot positions to make room for legend
    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.07, box.width, box.height - 0.07])

    # Get handles and labels from the first plot (they are identical for both)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncols=3)

    plt.suptitle('Nonstrongly Convex Quadratic Minimization, Distance to Optimality')

    # Save the figure
    # plt.show()
    plt.savefig(f'quad_nonstrongcvx.pdf')


if __name__ == '__main__':
    # main()
    # main_bounds()
    main_bounds_alg()