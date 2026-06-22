import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 17,
    "figure.figsize": (12, 5),
})

exp_K_max = 25
cvar_K_max = 25

# num_eps_vals = 7

num_per_group = 100
groups = 100

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


def compute_empirical_cvar(samples, k, alpha=0.1):
    samples_k = samples[samples['K'] == k]
    quantile = samples_k['obj_val'].quantile(1-alpha)
    tail_loss = samples_k[samples_k['obj_val'] >= quantile]
    return tail_loss['obj_val'].mean()


def select_dro_eps_by_K(dro, k_threshold=10, eps_idx_low=1, eps_idx_high=0):
    """Per-K DRO eps selection: use eps_idx_low for K <= k_threshold, eps_idx_high
    for K > k_threshold. Returns one row per K, sorted by K with index reset
    (so positional slicing [:K_max] aligns with range(1, K_max+1))."""
    low = dro[(dro['K'] <= k_threshold) & (dro['eps_idx'] == eps_idx_low)]
    high = dro[(dro['K'] > k_threshold) & (dro['eps_idx'] == eps_idx_high)]
    return pd.concat([low, high]).sort_values('K').reset_index(drop=True)

ISTA_samples = pd.read_csv('data/samples/ISTA_1_25/samples.csv')
FISTA_samples = pd.read_csv('data/samples/FISTA_1_25/samples.csv')
ISTA_worst_cases = ISTA_samples[['K', 'obj_val']].groupby(['K']).max()
FISTA_worst_cases = FISTA_samples[['K', 'obj_val']].groupby(['K']).max()
# OptISTA_samples = pd.read_csv('data/samples/OptISTA_1_25/samples.csv')
ISTA_samples = pd.read_csv('data/dro/ISTA_exp_1_25/samples.csv')
FISTA_samples = pd.read_csv('data/dro/FISTA_exp_1_25/samples.csv')

ISTA_pep = pd.read_csv('data/pep/ISTA_1_25/pep.csv')
FISTA_pep = pd.read_csv('data/pep/FISTA_1_25/pep.csv')
# OptISTA_pep = pd.read_csv('data/pep/OptISTA_1_25/pep.csv')

ISTA_exp_dro = pd.read_csv('data/dro/ISTA_exp_1_25/dro.csv')
ISTA_cvar_dro = pd.read_csv('data/dro/ISTA_cvar_1_25/dro.csv')
FISTA_exp_dro = pd.read_csv('data/dro/FISTA_exp_1_25/dro.csv')
FISTA_cvar_dro = pd.read_csv('data/dro/FISTA_cvar_1_25/dro.csv')
# OptISTA_exp_dro = pd.read_csv('data/dro/OptISTA_exp_1_25/dro.csv')
# OptISTA_cvar_dro = pd.read_csv('data/dro/OptISTA_cvar_1_25/dro.csv')


def main_bounds_alg():
    """
    Generates and saves a plot comparing ISTA and FGM algorithms
    across worst-case, expectation, and CVaR metrics.
    """

    # --- Data Preparation ---
    # Select specific epsilon index for DRO bounds
    ISTA_exp_dro_eps = ISTA_exp_dro[ISTA_exp_dro['eps_idx'] == 2]
    ISTA_cvar_dro_eps = ISTA_cvar_dro[ISTA_cvar_dro['eps_idx'] == 0]
    FISTA_exp_dro_eps = FISTA_exp_dro[FISTA_exp_dro['eps_idx'] == 0]
    FISTA_cvar_dro_eps = FISTA_cvar_dro[FISTA_cvar_dro['eps_idx'] == 0]

    # Compute empirical (sample) expectation and CVaR values
    ISTA_exp_k = []
    FISTA_exp_k = []
    ISTA_cvar_k = []
    FISTA_cvar_k = []

    for k in range(1, exp_K_max + 1):
        ISTA_exp_k.append(compute_empirical_avg(ISTA_samples, k))
        FISTA_exp_k.append(compute_empirical_avg(FISTA_samples, k))
    
    for k in range(1, cvar_K_max + 1):
        ISTA_cvar_k.append(compute_empirical_cvar(ISTA_samples, k))
        FISTA_cvar_k.append(compute_empirical_cvar(FISTA_samples, k))

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
        axi.set_xticks([10, 20, 30, 40])
    
    # Share Y and X axes
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])

    # --- Subplot 0: (ISTA) ---
    ax[0].set_title('ISTA')

    # Worst-case
    ax[0].plot(range(1, exp_K_max + 1), ISTA_pep[ISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    ax[0].plot(range(1, exp_K_max + 1), ISTA_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)
        
    # CVaR
    ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVaR (Bound)', color=cvar_color)
    ax[0].plot(range(1, cvar_K_max + 1), ISTA_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)
    
    # Expectation
    ax[0].plot(range(1, exp_K_max + 1), ISTA_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (Bound)', color=exp_color)
    ax[0].plot(range(1, exp_K_max + 1), ISTA_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # --- Subplot 1: FISTA ---
    ax[1].set_title('FISTA')

    # Worst-case
    ax[1].plot(range(1, exp_K_max + 1), FISTA_pep[FISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    ax[1].plot(range(1, exp_K_max + 1), FISTA_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (Bound)', color=exp_color)
    ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)

    # CVaR
    ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVaR (Bound)', color=cvar_color)
    ax[1].plot(range(1, cvar_K_max + 1), FISTA_cvar_k, label='CVaR (Sample)', linestyle='--', color=cvar_color)


    # --- Legend and Final Touches ---
    # Adjust subplot positions to make room for legend
    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.2, box.width, box.height - 0.2])

    # Get handles and labels from the first plot (they are identical for both)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncols=3)

    plt.suptitle('Lasso Minimization')

    # Save the figure
    # plt.show()
    plt.savefig(f'Lasso_all.pdf')


if __name__ == '__main__':
    # main()
    main_bounds_alg()
