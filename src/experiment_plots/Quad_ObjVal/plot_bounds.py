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

METRIC = 'obj_val'

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

GD_samples = pd.read_csv('data/samples/grad_desc/samples.csv')
NGD_samples = pd.read_csv('data/samples/nesterov_grad_desc/samples.csv')

GD_pep = pd.read_csv('data/pep/grad_desc/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc/pep.csv')

# GD_exp_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_exp_1_40/dro.csv')
# GD_cvar_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_cvar_1_40/dro.csv')
# NGD_exp_dro = pd.read_csv(f'data/dro/{precond}/nesterov_fgm_exp_1_40/dro.csv')
# NGD_cvar_dro = pd.read_csv(f'data/dro/{precond}/nesterov_fgm_cvar_1_40/dro.csv')

GD_exp_dro = pd.read_csv(f'data/dro/grad_desc/dro.csv')
NGD_exp_dro = pd.read_csv(f'data/dro/nesterov_grad_desc/dro.csv')

GD_exp_fit_params = pd.read_csv(f'data/gd_exp_fit_params.csv')
NGD_exp_fit_params = pd.read_csv(f'data/ngd_exp_fit_params.csv')


def main_bounds_alg():
    """
    Generates and saves a plot comparing GD and FGM algorithms
    across worst-case, expectation, and CVaR metrics.
    """

    # --- Data Preparation ---
    # Select specific epsilon index for DRO bounds
    GD_exp_dro_eps = GD_exp_dro[GD_exp_dro['eps_idx'] == 4]
    NGD_exp_dro_eps = NGD_exp_dro[NGD_exp_dro['eps_idx'] == 4]

    # Compute empirical (sample) expectation values
    GD_exp_k = []
    NGD_exp_k = []

    for k in range(1, exp_K_max + 1):
        GD_exp_k.append(compute_empirical_avg(GD_samples, k))
        NGD_exp_k.append(compute_empirical_avg(NGD_samples, k))

    # Compute empirical (sample) worst-case values
    GD_worst_cases = GD_samples[['K', METRIC]].groupby(['K']).max()
    NGD_worst_cases = NGD_samples[['K', METRIC]].groupby(['K']).max()

    # df = pd.read_csv(cf)
    # rho = df['rho'].values[0]
    # gamma = df['gamma'].values[0]
    # C = df['C'].values[0]
    # eta = df['eta'].values[0]
    # fitted_vals = C * np.power(rho, K) * np.power(K+1, -gamma) * np.power(np.log(K+1), eta)

    K = np.arange(1, exp_K_max + 1)
    gd_rho = GD_exp_fit_params['rho'].values[0]
    gd_gamma = GD_exp_fit_params['gamma'].values[0]
    gd_C = GD_exp_fit_params['C'].values[0]
    gd_eta = GD_exp_fit_params['eta'].values[0]
    gd_fit = gd_C * np.power(gd_rho, K) * np.power(K+1, -gd_gamma) * np.power(np.log(K+1), gd_eta)

    ngd_rho = NGD_exp_fit_params['rho'].values[0]
    ngd_gamma = NGD_exp_fit_params['gamma'].values[0]
    ngd_C = NGD_exp_fit_params['C'].values[0]
    ngd_eta = NGD_exp_fit_params['eta'].values[0]
    ngd_fit = ngd_C * np.power(ngd_rho, K) * np.power(K+1, -ngd_gamma) * np.power(np.log(K+1), ngd_eta)

    # Define colors for metrics
    worst_case_color = '#FFAA1C'
    exp_color = '#D81B60'

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
        axi.set_xticks([10, 20, 30, 40])
    
    # Share Y and X axes
    ax[1].sharey(ax[0])
    ax[1].sharex(ax[0])

    # --- Subplot 0: Gradient Descent (GD) ---
    ax[0].set_title('Gradient Descent (GD)')

    # Worst-case
    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    # ax[0].plot(range(1, exp_K_max + 1), GD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (DRO Bound)', color=exp_color)
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)
    ax[0].plot(range(1, exp_K_max + 1), gd_fit, label='Expectation (Fitted Bound)', linestyle='dashdot', color=exp_color)

    # --- Subplot 1: Fast Gradient Method (FGM) ---
    ax[1].set_title('Fast Gradient Method (FGM)')

    # Worst-case
    ax[1].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == METRIC]['val'][:exp_K_max], label='Worst-case (Bound)', color=worst_case_color)
    # ax[1].plot(range(1, exp_K_max + 1), NGD_worst_cases[:exp_K_max], label='Worst-case (Sample)', linestyle='--', color=worst_case_color)

    # Expectation
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Expectation (Bound)', color=exp_color)
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_k, label='Expectation (Sample)', linestyle='--', color=exp_color)
    ax[1].plot(range(1, exp_K_max + 1), ngd_fit, label='Expectation (Fitted Bound)', linestyle='dashdot', color=exp_color)

    # --- Legend and Final Touches ---
    # Adjust subplot positions to make room for legend
    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0 + 0.07, box.width, box.height - 0.07])

    # Get handles and labels from the first plot (they are identical for both)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Create a single figure-level legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncols=2)

    plt.suptitle('Quadratic Minimization, Objective Value')

    # Save the figure
    # plt.show()
    plt.savefig(f'quad_mp_comp.pdf')


if __name__ == '__main__':
    # main()
    # main_bounds()
    main_bounds_alg()