import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (12, 6),
})

exp_K_max = 25
cvar_K_max = 25

num_eps_vals = 7

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


def compute_cvar_prob(samples, pep, dro, k, alpha=0.1):
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

ISTA_samples = pd.read_csv('data/samples/ISTA_1_40/samples.csv')
FISTA_samples = pd.read_csv('data/samples/FISTA_1_40/samples.csv')
OptISTA_samples = pd.read_csv('data/samples/OptISTA_1_40/samples.csv')

ISTA_pep = pd.read_csv('data/pep/ISTA_1_40/pep.csv')
FISTA_pep = pd.read_csv('data/pep/FISTA_1_40/pep.csv')
OptISTA_pep = pd.read_csv('data/pep/OptISTA_1_40/pep.csv')

ISTA_exp_dro = pd.read_csv('data/dro/ISTA_exp_1_25/dro.csv')
ISTA_cvar_dro = pd.read_csv('data/dro/ISTA_cvar_1_25/dro.csv')
FISTA_exp_dro = pd.read_csv('data/dro/FISTA_exp_1_25/dro.csv')
FISTA_cvar_dro = pd.read_csv('data/dro/FISTA_cvar_1_25/dro.csv')
OptISTA_exp_dro = pd.read_csv('data/dro/OptISTA_exp_1_25/dro.csv')
OptISTA_cvar_dro = pd.read_csv('data/dro/OptISTA_cvar_1_25/dro.csv')


def main_bounds():
    ISTA_color = 'tab:blue'
    FISTA_color = 'tab:green'
    OptISTA_color = 'tab:red'

    ISTA_exp_dro_eps = ISTA_exp_dro[ISTA_exp_dro['eps_idx'] == 6]
    ISTA_cvar_dro_eps = ISTA_cvar_dro[ISTA_cvar_dro['eps_idx'] == 0]
    FISTA_exp_dro_eps = FISTA_exp_dro[FISTA_exp_dro['eps_idx'] == 6]
    FISTA_cvar_dro_eps = FISTA_cvar_dro[FISTA_cvar_dro['eps_idx'] == 0]
    OptISTA_exp_dro_eps = OptISTA_exp_dro[OptISTA_exp_dro['eps_idx'] == 6]
    OptISTA_cvar_dro_eps = OptISTA_cvar_dro[OptISTA_cvar_dro['eps_idx'] == 0]

    ISTA_worst_k = []
    FISTA_worst_k = []
    OptISTA_worst_k = []

    ISTA_exp_k = []
    FISTA_exp_k = []
    OptISTA_exp_k = []
    ISTA_cvar_k = []
    FISTA_cvar_k = []
    OptISTA_cvar_k = []

    for k in range(1, exp_K_max + 1):
        ISTA_exp_k.append(compute_empirical_avg(ISTA_samples, k))
        FISTA_exp_k.append(compute_empirical_avg(FISTA_samples, k))
        OptISTA_exp_k.append(compute_empirical_avg(OptISTA_samples, k))
    
    for k in range(1, cvar_K_max + 1):
        ISTA_cvar_k.append(compute_empirical_cvar(ISTA_samples, k))
        FISTA_cvar_k.append(compute_empirical_cvar(FISTA_samples, k))
        OptISTA_cvar_k.append(compute_empirical_cvar(OptISTA_samples, k))

    ISTA_worst_cases = ISTA_samples[['K', 'obj_val']].groupby(['K']).max()
    FISTA_worst_cases = FISTA_samples[['K', 'obj_val']].groupby(['K']).max()
    OptISTA_worst_cases = OptISTA_samples[['K', 'obj_val']].groupby(['K']).max()

    fig, ax = plt.subplots(1, 3)

    ax[0].set_ylabel(r'$f(x^K) - f^\star$')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')

    ax[0].grid(color='lightgray', alpha=0.3)
    ax[1].grid(color='lightgray', alpha=0.3)
    ax[2].grid(color='lightgray', alpha=0.3)

    ax[0].set_xticks([5, 10, 15, 20, 25])

    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])

    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    ax[0].set_title('Worst-case')
    ax[1].set_title('Expectation')
    ax[2].set_title('CVar')

    ax[0].plot(range(1, exp_K_max + 1), ISTA_pep[ISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='ISTA', color=ISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), ISTA_worst_cases[:exp_K_max], label='Sample ISTA', linestyle='--', color=ISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), FISTA_pep[FISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='FISTA', color=FISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), FISTA_worst_cases[:exp_K_max], label='Sample FISTA', linestyle='--', color=FISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), OptISTA_pep[OptISTA_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='OptISTA', color=OptISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), OptISTA_worst_cases[:exp_K_max], label='Sample OptISTA', linestyle='--', color=OptISTA_color)

    ax[1].plot(range(1, exp_K_max + 1), ISTA_exp_k, label='Sample ISTA', linestyle='--', color=ISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), ISTA_exp_dro_eps['dro_feas_sol'][:exp_K_max], color=ISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_k, label='Sample FISTA', linestyle='--', color=FISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), FISTA_exp_dro_eps['dro_feas_sol'][:exp_K_max], color=FISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), OptISTA_exp_k, label='Sample OptISTA', linestyle='--', color=OptISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), OptISTA_exp_dro_eps['dro_feas_sol'][:exp_K_max], color=OptISTA_color)

    ax[2].plot(range(1, cvar_K_max + 1), ISTA_cvar_k, label='Sample ISTA', linestyle='--', color=ISTA_color)
    ax[2].plot(range(1, cvar_K_max + 1), ISTA_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], color=ISTA_color)
    ax[2].plot(range(1, cvar_K_max + 1), FISTA_cvar_k, label='Sample FISTA', linestyle='--', color=FISTA_color)
    ax[2].plot(range(1, cvar_K_max + 1), FISTA_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], color=FISTA_color)
    ax[2].plot(range(1, cvar_K_max + 1), OptISTA_cvar_k, label='Sample OptISTA', linestyle='--', color=OptISTA_color)
    ax[2].plot(range(1, cvar_K_max + 1), OptISTA_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], color=OptISTA_color)

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.05, box.width, box.height-.05])

    # ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=6)

    plt.suptitle('Lasso Minimization')
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f'Lasso_all.pdf')


if __name__ == '__main__':
    # main()
    main_bounds()
