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

exp_K_max = 50
cvar_K_max = 50

num_eps_vals = 5

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


GD_samples = pd.read_csv('data/samples/grad_desc/samples.csv')
NGD_samples = pd.read_csv('data/samples/nesterov_grad_desc/samples.csv')

GD_pep = pd.read_csv('data/pep/grad_desc_1_68/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc_1_50/pep.csv')

GD_exp_dro = pd.read_csv('data/dro/grad_desc_exp_1_50/dro.csv')
GD_cvar_dro = pd.read_csv('data/dro/grad_desc_cvar_1_50/dro.csv')
NGD_exp_dro = pd.read_csv('data/dro/nesterov_grad_desc_exp_1_50/dro.csv')
NGD_cvar_dro = pd.read_csv('data/dro/nesterov_grad_desc_cvar_1_50/dro.csv')


def main_bounds():

    # ax[1].sharey(ax[0])

    GD_exp_dro_eps = GD_exp_dro[GD_exp_dro['eps_idx'] == 8]
    GD_cvar_dro_eps = GD_cvar_dro[GD_cvar_dro['eps_idx'] == 0]
    NGD_exp_dro_eps = NGD_exp_dro[NGD_exp_dro['eps_idx'] == 1]
    NGD_cvar_dro_eps = NGD_cvar_dro[NGD_cvar_dro['eps_idx'] == 0]

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

    fig, ax = plt.subplots(1, 3)

    ax[0].set_ylabel(r'$f(x^K) - f^\star$')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')

    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])

    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    ax[0].set_title('Worst-case')
    ax[1].set_title('Expectation')
    ax[2].set_title('CVar')

    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='GD')
    ax[1].plot(range(1, exp_K_max + 1), GD_exp_k, label='Sample', linestyle='--', color='black')
    ax[1].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Exp')

    ax[2].plot(range(1, cvar_K_max + 1), GD_cvar_k, label='Sample', linestyle='--', color='black')
    ax[2].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar')
    # # ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    ax[0].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='AGD')
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_k, label='Sample', linestyle='--', color='black')
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Exp')

    ax[2].plot(range(1, cvar_K_max + 1), NGD_cvar_k, label='Sample', linestyle='--', color='black')
    ax[2].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar')
    # ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    ax[0].legend()
    # plt.show()
    plt.savefig('huber_bound_plots_by_obj.pdf')


if __name__ == '__main__':
    # main()
    main_bounds()