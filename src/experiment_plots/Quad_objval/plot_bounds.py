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

exp_K_max = 35
cvar_K_max = 35

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


# precond = 'precond_avg'

GD_samples = pd.read_csv('data/samples/grad_desc_1_40/samples.csv')
NGD_samples = pd.read_csv('data/samples/nesterov_grad_desc_1_40/samples.csv')

GD_pep = pd.read_csv('data/pep/grad_desc_1_40/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc_1_40/pep.csv')

# GD_exp_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_exp_1_40/dro.csv')
# GD_cvar_dro = pd.read_csv(f'data/dro/{precond}/grad_desc_cvar_1_40/dro.csv')
# NGD_exp_dro = pd.read_csv(f'data/dro/{precond}/nesterov_grad_desc_exp_1_40/dro.csv')
# NGD_cvar_dro = pd.read_csv(f'data/dro/{precond}/nesterov_grad_desc_cvar_1_40/dro.csv')

GD_exp_dro = pd.read_csv(f'data/dro/grad_desc_exp_1_40/dro.csv')
GD_cvar_dro = pd.read_csv(f'data/dro/grad_desc_cvar_1_40/dro.csv')
NGD_exp_dro = pd.read_csv(f'data/dro/nesterov_grad_desc_exp_1_40/dro.csv')
NGD_cvar_dro = pd.read_csv(f'data/dro/nesterov_grad_desc_cvar_1_40/dro.csv')


def main_bounds():

    # ax[1].sharey(ax[0])
    GD_color = 'tab:blue'
    NGD_color = 'tab:green'

    GD_exp_dro_eps = GD_exp_dro[GD_exp_dro['eps_idx'] == 2]
    GD_cvar_dro_eps = GD_cvar_dro[GD_cvar_dro['eps_idx'] == 1]
    NGD_exp_dro_eps = NGD_exp_dro[NGD_exp_dro['eps_idx'] == 2]
    NGD_cvar_dro_eps = NGD_cvar_dro[NGD_cvar_dro['eps_idx'] == 1]

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

    GD_worst_cases = GD_samples[['K', 'obj_val']].groupby(['K']).max()
    NGD_worst_cases = NGD_samples[['K', 'obj_val']].groupby(['K']).max()

    fig, ax = plt.subplots(1, 3)

    ax[0].set_ylabel(r'$f(x^K) - f^\star$')
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
    ax[1].set_title('CVar')

    ax[0].plot(range(1, exp_K_max + 1), GD_pep[GD_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='GD', color=GD_color)
    ax[0].plot(range(1, exp_K_max + 1), GD_worst_cases[:exp_K_max], label='Sample GD', linestyle='--', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), GD_exp_k, label='Sample', linestyle='--', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Exp', color=GD_color)
    # ax[1].plot(range(1, exp_K_max + 1), GD_exp_dro_eps['mro_sol'][:exp_K_max], label='Exp', color=GD_color)

    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_k, label='Sample', linestyle='--', color=GD_color)
    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar', color=GD_color)
    # # ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    ax[0].plot(range(1, exp_K_max + 1), NGD_pep[NGD_pep['obj'] == 'obj_val']['val'][:exp_K_max], label='AGD', color=NGD_color)
    ax[0].plot(range(1, exp_K_max + 1), NGD_worst_cases[:exp_K_max], label='Sample AGD', linestyle='--', color=NGD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_k, label='Sample', linestyle='--', color=NGD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['dro_feas_sol'][:exp_K_max], label='Exp', color=NGD_color)
    # ax[1].plot(range(1, exp_K_max + 1), NGD_exp_dro_eps['mro_sol'][:exp_K_max], label='Exp', color=NGD_color)

    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_k, label='Sample', linestyle='--', color=NGD_color)
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['dro_feas_sol'][:cvar_K_max], label='CVar', color=NGD_color)
    # ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_dro_eps['mro_sol'][:cvar_K_max], label='CVar')

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.05, box.width, box.height-.05])

    # ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4)

    plt.suptitle('Quadratic Minimization, Objective Value')

    # plt.show()
    plt.savefig(f'quad_obj_val.pdf')


if __name__ == '__main__':
    # main()
    main_bounds()