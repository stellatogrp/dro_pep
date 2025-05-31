import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (10, 6),
})

pep_K_max = 40
exp_K_max = 40
cvar_K_max = 40

GD_pep = pd.read_csv('data/pep/grad_desc/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc/pep.csv')

GD_exp_dro = pd.read_csv('data/dro/grad_desc_exp_1_40/dro.csv')
GD_cvar_dro = pd.read_csv('data/dro/grad_desc_cvar_1_40/dro.csv')
NGD_exp_dro = pd.read_csv('data/dro/nesterov_grad_desc_exp_1_40/dro.csv')
NGD_cvar_dro = pd.read_csv('data/dro/nesterov_grad_desc_cvar_1_40/dro.csv')

PEP_OBJ = 'grad_sq_norm'


def plot_times():
    GD_pep_times = GD_pep[GD_pep['obj'] == PEP_OBJ]['solvetime'][:pep_K_max]
    NGD_pep_times = NGD_pep[NGD_pep['obj'] == PEP_OBJ]['solvetime'][:pep_K_max]

    GD_exp_times = GD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    GD_cvar_times = GD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_exp_times = NGD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_cvar_times = NGD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]

    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')
    ax[0].set_ylabel('Solve time (s)')

    ax[1].sharey(ax[0])

    ax[0].set_yscale('log')

    ax[0].set_title('GD')
    ax[1].set_title('AGD')

    ax[0].plot(range(1, pep_K_max + 1), GD_pep_times, label='PEP')
    ax[0].plot(range(1, exp_K_max + 1), GD_exp_times, label='Exp')
    ax[0].plot(range(1, cvar_K_max + 1), GD_cvar_times, label='CVar')

    ax[1].plot(range(1, pep_K_max + 1), NGD_pep_times, label='PEP')
    ax[1].plot(range(1, exp_K_max + 1), NGD_exp_times, label='Exp')
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_times, label='CVar')

    ax[0].legend()
    # plt.show()

    plt.savefig('quad_times.pdf')


if __name__ == '__main__':
    plot_times()
