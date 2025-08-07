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

exp_K_max = 40
cvar_K_max = 40

pep_K_max = 40
exp_K_max = 40
cvar_K_max = 40

GD_pep = pd.read_csv('data/pep/grad_desc_1_40/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_grad_desc_1_40/pep.csv')

GD_exp_dro = pd.read_csv('data/dro/grad_desc_exp_1_40/dro.csv')
GD_cvar_dro = pd.read_csv('data/dro/grad_desc_cvar_1_40/dro.csv')
NGD_exp_dro = pd.read_csv('data/dro/nesterov_grad_desc_exp_1_40/dro.csv')
NGD_cvar_dro = pd.read_csv('data/dro/nesterov_grad_desc_cvar_1_40/dro.csv')


def plot_times():
    GD_color = 'tab:blue'
    NGD_color = 'tab:green'

    GD_pep_times = GD_pep[GD_pep['obj'] == 'obj_val']['solvetime'][:pep_K_max]
    NGD_pep_times = NGD_pep[NGD_pep['obj'] == 'obj_val']['solvetime'][:pep_K_max]

    GD_exp_times = GD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    GD_cvar_times = GD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_exp_times = NGD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_cvar_times = NGD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]

    fig, ax = plt.subplots(1, 3)

    ax[0].set_ylabel('Solve time(s)')
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

    ax[0].plot(range(1, exp_K_max + 1), GD_pep_times, label='GD', color=GD_color)
    ax[0].plot(range(1, exp_K_max + 1), NGD_pep_times, label='NGD', color=NGD_color)
    
    ax[2].plot(range(1, exp_K_max + 1), GD_exp_times, label='GD', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_times, label='NGD', color=NGD_color)

    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_times, label='GD', color=GD_color)
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_times, label='NGD', color=NGD_color)

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.05, box.width, box.height-.05])

    # ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4)

    plt.suptitle('Logistic Regression Minimization, Solve times')

    # plt.show()

    plt.savefig('logreg_times.pdf')


if __name__ == '__main__':
    plot_times()