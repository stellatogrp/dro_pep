import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (11, 5),
})


def main():
    # fig, ax = plt.subplots()

    # pep_df = pd.read_csv('data/pep.csv')
    # dro_df = pd.read_csv('data/dro.csv')

    # ax.set_xlabel(r'$\varepsilon$')
    # ax.set_ylabel(r'$f(x^K) - f(x^\star)$')

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # K_vals = pep_df['K'].unique()
    # print(K_vals)

    # for k in K_vals:
    #     droK = dro_df[dro_df['K'] == k]

    #     plt.plot(droK['eps'], droK['dro_feas_sol'], label=f'K={k}')
    #     plt.plot(droK['eps'], droK['mro_sol'], label=f'MRO, K={k}')
    
    # for k in K_vals:
    #     ax.axhline(y=pep_df[pep_df['K'] == k]['pep_obj'].iloc[0], color='black', linestyle='dashed')
    
    # plt.grid(True, color='lightgray', alpha=0.3)
    # plt.legend()
    # plt.suptitle('Quadratic GD experiment')

    # # plt.show()
    # # plt.savefig('simple_quad_gd.pdf')
    # plt.savefig('K1_5_10.pdf')

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    pep_df = pd.read_csv('data/pep.csv')
    dro_df = pd.read_csv('data/dro.csv')

    ax[0].set_xlabel(r'$\varepsilon$')
    ax[1].set_xlabel(r'$\varepsilon$')
    ax[2].set_xlabel(r'$\varepsilon$')

    ax[0].set_ylabel(r'$f(x^K) - f(x^\star)$')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    for axi in ax:
        axi.grid(color='lightgray', alpha=0.4)

    K_vals = pep_df['K'].unique()
    for i, k in enumerate(K_vals):
        droK = dro_df[(dro_df['K'] == k) & (dro_df['eps_idx'] > 0)]

        pep_bound = pep_df[pep_df['K'] == k]['pep_obj'].iloc[0]
        droK_epsbar = droK[droK['dro_feas_sol'] >= pep_bound - 1e-6]['eps'].min()

        ax[i].axhline(y=pep_bound, color='black', linestyle='dashed', label='Worst-case')
        ax[i].axhline(y=dro_df[dro_df['K'] == k]['dro_feas_sol'].iloc[0], color='black', linestyle='dashdot', label='Sample average')
        ax[i].axvline(x=droK_epsbar, color='black', linestyle='dotted', label=r'Experimental $\bar{\varepsilon}$')

        ax[i].plot(droK['eps'], droK['dro_feas_sol'])

        ax[i].set_title(fr'$K = {int(k)}$')
    
    # for (i,j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    #     ax_curr = ax[i, j]
    #     box = ax_curr.get_position()
    #     if j == 0:
    #         ax_curr.set_position([box.x0, box.y0, box.width*.875, box.height])
    #     else:
    #         ax_curr.set_position([box.x0-.05, box.y0, box.width*.875, box.height])

    for axi in ax:
        box = axi.get_position()
        axi.set_position([box.x0, box.y0+0.1, box.width, box.height * .85])
    
    # bbox_to_anchor = (0.5, 0.5)
    ax[1].legend(ncol=3, bbox_to_anchor = (1.7, -0.15))

    plt.suptitle(r'Average case DRO-PEP on Gradient Descent for Unconstrained QPs')
    # plt.show()
    plt.savefig('GD_QP_eps.pdf')


if __name__ == '__main__':
    main()
