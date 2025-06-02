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


def main():
    fig, ax = plt.subplots()

    pep_df = pd.read_csv('data/pep.csv')
    dro_df = pd.read_csv('data/dro.csv')

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$f(x^K) - f(x^\star)$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    K_vals = pep_df['K'].unique()
    print(K_vals)

    for k in K_vals:
        droK = dro_df[dro_df['K'] == k]

        plt.plot(droK['eps'], droK['dro_feas_sol'], label=f'K={k}')
        plt.plot(droK['eps'], droK['mro_sol'], label=f'MRO, K={k}')
    
    for k in K_vals:
        ax.axhline(y=pep_df[pep_df['K'] == k]['pep_obj'].iloc[0], color='black', linestyle='dashed')
    
    plt.grid(True, color='lightgray', alpha=0.3)
    plt.legend()
    plt.suptitle('Quadratic GD experiment')

    # plt.show()
    plt.savefig('simple_quad_gd.pdf')


if __name__ == '__main__':
    main()
