import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "figure.figsize": (9, 6),
})


def main():
    fig, ax = plt.subplots()

    dro_df = pd.read_csv('talk_data/dro/dro.csv')
    samples_df = pd.read_csv('talk_data/samples/samples.csv')
    dro_samples_df = pd.read_csv('talk_data/dro/samples.csv')
    pep_df = pd.read_csv('talk_data/pep/pep.csv')

    # Calculate out-of-sample expectation (average of obj_val for K=10)
    oos_expectation = samples_df[samples_df['K'] == 10]['obj_val'].mean()

    # Calculate sample average from dro/samples.csv
    sample_average = dro_samples_df['obj_val'].mean()

    # Get PEP worst-case bound
    pep_bound = pep_df['val'].iloc[0]

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'DRO-PEP Objective')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(dro_df['eps'], dro_df['dro_feas_sol'], color='tab:blue', label='DRO-PEP objective')
    ax.axhline(y=oos_expectation, color='tab:orange', linestyle='--', label='Out-of-sample expectation')
    ax.axhline(y=sample_average, color='gray', linestyle='--', label='Sample average')
    ax.axhline(y=pep_bound, color='black', linestyle='--', label='Worst-case bound (PEP)')

    ax.legend()
    ax.grid(color='lightgray', alpha=0.4)

    plt.title(r'Gradient descent with $L$-smooth convex functions ($N=50$)')
    plt.tight_layout()
    plt.savefig('talk_quad.pdf')
    plt.show()


if __name__ == '__main__':
    main()
