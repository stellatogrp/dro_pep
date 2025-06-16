import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (9, 6),
})


def main():
    avg_color = 'tab:purple'
    pep_color = 'tab:green'
    sm_color = 'tab:red'

    fig, ax = plt.subplots()
    sample_df = pd.read_csv('data/samples.csv', header=None)
    theory_df = pd.read_csv('data/theory.csv', header=None)
    scaled_theory_df = pd.read_csv('data/scaled_theory.csv', header=None)
    samples = sample_df.to_numpy()
    theory = theory_df.to_numpy()
    scaled_theory = scaled_theory_df.to_numpy()
    
    ax.set_yscale('log')
    # ax.set_xscale('log')

    # ax.set_ylabel(r'$\| x^K - x^\star\|$')
    ax.set_ylabel(r'$f(x^K) - f(x^\star)$')
    ax.set_xlabel(r'$K$')

    average = np.mean(samples, axis=0)
    quantiles = np.percentile(samples, 99, axis=0)
    sample_max = np.max(samples, axis=0)

    plt.title('Projected Gradient Descent for Box-constrained Least Squares')

    K_max = 100
    K_vals = range(1, K_max+1)

    plt.plot(K_vals, average, label='Sample average', color=avg_color)
    # plt.plot(K_vals, quantiles, label='Sample 99th percentile')
    plt.plot(K_vals, sample_max, label='Sample worst-case', color=sm_color)
    plt.plot(K_vals, theory, label='Theoretical worst-case', color=pep_color)
    # plt.plot(K_vals, scaled_theory)
    plt.legend()

    ax.grid(color='lightgray', alpha=0.3)

    # plt.show()
    plt.savefig('box_pgd.pdf')


if __name__ == '__main__':
    main()
