import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 17, # default: 14
    "figure.figsize": (9, 6),
    "lines.linewidth": 2.5,
})


def main():
    # colorblind-safe: #D81B60 (magenta) #FFC107 (yellow) #004D40 (dark green) #1E88E5 (blue)
    pep_color = 'tab:gray' # 'tag:green'
    # sm_color = 'tab:orange'
    p90_color = 'k'  # 'tab:gray'
    # avg_color = 'tab:blue'
    avg_color = 'tab:green' # 'tab:blue'

    fig, ax = plt.subplots()
    sample_df = pd.read_csv('data/samples.csv', header=None)
    theory_df = pd.read_csv('data/theory.csv', header=None)
    # scaled_theory_df = pd.read_csv('data/scaled_theory.csv', header=None)
    samples = sample_df.to_numpy()
    theory = theory_df.to_numpy()
    # scaled_theory = scaled_theory_df.to_numpy()
    
    ax.set_yscale('log')
    # ax.set_xscale('log')

    # ax.set_ylabel(r'$\| x^K - x^\star\|$')
    ax.set_ylabel(r'$f(x^K) - f(x^\star)$')
    # ax.set_ylabel(r'$f(z^K) - f(z^\star)$')
    ax.set_xlabel(r'$K$')

    average = np.mean(samples, axis=0)
    quantiles = np.percentile(samples, 90, axis=0)
    sample_max = np.max(samples, axis=0)

    plt.title('Projected Gradient Descent for Box-constrained Least Squares')

    K_max = 100
    K_vals = range(1, K_max+1)

    worst_case,  = ax.plot(K_vals, theory, label='Theoretical worst-case', color=pep_color)
    # plt.plot(K_vals, sample_max, label='Sample worst-case', color=sm_color, linestyle='dashdot')
    percentile, = ax.plot(K_vals, quantiles, label='Sample 90th percentile', color=p90_color, linestyle='dotted')
    average, = ax.plot(K_vals, average, label='Sample average', color=avg_color, linestyle='dashed')
    # plt.plot(K_vals, scaled_theory)

    ax.legend(handles=[worst_case, percentile, average])

    ax.grid(color='lightgray', alpha=0.3)

    # plt.show()
    plt.savefig('box_pgd_new2.pdf')


if __name__ == '__main__':
    main()
