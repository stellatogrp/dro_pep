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
    fig, ax = plt.subplots()
    sample_df = pd.read_csv('data/strong_cvx/samples.csv', header=None)
    theory_df = pd.read_csv('data/strong_cvx/theory.csv', header=None)
    samples = sample_df.to_numpy()
    theory = theory_df.to_numpy()
    
    ax.set_yscale('log')
    # ax.set_xscale('log')

    ax.set_ylabel(r'$\| x^K - x^\star\|$')
    ax.set_xlabel(r'$K$')

    average = np.mean(samples, axis=0)
    quantiles = np.percentile(samples, 99, axis=0)

    plt.plot(average, label='Sample average')
    plt.plot(quantiles, label='Sample 99th percentile')
    plt.plot(theory, label='Theoretical worst-case')
    plt.legend()

    # plt.show()
    plt.savefig('strong_cvx.pdf')


if __name__ == '__main__':
    main()
