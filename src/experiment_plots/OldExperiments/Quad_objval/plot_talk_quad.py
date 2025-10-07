import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 28,
    "figure.figsize": (12, 8),
    })

exp_K_max = 40
cvar_K_max = 40

def main_bounds():
    avg_color = 'tab:purple'
    pep_color = 'tab:green'
    sm_color = 'tab:red'

    fig, ax = plt.subplots()
    sample_df = pd.read_csv('data/samples/grad_desc_1_50/samples.csv')
    theory_df = pd.read_csv('data/pep/grad_desc_1_40/pep.csv')

    theory_df = theory_df[theory_df['obj'] == 'obj_val']['val']
    average = sample_df.groupby('K')['obj_val'].mean().to_numpy()
    sample_max = sample_df.groupby('K')['obj_val'].max().to_numpy()
    theory = theory_df.to_numpy()
    
    ax.set_yscale('log')
    # ax.set_xscale('log')

    # ax.set_ylabel(r'$\| x^K - x^\star\|$')
    ax.set_ylabel(r'$f(x^K) - f(x^\star)$')
    ax.set_xlabel(r'$K$')

    # average = np.mean(samples, axis=1)
    # quantiles = np.percentile(samples, 99, axis=1)
    # sample_max = np.max(samples, axis=1)

    plt.title('Gradient Descent for Quadratic Minimization')

    K_max = 40
    K_vals = range(1, K_max+1)

    plt.plot(K_vals, average, label='Sample average', color=avg_color)
    # plt.plot(K_vals, quantiles, label='Sample 99th percentile')
    plt.plot(K_vals, sample_max, label='Sample worst-case', color=sm_color)
    plt.plot(K_vals, theory, label='Theoretical worst-case', color=pep_color)
    # plt.plot(K_vals, scaled_theory)
    plt.legend()

    ax.grid(color='lightgray', alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig('quad_gap.pdf')


if __name__ == '__main__':
    # main()
    main_bounds()