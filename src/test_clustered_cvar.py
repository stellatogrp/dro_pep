import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from dro_reformulator import DROReformulator as OldReformulator
from reformulator.dro_reformulator import DROReformulator as NewReformulator
from tqdm import tqdm
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, SmoothStronglyConvexQuadraticFunction
from scipy.stats import ortho_group

from algorithm import gradient_descent, nesterov_accelerated_gradient
from generate_sample import generate_P, generate_P_beta, generate_trajectories, marchenko_pastur

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def generate_P(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = np.zeros(d)
    sigma[0] = mu
    sigma[-1] = L
    sigma[1:d-1] = np.random.uniform(low=mu, high=L, size=(d-2, ))

    return U @ np.diag(sigma) @ U.T


def main():

    seed = 0
    mu = 1
    L = 10
    R = 1
    t = 0.1
    K_max = 10

    d = 5
    N = 30

    params = {'N': N, 'd': d, 'mu': mu, 'L': L, 'R': R, 'K_max': K_max, 'K': 0, 't': t}

    np.random.seed(seed)

    problem = PEP()
    # could do SmoothStronglyConvexQuadraticFunction if we want
    # func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L)

    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    x = x0
    for _ in range(K_max):
        x = x - t * func.gradient(x)

    problem.set_performance_metric(func(x) - fs)

    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')

    print('tau from pepit:', pepit_tau)

    x0 = np.zeros(d)
    x0[0] = R
    # trajectories = generate_trajectories(N, d, mu, L, R, t, K, x0, traj_seed=1)
    algorithm = gradient_descent
    # matrix_generation = marchenko_pastur
    matrix_generation = generate_P

    trajectories, avg_trajectories = generate_trajectories(params, x0, algorithm, matrix_generation, traj_seed=1)

    CVar_DR = NewReformulator(
        problem,
        trajectories,
        'expectation',
        'clarabel',
        precond=False,
        mro_clusters=10,
    )

    NonMRO_CVar_DR = NewReformulator(
        problem,
        trajectories,
        'expectation',
        'clarabel',
        precond=False,
        mro_clusters=None,
    )

    eps_vals = np.logspace(-3, 1, num=10)
    alpha = 0.1

    objs = []
    mro_diffs = []
    objs_with_mro = []
    true_objs = []
    for eps in eps_vals:
        CVar_DR.set_params(eps=eps, alpha=alpha)
        out = CVar_DR.solve()
        objs.append(out['obj'])
        mro_diff = CVar_DR.extract_mro_diff()
        mro_diffs.append(mro_diff)
        objs_with_mro.append(out['obj'] + mro_diff)

        NonMRO_CVar_DR.set_params(eps=eps, alpha=alpha)
        out = NonMRO_CVar_DR.solve()
        true_objs.append(out['obj'])

    print(objs)
    print(mro_diffs)
    print(objs_with_mro)
    print(true_objs)

    plt.figure()
    # plt.plot(eps_vals, res, label='full samples')
    plt.plot(eps_vals, objs, label='mro obj values')
    plt.plot(eps_vals, objs_with_mro, label='bound with mro delta')
    plt.plot(eps_vals, true_objs, label='full sample bound')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    # plt.axhline(y=sample_obj, color='gray', linestyle='--', label='used sample')
    # plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')
    # # plt.plot(eps_vals, np.sqrt(np.array(eps_vals)), linestyle='--', color='gold')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('DRO obj value')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.title(fr'$N$={N}, 10 clusters, $K$={K_max}')

    plt.tight_layout()
    # plt.savefig(f'../plots/expectation/N{N}K{K}.pdf')
    plt.show()

    # plt.clf()
    # plt.cla()
    # plt.close()


def old_mro_main():
    seed = 0
    mu = 1
    L = 10
    R = 1
    t = 0.1
    K_max = 5

    d = 5
    N = 100

    params = {'N': N, 'd': d, 'mu': mu, 'L': L, 'R': R, 'K_max': K_max, 'K': 0, 't': t}

    np.random.seed(seed)

    problem = PEP()
    # could do SmoothStronglyConvexQuadraticFunction if we want
    # func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L)

    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    x = x0
    for _ in range(K_max):
        x = x - t * func.gradient(x)

    problem.set_performance_metric(func(x) - fs)

    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')

    print('tau from pepit:', pepit_tau)

    x0 = np.zeros(d)
    x0[0] = R
    # trajectories = generate_trajectories(N, d, mu, L, R, t, K, x0, traj_seed=1)
    algorithm = gradient_descent
    # matrix_generation = marchenko_pastur
    matrix_generation = generate_P

    trajectories, avg_trajectories = generate_trajectories(params, x0, algorithm, matrix_generation, traj_seed=1)

    CVar_DR = OldReformulator(
        problem,
        trajectories,
        'cvar_mro',
        'cvxpy',
        precond=True,
    )

    eps = 0.1
    alpha = 0.1

    out = CVar_DR.solve_single_alpha_eps_val(alpha, eps)
    print(out)


if __name__ == '__main__':
    main()
    # old_mro_main()
