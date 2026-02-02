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
    K_max = 5

    d = 5
    N = 5

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

    print('---testing expectation--')

    Old_DR = OldReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
        precond=True,
    )

    eps = 0.1
    out = Old_DR.solve_single_eps_val(eps)
    print('old version:', out)

    New_DR = NewReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
        precond=True,
    )

    New_DR.set_params(eps=eps)
    out = New_DR.solve()
    print('new version cvxpy:', out)
    data = New_DR.extract_solution()
    # print('lambda:', data['lambda'])
    # print('s:', data['s'])
    # print('y:', data['y'])
    # print('Gz:', data['Gz'])
    # print('Fz:', data['Fz'])
    # print('H:', data['H'])

    New_DR = NewReformulator(
        problem,
        trajectories,
        'expectation',
        'clarabel',
        precond=True,
    )

    # New_DR.set_single_eps_val(eps)
    New_DR.set_params(eps=eps)
    out = New_DR.solve()
    print('new version clarabel:', out)
    data = New_DR.extract_solution()
    # print('lambda:', data['lambda'])
    # print('s:', data['s'])
    # print('y:', data['y'])
    # print('Gz:', data['Gz'])
    # print('Fz:', data['Fz'])
    # print('H:', data['H'])

    print('---testing cvar---')
    OldCVar_DR = OldReformulator(
        problem,
        trajectories,
        'cvar',
        'cvxpy',
        precond=True,
    )
    alpha = 0.1
    out = OldCVar_DR.solve_single_alpha_eps_val(alpha, eps)
    print('old version:', out)

    NewCVar_DR = NewReformulator(
        problem,
        trajectories,
        'cvar',
        'cvxpy',
        precond=True,
    )

    NewCVar_DR.set_params(eps=eps, alpha=alpha)
    out = NewCVar_DR.solve()
    print('new version cvxpy:', out)
    data = NewCVar_DR.extract_solution()

    NewCVar_DR = NewReformulator(
        problem,
        trajectories,
        'cvar',
        'clarabel',
        precond=True,
    )

    NewCVar_DR.set_params(eps=eps, alpha=alpha)
    out = NewCVar_DR.solve()
    print('new version clarabel:', out)
    data = NewCVar_DR.extract_solution()


if __name__ == '__main__':
    main()
