import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from dro_reformulator import DROReformulator
# from reformulator.dro_reformulator import DROReformulator
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


# def generate_trajectories(N, d, mu, L, R, t, K, x0, traj_seed=1):
#     '''
#     Ghalf = [x^star, x0, g0, g1]
#     and
#     F = [f^star, f0, f1, tau] where tau = f1 - f^star
#     '''
#     np.random.seed(traj_seed)
#     out = []
#     for _ in range(N):
#         P = generate_P(d, mu, L)

#         def func(x):
#             return .5 * x.T @ P @ x
        
#         def grad(x):
#             return P @ x

#         Ghalf = np.zeros((d, K + 3))
#         F = np.zeros(K + 3)

#         Ghalf[:, 1] = x0
#         F[1] = func(x0)
        
#         Ghalf[:, 2] = grad(x0)

#         x = x0
#         for k in range(K):
#             x = x - t * grad(x)
#             F[k + 2] = func(x)
#             Ghalf[:, k + 3] = grad(x)

#         F[-1] = F[K + 1] - F[0]
#         out.append((Ghalf.T @ Ghalf, F))

#     return out


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

    DR = DROReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
        precond=True,
    )

    eps = 0.1
    out = DR.solve_single_eps_val(eps)
    print(out)

    CDR = DROReformulator(
        problem,
        trajectories,
        'expectation',
        'clarabel',
        precond=True,
    )

    # out = CDR.solve_single_eps_val(eps)
    out = CDR.setup_clarabel_expectation_problem(eps=eps)
    # CDR.set_params(eps=0.1, alpha=0.05)
    # out = CDR.solve()
    print(out)

    print('---testing cvar---')
    CVar_DR = DROReformulator(
        problem,
        trajectories,
        'cvar',
        'cvxpy',
        precond=True,
    )
    alpha = 0.1
    out = CVar_DR.solve_fixed_alpha_eps_vals(alpha, [eps])
    print('from cvxpy:', out)
    out = CVar_DR.setup_clarabel_cvar_problem(eps=eps, alpha=alpha)
    print(out)


if __name__ == '__main__':
    main()
