import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from dro_reformulator import DROReformulator
from tqdm import tqdm
from PEPit import PEP
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from PEPit.functions import SmoothStronglyConvexFunction,SmoothStronglyConvexQuadraticFunction
from scipy.stats import ortho_group

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

def algorithm(func_eval, grad_eval, x0, xs, K_max, t):
    xk = x0
    gk = grad_eval(x0)
    fk = func_eval(x0)

    x = [xs, xk]
    g = [grad_eval(xs), gk]
    f = [func_eval(xs), fk]
    
    for k in range(K_max) :
        xk = xk - t * gk
        gk = grad_eval(xk)
        fk = func_eval(xk)
        x.append(xk)
        g.append(gk)
        f.append(fk)

    return x, g, f


def generate_samples(N, d, functions, gradients, algorithm, x0, K, t, traj_seed=1):

    out = []
    for (f, g) in zip(functions, gradients) :
        xs = np.zeros((d,))
        x, g, f = algorithm(f, g, x0, xs, K, t)
        G_half = np.hstack([x[:1], g])
        F = np.array(f)

        out.append((G_half.T @ G_half, F))
    pass


def generate_trajectories(N, d, mu, L, R, t, K, x0, traj_seed=1):
    '''
    Ghalf = [x^star, x0, g0, g1]
    and
    F = [f^star, f0, f1, tau] where tau = f1 - f^star
    '''
    np.random.seed(traj_seed)
    out = []
    for _ in range(N):
        P = generate_P(d, mu, L)

        def func(x):
            return .5 * x.T @ P @ x
        
        def grad(x):
            return P @ x

        Ghalf = np.zeros((d, K + 3))
        F = np.zeros(K + 3)

        Ghalf[:, 1] = x0
        F[1] = func(x0)
        
        Ghalf[:, 2] = grad(x0)

        x = x0
        for k in range(K):
            x = x - t * grad(x)
            F[k + 2] = func(x)
            Ghalf[:, k + 3] = grad(x)

        F[-1] = F[K + 1] - F[0]
        out.append((Ghalf.T @ Ghalf, F))
    
    avg_G = np.average([sample[0] for sample in out], axis=0)
    avg_F = np.average([sample[1] for sample in out], axis=0)
    avg_out = [(avg_G, avg_F)]
        
    return out, avg_out


def main():
    seed = 0
    mu = 1
    L = 10
    R = 1
    t = 0.1
    K = 7

    d = 10
    N = 20

    np.random.seed(seed)

    problem = PEP()
    # could do SmoothStronglyConvexQuadraticFunction if we want
    # func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L)


    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    # x = x0
    # for _ in range(K):
    #     x = x - t * func.gradient(x)
    x_stack, g_stack, f_stack = algorithm(func, func.gradient, x0, xs, K, t)

    x = x_stack[-1]
    problem.set_performance_metric(func(x) - fs)

    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')

    print('tau from pepit:', pepit_tau)

    x0 = np.zeros(d)
    x0[0] = R
    trajectories, avg_trajectories = generate_trajectories(N, d, mu, L, R, t, K, x0, traj_seed=1)
    _, oos_avg_trajectories = generate_trajectories(100000, d, mu, L, R, t, K, x0, traj_seed=10)

    DR = DROReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
    )

    eps_vals = np.logspace(-4, 2, num=25)

    res = DR.solve_eps_vals(eps_vals)
    print('obj vals full sample:', res)

    # all_G = [traj[0] for traj in trajectories]
    # all_F = [traj[1] for traj in trajectories]

    # avg_G = np.average(all_G, axis=0)
    # avg_F = np.average(all_F, axis=0)
    # avg_trajectories = [(avg_G, avg_F)]

    Avg_DR = DROReformulator(
        problem,
        avg_trajectories,
        'expectation',
        'cvxpy',
    )

    avg_res = Avg_DR.solve_eps_vals(eps_vals)
    print('obj vals with traj avg:', avg_res)


    A_obj, b_obj, _ = expression_to_matrices(problem.objective)
    sample_obj = np.trace(A_obj@avg_trajectories[0][0]) + b_obj@avg_trajectories[0][1]
    oos_sample_obj = np.trace(A_obj@oos_avg_trajectories[0][0]) + b_obj@oos_avg_trajectories[0][1]

    plt.figure()
    plt.plot(eps_vals, res, label='full samples')
    plt.plot(eps_vals, avg_res, label='avg of samples')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    plt.axhline(y=sample_obj, color='gray', linestyle='--', label='used sample')
    plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')

    plt.xscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('DRO obj value')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.title(fr'$N$={N}, $K$={K}')

    plt.tight_layout()
    plt.savefig(f'../plots/expectation/N{N}K{K}.pdf')
    # plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    CVar_DR = DROReformulator(
        problem,
        trajectories,
        'cvar',
        'cvxpy',
    )
    alpha = 0.05  # alpha should be small
    res, t_value = CVar_DR.solve_fixed_alpha_eps_vals(alpha, eps_vals)
    print('obj vals cvar, full sample:', res)

    CVar_Avg_DR = DROReformulator(
        problem,
        avg_trajectories,
        'cvar',
        'cvxpy',
    )

    avg_res, avg_t_value = CVar_Avg_DR.solve_fixed_alpha_eps_vals(alpha, eps_vals)
    print('obj vals cvar, with traj avg:', avg_res)

    plt.figure()
    plt.plot(eps_vals, res, label='full samples')
    plt.plot(eps_vals, avg_res, label='avg of samples')
    plt.plot(eps_vals, t_value, label='full samples (t)')
    plt.plot(eps_vals, avg_t_value, label='avg of samples (t)')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    plt.axhline(y=sample_obj, color='gray', linestyle='--', label='used sample')
    plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')

    plt.xscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('DRO obj value')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.title(fr'$N$={N}, $K$={K}, $\alpha$={alpha}')

    plt.tight_layout()
    plt.savefig(f'../plots/cvar/N{N}K{K}a{alpha}.pdf')



if __name__ == '__main__':
    main()
