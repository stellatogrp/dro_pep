import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from dro_reformulator import DROReformulator
from tqdm import tqdm
from PEPit import PEP, Expression, PSDMatrix
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from PEPit.functions import SmoothStronglyConvexFunction,SmoothStronglyConvexQuadraticFunction
from scipy.stats import ortho_group
from algorithm import gradient_descent, nesterov_accelerated_gradient
from generate_sample import generate_P, generate_P_beta, marchenko_pastur, generate_trajectories

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def main():
    seed = 0
    mu = 0
    L = 10
    R = 1
    t = 0.1
    K = 5
    K_max = 10

    d = 100 # set large for asymptotic average-case rates
    N = 50  # number of samples

    params = {'N': N, 'd': d, 'mu': mu, 'L': L, 'R': R, 'K_max': K_max, 'K': K, 't': t}
    test_algorithm = gradient_descent
    matrix_generation = marchenko_pastur
    # matrix_generation = generate_P_beta

    np.random.seed(seed)

    problem = PEP()
    # func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L)

    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    x_stack, g_stack, f_stack = test_algorithm(func, func.gradient, x0, xs, params)

    # problem.set_performance_metric(func(x) - fs)
    problem.set_performance_metric(f_stack[-1] - fs)

    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')

    print('tau from pepit:', pepit_tau)
    # print(expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression))

    x0 = np.zeros(d)
    x0[0] = R
    
    trajectories, avg_trajectories = generate_trajectories(params, x0, test_algorithm, matrix_generation, traj_seed=1)
    
    sample_params = params.copy()
    sample_params['N'] = 10000
    oos_trajectories, oos_avg_trajectories = generate_trajectories(sample_params, x0, test_algorithm, matrix_generation, traj_seed=10)

    A_obj, b_obj, _ = expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression)
    A_obj = - A_obj
    b_obj = - b_obj[:-1]
    sample_obj = np.trace(A_obj@avg_trajectories[0][0]) + b_obj@avg_trajectories[0][1]
    oos_sample_obj = np.trace(A_obj@oos_avg_trajectories[0][0]) + b_obj@oos_avg_trajectories[0][1]
    print(f'sample mean = {sample_obj}, oos sample mean = {oos_sample_obj}')

    # Solve DRO

    DR = DROReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
    )

    eps_vals = np.logspace(-3, 2, num=26)

    res = DR.solve_eps_vals(eps_vals)
    print('obj vals full sample:', res)

    Avg_DR = DROReformulator(
        problem,
        trajectories,
        'expectation_mro',
        'cvxpy',
    )

    avg_res = Avg_DR.solve_eps_vals(eps_vals)
    print('obj vals with traj avg:', avg_res)

    plt.figure()
    plt.plot(eps_vals, res, label='full samples')
    plt.plot(eps_vals, avg_res, label='avg of samples')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    plt.axhline(y=sample_obj, color='gray', linestyle='--', label='used sample')
    plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')
    # plt.plot(eps_vals, np.sqrt(np.array(eps_vals)), linestyle='--', color='gold')

    plt.xscale('log')
    plt.yscale('log')
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
    alpha = 0.1  # alpha should be small
    res, t_value = CVar_DR.solve_fixed_alpha_eps_vals(alpha, eps_vals)
    print('obj vals cvar, full sample:', res)

    CVar_Avg_DR = DROReformulator(
        problem,
        trajectories,
        'cvar_mro',
        'cvxpy',
    )

    avg_res, avg_t_value = CVar_Avg_DR.solve_fixed_alpha_eps_vals(alpha, eps_vals)
    print('obj vals cvar, with traj avg:', avg_res)

    plt.figure()
    plt.plot(eps_vals, res, label='full samples', linewidth=2)
    plt.plot(eps_vals, t_value, label='full samples (t)', linewidth=2)
    # plt.plot(eps_vals, avg_res, label='avg of samples')
    # plt.plot(eps_vals, avg_t_value, label='avg of samples (t)')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    sample_quantile = np.quantile([np.trace(A_obj@G)+b_obj.T@F for (G, F) in trajectories], 1-alpha)
    plt.axhline(y=sample_quantile, color='gray', linestyle='--', label='used sample')
    # plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')
    oos_sample_quantile = np.quantile([np.trace(A_obj@G)+b_obj.T@F for (G, F) in oos_trajectories], 1-alpha)
    plt.axhline(y=oos_sample_quantile, color='gold', linestyle='-.', label='oos sample')

    plt.xscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('DRO obj value')
    # plt.ylim([-1e-3, pepit_tau+1e-3])

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.title(fr'$N$={N}, $K$={K}, $\alpha$={alpha}')

    plt.tight_layout()
    plt.savefig(f'../plots/cvar/N{N}K{K}a{alpha}.pdf')



    eps = 0.1
    alpha_vals = np.linspace(0.0, 1.0, num=51)
    alpha_vals[0] = 1e-4
    alpha_vals[-1] = 1.0 - 1e-4
    res_alpha, t_value_alpha = CVar_DR.solve_fixed_eps_alpha_vals(alpha_vals, eps)
    avg_res_alpha, avg_t_value_alpha = CVar_Avg_DR.solve_fixed_eps_alpha_vals(alpha_vals, eps)


    plt.figure()
    plt.plot(alpha_vals, res_alpha, label='full samples', linewidth=2)
    plt.plot(alpha_vals, t_value_alpha, label='quantile estimate', linewidth=2)
    # plt.plot(alpha_vals, avg_res_alpha, label='avg of samples')
    # plt.plot(alpha_vals, avg_t_value_alpha, label='avg of samples (t)')
    plt.axhline(y=pepit_tau, color='black', linestyle='--', label='PEP bound')
    sample_quantiles = np.quantile([np.trace(A_obj@G)+b_obj.T@F for (G, F) in trajectories], 1-np.array(alpha_vals))
    plt.plot(alpha_vals, sample_quantiles, color='gray', linestyle='--', label='used sample quantile')
    # plt.axhline(y=sample_obj, color='gray', linestyle='--', label='used sample')
    # plt.axhline(y=oos_sample_obj, color='gold', linestyle='-.', label='oos sample')
    oos_sample_quantiles = np.quantile([np.trace(A_obj@G)+b_obj.T@F for (G, F) in oos_trajectories], 1-np.array(alpha_vals))
    plt.plot(alpha_vals, oos_sample_quantiles, color='gold', linestyle='--', label='oos quantile')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('DRO obj value')
    # plt.ylim([-1e-3, pepit_tau+1e-3])

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.title(fr'$N$={N}, $K$={K}, $\epsilon$={eps}')

    plt.tight_layout()
    plt.savefig(f'../plots/cvar/N{N}K{K}eps{eps}.pdf')


if __name__ == '__main__':
    main()
