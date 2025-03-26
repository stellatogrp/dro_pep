import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from src.dro_reformulator import DROReformulator
from tqdm import tqdm
from PEPit import PEP
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from PEPit.functions import SmoothStronglyConvexFunction, SmoothStronglyConvexQuadraticFunction
from scipy.stats import ortho_group

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation

from src.algorithm import gradient_descent, nesterov_accelerated_gradient
from src.generate_sample import generate_P, generate_P_beta, generate_trajectories, marchenko_pastur

# Function which gives the DRO-PEP result for given algorithm over given function class, with sample distribution
def pep_problem(params, algorithm, function_class) :
    K = params['K']
    
    problem = PEP()
    f = problem.declare_function(function_class, mu=params['mu'], L=params['L'])
    g = f.gradient

    xs = f.stationary_point()
    x0 = problem.set_initial_point()
    
    problem.set_initial_condition((x0 - xs)**2 <= params['R']**2 )
    x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)

    problem.set_performance_metric(f_stack[K+1] - f(xs))

    return problem


def calculate_objective(trajectories, problem) :
    A_obj, b_obj, c_obj = expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression)
    A_obj = - A_obj
    b_obj = - b_obj[:-1]

    return np.array([np.trace(A_obj@G) + b_obj.T@F + c_obj for (G, F) in trajectories])


def dro_pep_result(params, algorithm, function_class, matrix_generation, traj_seed=1) :
    
    # DRO-PEP problem as in PEPit
    problem = pep_problem(params, algorithm, function_class)
    pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK', verbose=True)
    print('worst-case rate from pepit:', pepit_tau)

    # sample generation
    x0 = np.zeros((params['d'],))
    x0[0] = params['R']
    trajectories, avg_trajectories = generate_trajectories(params, x0, algorithm, matrix_generation, traj_seed=traj_seed)
    
    # 1. Expectation
    EXP = DROReformulator(
        problem,
        trajectories,
        'expectation',
        'cvxpy',
    )
    
    Avg_EXP = DROReformulator(
        problem,
        trajectories,
        'expectation_mro',
        'cvxpy',
    )
    
    # 2. CVaR
    CVAR = DROReformulator(
        problem,
        trajectories,
        'cvar',
        'cvxpy',
    )
    
    Avg_CVAR = DROReformulator(
        problem,
        trajectories,
        'cvar_mro',
        'cvxpy',
    )

    # Run the DRO-PEP and obtain solutions
    eps_bin, alp_bin = 11, 11
    # epsi_list = np.logspace(-3.5, 1.5, eps_bin) * (1 + np.sqrt(np.sum(avg_trajectories[0][0])**2 + np.sum(avg_trajectories[0][1]**2)))
    epsi_list = np.logspace(-3.5, 1.5, eps_bin)
    alpi_list = np.linspace(0.0, 1.0, alp_bin)
    alpi_list[0] = 1e-4

    sample_obj_list = calculate_objective(trajectories, problem)
    sample_mean = np.average(sample_obj_list)
    sample_quantile = np.quantile(sample_obj_list, alpi_list)

    EXP_result = EXP.solve_eps_vals(epsi_list)
    avg_EXP_result = Avg_EXP.solve_eps_vals(epsi_list)

    CVAR_result = []
    CVAR_t_result = []
    avg_CVAR_result = []
    avg_CVAR_t_result = []

    for eps in epsi_list :
        res, t = CVAR.solve_fixed_eps_alpha_vals(alpi_list, eps)
        CVAR_result.append(res)
        CVAR_t_result.append(t)
        avg_res, avg_t = Avg_CVAR.solve_fixed_eps_alpha_vals(alpi_list, eps)
        avg_CVAR_result.append(avg_res)
        avg_CVAR_t_result.append(avg_t)

    result_dict = {'epsi_list': epsi_list, 'alpi_list': alpi_list, 'PEP': pepit_tau,  \
                   'EXP': np.array(EXP_result), 'avg_EXP': np.array(avg_EXP_result), \
                   'CVAR': np.array(CVAR_result), 'avg_CVAR': np.array(avg_CVAR_result), \
                   'CVAR_t': np.array(CVAR_t_result), 'avg_CVAR_t': np.array(avg_CVAR_t_result), \
                   'sample_quantile': sample_quantile, 'sample_mean': sample_mean, 'params': params, \
                   'algorithm': algorithm.__name__, 'matrix': matrix_generation.__name__, 'function': function_class.__name__
                  }

    return result_dict


def main() :
    params = {'N': 10, 'd': 100, 'mu': 0.0, 'L': 1.0, 'R': 1.0, 'K_max': 20, 'K': 0, 't': 1/1.0}
    K_max = 20

    function_class = SmoothStronglyConvexQuadraticFunction
    matrix_generation = marchenko_pastur

    for K in range(10) :
        for algorithm in [gradient_descent, nesterov_accelerated_gradient] :
            params['K'] = K
            params['K_max'] = K_max
            result_dict = dro_pep_result(params, algorithm, function_class, matrix_generation)
            dict_name = f'result/{function_class.__name__}/{algorithm.__name__}_K{params['K']}_over_{params['K_max']}.npy'
            np.save(dict_name, result_dict, allow_pickle=True)


if __name__ == "__main__" :
    main()
