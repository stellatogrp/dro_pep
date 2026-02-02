"""
CvxpyLayers setup functions for DRO-PEP optimization.

This module contains CvxpyLayers-based functions for creating differentiable
optimization layers for PEP and DRO problems.
"""
import cvxpy as cp
import numpy as np
import logging

log = logging.getLogger(__name__)

# NOTE: We investigated adding ignore_dpp=True to speed up cvxpylayers compilation,
# but this is incompatible with how cvxpylayers handles parametric problems.
# With ignore_dpp=True, CVXPY tries to evaluate parameters immediately, breaking the layer.
# The DPP warning about "too many parameters" can cause OOM for very large problems.
# Alternative solutions:
#   1. Reduce N (number of samples) or M (constraint count)
#   2. Use a different backend (e.g., custom JAX solver)
#   3. Split the problem into smaller sub-problems

from cvxpylayers.jax import CvxpyLayer


class CvxpyLayerWithDefaults:
    """Wrapper for CvxpyLayer that properly merges default solver_args.
    
    Works around a bug in cvxpylayers where solver_args passed to __init__
    are not merged with solver_args passed to __call__.
    Also sets ignore_dpp=True by default to avoid slow DPP compilation with many parameters.
    """
    def __init__(self, problem, parameters, variables, solver_args=None):
        self._default_solver_args = solver_args or {}
        # Note: The DPP warning about "too many parameters" is purely about compilation speed,
        # not correctness. Gradients still work correctly without DPP optimization.
        self._layer = CvxpyLayer(problem, parameters=parameters, variables=variables)
    
    def __call__(self, *params, solver_args=None):
        merged = {**self._default_solver_args, **(solver_args or {})}
        return self._layer(*params, solver_args=merged)


def create_full_pep_layer(M, mat_shape, vec_shape):
    """Create vanilla PEP cvxpylayer (primal formulation) with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters.
    
    The PEP primal problem is:
        max  trace(A_obj @ G) + b_obj^T @ F
        s.t. G >> 0
             trace(A_m @ G) + b_m^T @ F + c_m <= 0  for all m
    
    Args:
        M: Number of interpolation constraints
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, c_vals, A_obj, b_obj]
        
        And returns: [G, F] the optimal Gram matrix and function values
    """
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]  # M constraint matrices
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]  # M constraint vectors
    A_obj_param = cp.Parameter(mat_shape)                    # Objective matrix
    b_obj_param = cp.Parameter(vec_shape)                    # Objective vector
    c_vals_param = cp.Parameter(M)                           # Objective coefficients
    
    G = cp.Variable(mat_shape)
    F = cp.Variable(vec_shape)

    constraints = [G >> 0]

    for m in range(M):
        Am = A_params[m]
        bm = b_params[m]
        cm = c_vals_param[m]
        constraints += [cp.trace(Am @ G) + bm.T @ F + cm <= 0]
    
    obj = cp.trace(A_obj_param @ G) + b_obj_param.T @ F

    prob = cp.Problem(cp.Maximize(obj), constraints)

    all_params = A_params + b_params + [c_vals_param, A_obj_param, b_obj_param]
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[G, F])


def create_full_dro_exp_layer(M, N, mat_shape, vec_shape, obj_mat_shape, obj_vec_shape, 
                               c_vals, precond_inv, eps):
    """Create expectation-based DRO cvxpylayer with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters (not constants).
    
    Args:
        M: Number of interpolation constraints
        N: Number of samples
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
        obj_mat_shape: Shape of A_obj (dimG, dimG)
        obj_vec_shape: Shape of b_obj (dimF,)
        c_vals: Constraint constants (fixed numpy array)
        precond_inv: Tuple (precond_inv_G, precond_inv_F) for preconditioning
        eps: Wasserstein ball radius
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, A_obj, b_obj, G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    """
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]  # M constraint matrices
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]  # M constraint vectors
    A_obj_param = cp.Parameter(obj_mat_shape)               # Objective matrix
    b_obj_param = cp.Parameter(obj_vec_shape)               # Objective vector
    
    # Create parameters for sample Gram matrices (also computed via JAX)
    G_params = [cp.Parameter(obj_mat_shape) for _ in range(N)]  # N sample G matrices
    F_params = [cp.Parameter(obj_vec_shape) for _ in range(N)]  # N sample F vectors
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(obj_vec_shape) for _ in range(N)]
    
    # Objective: minimize lambda * eps + (1/N) * sum(s)
    obj = lambd * eps + 1 / N * cp.sum(s)
    
    # Constraints
    constraints = [y >= 0]
    
    # Preconditioning
    G_preconditioner = np.diag(precond_inv[0])
    F_preconditioner = precond_inv[1]
    
    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        
        # Epigraph constraint
        constraints += [- c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        
        # SOC constraint for Wasserstein ball
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz[i])
        ]))]
        
        # L* constraint: sum_m (y[i,m] * A_m) - Gz[i] - A_obj >> 0
        LstarG = 0
        LstarF = 0
        for m in range(M):
            LstarG = LstarG + y[i, m] * A_params[m]
            LstarF = LstarF + y[i, m] * b_params[m]
        
        constraints += [LstarG - Gz[i] - A_obj_param >> 0]
        constraints += [LstarF - Fz[i] - b_obj_param == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Parameters ordered: A_params, b_params, A_obj, b_obj, G_params, F_params
    all_params = A_params + b_params + [A_obj_param, b_obj_param] + G_params + F_params
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s])


def create_full_dro_cvar_layer(M, N, mat_shape, vec_shape, obj_mat_shape, obj_vec_shape,
                                c_vals, precond_inv, eps, alpha):
    """Create CVaR-based DRO cvxpylayer with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters (not constants).
    
    Args:
        M: Number of interpolation constraints
        N: Number of samples
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
        obj_mat_shape: Shape of A_obj (dimG, dimG)
        obj_vec_shape: Shape of b_obj (dimF,)
        c_vals: Constraint constants (fixed numpy array)
        precond_inv: Tuple (precond_inv_G, precond_inv_F) for preconditioning
        eps: Wasserstein ball radius
        alpha: CVaR confidence level
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, A_obj, b_obj, G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    """
    alpha_inv = 1 / alpha
    
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]
    A_obj_param = cp.Parameter(obj_mat_shape)
    b_obj_param = cp.Parameter(obj_vec_shape)
    
    # Create parameters for sample Gram matrices
    G_params = [cp.Parameter(obj_mat_shape) for _ in range(N)]
    F_params = [cp.Parameter(obj_vec_shape) for _ in range(N)]
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    t_var = cp.Variable()
    y1 = cp.Variable((N, M))
    y2 = cp.Variable((N, M))
    
    Gz1 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz1 = [cp.Variable(obj_vec_shape) for _ in range(N)]
    Gz2 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz2 = [cp.Variable(obj_vec_shape) for _ in range(N)]
    
    # Objective
    obj = lambd * eps + 1 / N * cp.sum(s)
    
    # Constraints
    constraints = [y1 >= 0, y2 >= 0]
    
    # Preconditioning
    G_preconditioner = np.diag(precond_inv[0])
    F_preconditioner = precond_inv[1]
    
    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        
        # CVaR epigraph constraints
        constraints += [t_var - c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
        constraints += [-(alpha_inv - 1) * t_var - c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]
        
        # SOC constraints
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz1[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz1[i])
        ]))]
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz2[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz2[i])
        ]))]
        
        # L* constraints with parameterized A_vals, b_vals
        y1A_adj = 0
        y2A_adj = 0
        y1b_adj = 0
        y2b_adj = 0
        
        for m in range(M):
            y1A_adj = y1A_adj + y1[i, m] * A_params[m]
            y2A_adj = y2A_adj + y2[i, m] * A_params[m]
            y1b_adj = y1b_adj + y1[i, m] * b_params[m]
            y2b_adj = y2b_adj + y2[i, m] * b_params[m]
        
        constraints += [y1A_adj - Gz1[i] >> 0]
        constraints += [y1b_adj - Fz1[i] == 0]
        constraints += [y2A_adj - Gz2[i] - alpha_inv * A_obj_param >> 0]
        constraints += [y2b_adj - Fz2[i] - alpha_inv * b_obj_param == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Parameters ordered: A_params, b_params, A_obj, b_obj, G_params, F_params
    all_params = A_params + b_params + [A_obj_param, b_obj_param] + G_params + F_params
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s])
