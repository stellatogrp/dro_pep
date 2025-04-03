from math import sqrt
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
import functools
from collections import namedtuple
from sdp_data_matrix_scalable import smooth_convex_gd_close_to_sample, generate_sample, smooth_convex_agm_close_to_sample, generate_agm_sample, smooth_convex_hbm_close_to_sample, generate_hbm_sample





### Generate problem data for following performance estimation problem:
#       maximize        f(x^{K-1}) - f*
#       subject to
#           (Initial condition)             ||x^0-x*||^2 <= R^2
#           (Algorithm update)              x^{i+1} = x^i - gamma * f.grad(x^i),  i=0:K-2
#           (Sample proximity condition)    ||x^i - y^i||^2 <= radius^2
#   given the sample trajectory {y^i}_{i=0}^{K-1} in d-dimensional Euclidean space.
## Algorithm parameter
d = 20          # problem dimension
K = 10          # iteration number
mu = 0.1        # strong convexity parameter
L = 1.0         # smoothness parameter
gamma = 1.0/L   # GD stepsize
R = 1.0         # initial condition parameter
radius = 0.01*R # sample proximity parameter

## Measure elapsed time for Burer-Monteiro algorithm
start = time.time()

## Optimization variable:
# R @ R.T = [ P.T@P  P.T    O       O   ]   (P.T@P = G with each element being inner product values)
#           [   P     I     O       O   ]   (P as stack of x0 and gradients)
#           [   O     O   diag(F)   O   ]   (function values as diagonal components)
#           [   O     O     O    diag(S)]   (slack variable for inequalities)
# - len(b)  =  the number of equality constraints)
# - For numerical scalability, we only keep P, diag(f) = sqrt(diag(F)), and diag(s) = sqrt(diag(S)).
# - We can express the quadratic inequalities and equalities in terms of LMI with above R@R.T as :
#           np.trace(A[i] @ (R@R.T)) <= b[i]
#   Ignoring the zero blocks of A[i], we only keep blocks related to diag(F), diag(S), and
#           [ P.T@P  P.T ]
#           [   P     I  ]

# data type storing the linear map A in inequality
const = namedtuple('PyTree', ['G', 'F', 'S'])
# A, b, C = smooth_convex_gd_close_to_sample(d, K, mu, L, gamma, radius)
# A, b, C = smooth_convex_agm_close_to_sample(d, K, 0.0, L, 1.0/L, radius)
A, b, C = smooth_convex_hbm_close_to_sample(d, K, 0.0, L, 1.0/L, radius)
print(f'number of equality constraints: {len(b)}')

# Form the following matrix in terms of P :
#       [ P.T@P  P.T ]
#       [   P     I  ]
@jax.jit
def PTP_full_form(R):
    return jnp.vstack([jnp.hstack([R.P.T@R.P, R.P.T]), jnp.hstack([R.P, np.eye(d)])])

## Calculate the constraint gap $\mathcal{A}(RR^\intercal) - b$ :
@jax.jit
def inf_gap(R):
    PTP_full = PTP_full_form(R)
    length = len(b)
    gap = jnp.zeros((length, 1))
    for i in range(length):
        gap = gap.at[i].set(jnp.trace(A[i].G@PTP_full) + jnp.sum(A[i].F*(R.f**2)) + jnp.sum(A[i].S*(R.s**2)) - b[i])
    return gap

## Simple Lagrangian:
# With dual variable y[i] <--> equality constraint A[i]@R@R.T == b[i]
#       L(R, y) = <C, R@R.T> + \sum_i y[i] * (A[i]@R@R.T - b[i])
@jax.jit
def Lagrangian(R, y):
    PTP_full = PTP_full_form(R)
    gap_vec = inf_gap(R)
    lagrangian = jnp.trace(C.G@PTP_full) + jnp.sum(C.F*(R.f**2)) + jnp.sum(C.S*(R.s**2))
    lagrangian += jnp.sum(gap_vec*y)
    return lagrangian

## Augmented Lagrangian:
# With augmented Lagrangian parameter sigma
#       L(R, y) + (sigma/2) * \sum_i (A[i]@R@R.T - b[i])**2
@jax.jit
def augmented_Lagrangian(R, y, sigma):
    gap_vec = inf_gap(R)
    PTP_full = PTP_full_form(R)
    lagrangian = jnp.trace(C.G@PTP_full) + jnp.sum(C.F*(R.f**2)) + jnp.sum(C.S*(R.s**2))
    lagrangian += jnp.sum(gap_vec*y)
    lagrangian += sigma/2 * jnp.sum(gap_vec**2)
    return lagrangian

## Apply Limited-memory BFGS to obtain
#       argmin_R {augmented_Lagrangian(R, yk, sigmak)}
# given dual variable yk and augmented Lagrangian parameter sigmak.
# - (mask.shape == Rk.shape)
# - The mask is used to remove the gradient corresponding to the fixed entries of Rk
@jax.jit
def run_lbfgs_with_mask(Rk, yk, sigmak, tol, max_iter=10000): # masks entries of Rk

    # linesearch module with Zoom [Wright & Nocedal]
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=20, verbose=False)

    # L-BFGS optimizer
    optimizer = optax.lbfgs(memory_size=7, linesearch=linesearch)

    loss = functools.partial(augmented_Lagrangian, y=yk, sigma=sigmak)
    value_and_grad = optax.value_and_grad_from_state(loss)

    # Algorithm parameter update
    def step(carry):
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = optimizer.update(
            grad, state, params, value=value, grad=grad, value_fn=loss
        )
        params = optax.apply_updates(params, updates)
        return params, state

    # Termination criterion
    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        # err = jnp.sqrt(sigmak) * otu.tree_l2_norm(grad) # [Rockafellar, 2023]
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (Rk, optimizer.init(Rk))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params


############################################################
##### Apply Burer-Monteiro to solve QCQP PEP :

### [1] Initialization

## algorithm parameter
gam = 10.0          # default: 10
eta = 1/2           # default: 1/4
max_iter = 100      # maximum iteration number for outer iteration
lbfgs_max_iter = 10000  # maximum iteration number for inner loop (L-BFGS)
term_crit = 1e-4    # convergence threshold

m = len(A) # the number of matrix equalities
key = jax.random.key(0)

## Rk (primal variable) initialization:
#                d         K+1            m
#       Rk = [  P.T         O             O       ]  K+1
#            [ eye(d)       O             O       ]   d
#            [   O    diag(sqrt(f))       O       ]  K+1
#            [   O          O       diag(sqrt(s)) ]   m
# - Set every entry non-fixed entry by 1.0
PyTree = namedtuple('PyTree', ['P', 'f', 's'])

# Initialization #1: use sample trajectory
sample_xks, sample_gks, sample_fks = generate_sample(d, K, mu, L, gamma, seed=1)
P_init = jnp.vstack([sample_xks[0], jnp.vstack(sample_gks[:-1])]).T
f_init = jnp.sqrt(jnp.array(sample_fks))
Rk = PyTree(P = P_init, f = f_init, s = jnp.ones((m)))
# s_init = jnp.sqrt(jnp.abs(inf_gap(Rk))).reshape((m))
# Rk = PyTree(P = P_init, f = f_init, s = s_init)

# # Initialization #2: initialize by 1's
# Rk = PyTree(P = jnp.ones((d, K+1)), f = jnp.ones((K+1)), s = jnp.ones((m)))

## dual, slack variable initialization
yk = jnp.ones((m, 1))   # dual variable
vk = jnp.sum(inf_gap(Rk)**2)    # infeasibility measure: $\|\mathcal{A}(RR^\intercal) - b\|_2^2$

# algorithm parameter initialization
sigmak = 10.0            # augmented Lagrangian coefficient
tolk = 1/sigmak
etak = 1/(sigmak**0.1)

print(f'[test] augmented_lagrangian = {augmented_Lagrangian(Rk, yk, sigmak)}')
print(f'[test] gap v = A(RR.T)-b and |v|^2 = {vk}')

## Gradient evaluation for augmented Lagrangian w.r.t primal variable R
grad_aug_Lag = jax.grad(augmented_Lagrangian)

### [2] Apply augmented Lagrangian method

### Update loop for Burer-Monteiro
print('\n=================================')
print('[[[Start applying Burer-Monteiro]]]\n')
converged = False
k = 0
while not converged:
    k = k+1

    ## Step 1: find Rk = argmin_R L(RR^T) + <y, A(RR^T) - b> + sigmak/2 * sum(A(RR^T)-b)**2
    Rk = run_lbfgs_with_mask(Rk, yk, sigmak, tolk, lbfgs_max_iter)

    ## Step 2: Check feasibility
    v_vec = inf_gap(Rk)
    v = jnp.sqrt(jnp.sum(v_vec**2))

    ## Step 3: Parameter update as in LANCELOT at p.521 of [Nocedal et al, 2009]
    if v < etak:
        grad_vec = grad_aug_Lag(Rk, yk, sigmak)
        grad_val = jnp.sqrt(jnp.sum(grad_vec.P**2) + jnp.sum(grad_vec.f**2) + jnp.sum(grad_vec.s**2))
        if v < term_crit or grad_val < term_crit:
            converged = True
        yk = yk + sigmak * v_vec
        sigmak = sigmak
        etak = etak / (sigmak**0.9)
        tolk = tolk / sigmak
    else:
        yk = yk
        sigmak = 10.0 * sigmak
        etak = 1.0 / (sigmak**0.1)
        tolk = 1.0 / sigmak

    # ## Step 3': Parameter update (Type 2)
    # grad_vec = grad_aug_Lag(Rk, yk, sigmak)
    # grad_val = jnp.sqrt(jnp.sum(grad_vec.P**2) + jnp.sum(grad_vec.f**2) + jnp.sum(grad_vec.s**2))
    # if grad_val < tolk:
    #     tolk = 0.8*tolk
    # if v < eta*vk:
    #     yk = yk + sigmak * v_vec
    #     vk = eta*vk
    # else:
    #     sigmak = gam * sigmak

    ## Step 4: Check termination criterion
    value = augmented_Lagrangian(Rk, yk, sigmak)
    PTP_full = jnp.vstack([jnp.hstack([Rk.P.T@Rk.P, Rk.P.T]), jnp.hstack([Rk.P, np.eye(d)])])
    obj_val = jnp.trace(C.G@PTP_full) + jnp.sum(C.F*(Rk.f**2)) + jnp.sum(C.S*(Rk.s**2))
    
    grad_vec = grad_aug_Lag(Rk, yk, sigmak)
    grad_val = jnp.sqrt(jnp.sum(grad_vec.P**2) + jnp.sum(grad_vec.f**2) + jnp.sum(grad_vec.s**2))

    v = jnp.sqrt(jnp.sum(inf_gap(Rk)**2))
    print(f'[{k}-th iteration] augmented Lagrangian after L-BFGS = {value}')
    print(f'sigmak={sigmak}, objective = {obj_val}')
    print(f'v(infeasibility) = {v}, grad = {grad_val} < {tolk} = tol\n')

    # Terminate if stopping criterion is satisfied or been more than max_iter iterations
    if (grad_val < term_crit) or (v < term_crit) or (k > max_iter) or jnp.isnan(value) or jnp.isinf(grad_val) :
        converged = True
        if k > max_iter :
            print('[Iter max out] Reached the maximum iteration number for augmented Lagrangian')
        elif jnp.isnan(value) :
            print('[NaN encountered]')
        elif jnp.isinf(grad_val) :
            print('[inf encountered]')


end = time.time()
print('elapsed time:', end - start, 'sec')

## Print the results
P = Rk.P
G = P.T@P
F = Rk.f**2
F = F[:-1]
S = Rk.s**2
print('\n==============================================')
print(f'[Result] Dimension={d}, Iteration number={K}\n')
print(f'[Objective value] f(x^{K-1}) - f* <= {-obj_val} given ||x0-x*|| = {sqrt(G[0,0])}\n')
print('[Trajectory information] P = [x0 g0 g1 ... g{K-1}] =')
print(P)
print('[Inner products (Gram matrix)] G = P.T@P =')
print(G)
print('\n[Objective value] f = [f0-f*, f1-f*, ..., f{K-1}-f*] =\n', F)

print('\n[Dual variable] Each dual variable y[i] corresponds to i-th equality constraint.')
print(f'y = {yk.T}')

print(f'\n[Slack variable] s = {S}\n')

### [3] Store the results
data_dict = {}
data_dict['dimension'] = d
data_dict['iter_num'] = K
data_dict['trajectory_info'] = P
data_dict['inner_product'] = G
data_dict['objective_values'] = F
data_dict['dual_var'] = yk
data_dict['slack_var'] = S
data_dict['solve_time'] = end - start
np.save(f'result/bm_data_d={d}_K={K}_radius={radius}.npy', data_dict)

# Check dual constraint for SDP: C - sum(y[i]*A[i]) >= 0 (PSD)
GG = np.copy(C.G)
FF = np.copy(C.F)
SS = np.copy(C.S)
for i in range(len(b)):
    GG = GG + yk[i]*A[i].G
    FF = FF + yk[i]*A[i].F
    SS = SS + yk[i]*A[i].S
_, V, _ = np.linalg.svd(GG)
print('\n[Check dual constraints] must be all nonnegative values.\n')
print(f'[0] Dual objective b^T y = {np.sum(yk.T@np.array(b))}')
print('[1] Dual variable corresponding to RR^\\intercal (must be semidefinite):', V)
print('[2] Dual variable corresponding to function value vector (must be nonnegative):', FF)
print('[3] Dual variable corresponding to the slack variable (must be nonnegative):', SS)
print()
