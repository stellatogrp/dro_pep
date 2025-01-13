from math import sqrt
import time
import numpy as np
import jax.numpy as jnp
import jax
import optax
import optax.tree_utils as otu
import functools
from sdp_data_matrix import smooth_convex_gd_close_to_sample, generate_sample





### Generate problem data for following performance estimation problem:
#       maximize        f(x^{K-1}) - f*
#       subject to
#           (Initial condition)     ||x^0-x*|| <= R.0
#           (Algorithm update)      x^{i+1} = x^i - gamma * f.grad(x^i),  i=0,...,K-2
#           (Sample codnition)      ||x^i - y^i|| <= 0.25*R
#   given the sample trajectory {y^i}_{i=0}^{K-1},
#   in terms of linear matrix equality for SDP.

## Measure elapsed time for Burer-Monteiro algorithm
start = time.time()

## Algorithm parameter
d = 10          # problem dimension
K = 5           # iteration number
mu = 0.1        # strong convexity parameter
L = 1.0         # smoothness parameter
gamma = 1.0/L   # GD stepsize

## Optimization variable:
# R @ R.T = [ P.T@P  P.T    O       O   ]   (P.T@P = G with each element being inner product values)
#           [   P     I     O       O   ]   (P as stack of x0 and gradients)
#           [   O     O   diag(f)   O   ]   (function values as diagonal components)
#           [   O     O     O    diag(s)]   (slack variable for inequalities)
# - Here, A.shape == (R@R.T).shape
# - len(b)  =  the number of equality constraints)
radius = 0.75*1.0
A, b, C = smooth_convex_gd_close_to_sample(d, K, mu, L, gamma, radius)
print(f'number of equality constraints: {len(b)}')

## Measure of infeasibility: \sum_i (A[i]@R@R.T - b[i])
@jax.jit
def inf_gap(R):
    length = len(b)
    gap = jnp.zeros((length, 1))
    for i in range(length):
        gap = gap.at[i].set(jnp.trace(A[i]@R@R.T) - b[i])
    return gap

## Simple Lagrangian:
# With dual variable y[i] <--> equality constraint A[i]@R@R.T == b[i]
#       L(R, y) = <C, R@R.T> + \sum_i y[i] * (A[i]@R@R.T - b[i])
@jax.jit
def Lagrangian(R, y):
    m = len(b)
    S = jnp.copy(C)
    for i in range(m):
        S = S - y[i] * A[i]
    lagrangian = jnp.trace(S@R@R.T)
    lagrangian += jnp.sum(jnp.array([y[i]*b[i] for i in range(m)]))
    return lagrangian

## Augmented Lagrangian:
# With augmented Lagrangian parameter sigma
#       L(R, y) + (sigma/2) * \sum_i (A[i]@R@R.T - b[i])**2
@jax.jit
def augmented_Lagrangian(R, y, sigma):
    gap = inf_gap(R)
    lagrangian = jnp.trace(C@R@R.T)
    lagrangian -= jnp.sum(gap*y)
    lagrangian += sigma/2 * jnp.sum(gap**2)
    return lagrangian

## Apply Limited-memory BFGS to obtain
#       argmin_R {augmented_Lagrangian(R, yk, sigmak)}
# given dual variable yk and augmented Lagrangian parameter sigmak.
# - (mask.shape == Rk.shape)
# - The mask is used to remove the gradient corresponding to the fixed entries of Rk
@jax.jit
def run_lbfgs_with_mask(Rk, yk, sigmak, mask, tol): # masks entries of Rk
    max_iter = 1000
    # tol = 1e-5

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
        grad = grad * mask # masking phase: zero-out the gradient for fixed entries
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
        grad = grad * mask # masking phase
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (Rk, optimizer.init(Rk))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params


def main():
    ############################################################
    ##### Apply Burer-Monteiro to solve QCQP PEP :

    ## algorithm parameter
    gam = 10
    eta = 1/4
    term_crit = 1e-5
    m = len(A) # the number of matrix equalities
    key = jax.random.key(0)

    ## mask initialization:
    #                d         K+1            m
    #       Rk = [  P.T         O             O       ]  K+1
    #            [ eye(d)       O             O       ]   d
    #            [   O    diag(sqrt(f))       O       ]  K+1
    #            [   O          O       diag(sqrt(s)) ]   m
    # so 
    #       mask = [  ones((K+1,d))     O         O    ]
    #              [       O            O         O    ]
    #              [       O         eye(K+1)     O    ]
    #              [       O            O       eye(m) ]
    mask = np.zeros((C.shape[0], C.shape[0]-(K+1)))
    mask[:K+1,:d] = 1.0
    mask[K+1+d:,d:] = np.eye(mask.shape[1]-d)

    ## Rk (primal variable) initialization:
    # - Set every entry non-fixed entry by 1.0
    Rk = np.copy(mask)
    Rk[K+1:K+1+d,:d] = np.eye(d)

    ## dual, slack variable initialization
    yk = jnp.ones((m, 1)) # dual variable
    vk = jnp.sum(inf_gap(Rk)**2)
    sigmak = 1.0

    print(f'[test] augmented_lagrangian = {augmented_Lagrangian(Rk, yk, sigmak)}')
    print(f'[test] infeasibility (v) = {vk}')

    ## Gradient evaluation for augmented Lagrangian w.r.t primal variable R
    grad_aug_Lag = jax.grad(augmented_Lagrangian)

    ### Update loop for Burer-Monteiro
    print('\n=================================')
    print('[[[Start applying Burer-Monteiro]]]\n')
    converged = False
    k = 0
    tol = 1e-1
    while not converged:
        k = k+1

        ## Step 1: Compute infeasibility gap
        v_vec = inf_gap(Rk)
        v = jnp.sum(v_vec**2)

        ## Step 2: Choose the slack and dual variables
        if v < eta*vk:
            yk = yk - sigmak * v_vec
            vk = v
        else:
            sigmak = gam * sigmak

        ## Step 3: Apply L-BFGS to obtain minimizer R of augmented Lagrangian
        tol = 0.5*tol
        Rk = run_lbfgs_with_mask(Rk, yk, sigmak, mask, tol)

        ## Step 4: Check termination criterion
        value = augmented_Lagrangian(Rk, yk, sigmak)
        obj_val = jnp.trace(C@Rk@Rk.T)
        term_check = jnp.abs(value - obj_val) / jnp.max(jnp.array([1, jnp.abs(Lagrangian(Rk, yk))]))

        print(f'[Iter {k}] augmented Lagrangian after L-BFGS = {value}')
        print(f'sigmak={sigmak}, objective = {obj_val}, termination_criterion = {term_check}, v(infeasibility)={v}\n')

        # Terminate if stopping criterion is satisfied or been more than 100 iterations
        if (term_check < term_crit and v < term_crit) or (k > 100):
            converged = True


    end = time.time()
    print('elapsed time:', end - start, 'sec')

    # ## Print the results
    # # np.set_printoptions(precision=1)
    # P = Rk[:K+1,:d].T
    # G = P.T@P
    # F = (Rk[np.arange(K+1+d, K+1+d+K), np.arange(d, d+K)])**2
    # print('\n==============================================')
    # print(f'[Result] Dimension={d}, Iteration number={K}\n')
    # print(f'[Objective value] f(x^{K-1}) - f* <= {-obj_val}')
    # print(f'given ||x0-x*|| = {sqrt(G[0,0])}\n')
    # print('[Trajectory information] P = [x0 g0 g1 ... g{K-1}] =')
    # print(P)
    # print('[Inner products (Gram matrix)] G = P.T@P =')
    # print(G)
    # print('\n[Objective value] f = [f0-f*, f1-f*, ..., f{K-1}-f*] =\n', F)

    # print('\n[Dual variable] Each dual variable y[i] corresponds to i-th equality constraint.')
    # print(f'y = {yk.T}')

    # ## Store the results
    # data_dict = {}
    # data_dict['dimension'] = d
    # data_dict['iter_num'] = K
    # data_dict['trajectory_info'] = P
    # data_dict['inner_product'] = G
    # data_dict['objective_values'] = F
    # data_dict['dual_var'] = yk
    # np.save(f'additional/bm_data_d={d}_K={K}_radius={radius}.npy', data_dict)

if __name__ == '__main__':
    main()
