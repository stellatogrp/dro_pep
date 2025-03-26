import numpy as np
from scipy.stats import ortho_group

def generate_P(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = np.zeros(d)
    sigma[0] = mu
    sigma[-1] = L
    sigma[1:d-1] = np.random.uniform(low=mu, high=L, size=(d-2, ))
    return U @ np.diag(sigma) @ U.T

def generate_P_beta(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = (L-mu)*np.random.beta(1/2, 3/2, size=(d,)) + mu
    return U @ np.diag(sigma) @ U.T

def marchenko_pastur(d, mu, L):
    assert mu == 0
    sigma = np.sqrt(L)/2
    X = np.random.normal(0, sigma, (d, d))
    H = X.T@X/d
    # while not all(np.linalg.eigvals(H) <= L) :
    #     X = np.random.normal(0, sigma, (d, d))
    #     H = X.T@X/d
    return H

def generate_trajectories(params, x0, algorithm, matrix_generation, traj_seed=1):
    N = params['N']
    d = params['d']
    mu = params['mu']
    L = params['L']
    K_max = params['K_max']
    
    np.random.seed(traj_seed)
    
    samples = []
    for _ in range(N):
        P = matrix_generation(d, mu, L)

        def f(x):
            return .5 * x.T @ P @ x
        
        def g(x):
            return P @ x

        xs = 0.0 * x0
        x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)

        x_stack = np.array(x_stack).T
        g_stack = np.array(g_stack).T
        f_stack = np.array(f_stack)
    
        G_half = np.hstack([x_stack[:,:2], g_stack[:,1:]])
        # F = np.concatenate([f_stack, [f_stack[-1]-f_stack[0]]])
        F = f_stack
        samples.append((G_half.T@G_half, F))
        
    avg_G = np.average([out[0] for out in samples], axis=0)
    avg_F = np.average([out[1] for out in samples], axis=0)
    
    return samples, [(avg_G, avg_F)]