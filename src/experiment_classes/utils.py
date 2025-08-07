import numpy as np
from scipy.stats import ortho_group


def generate_P_fixed_mu_L(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = np.zeros(d)
    sigma[0] = mu
    sigma[-1] = L
    sigma[1:d-1] = np.random.uniform(low=mu, high=L, size=(d-2, ))
    return U @ np.diag(sigma) @ U.T


def generate_P_bounded_mu_L(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = np.random.uniform(low=mu, high=L, size=(d, ))
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

def gradient_descent(f, g, x0, xs, params):
    t = params['t']
    # K = params['K']
    K_max = params['K_max']

    x_stack = []
    g_stack = []
    f_stack = []

    x = x0
    x_stack = [xs, x0]
    g_stack = [g(xs), g(x0)]
    f_stack = [f(xs), f(x0)]

    for k in range(1, K_max+1):
        # algorithm: GD
        x = x - t * g(x)
        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))

    return x_stack, g_stack, f_stack
    
def nesterov_accelerated_gradient(f, g, x0, xs, params):
    t = params['t']
    # K = params['K']
    K_max = params['K_max']

    x_stack = []
    g_stack = []
    f_stack = []

    x = x0
    y = x0
    x_stack = [xs, x0]
    g_stack = [g(xs), g(x0)]
    f_stack = [f(xs), f(x0)]

    for k in range(1, K_max+1):
        # algorithm: NAG
        y_prev = y
        y = x - t * g(x)
        x = y + (k-1)/(k+2)*(y - y_prev)

        # x_prev = x
        # x = y - t * g(y)
        # y = x + (k-1)/(k+2)*(x - x_prev)

        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))
    x_stack = np.array(x_stack)
    g_stack = np.array(g_stack)
    f_stack = np.array(f_stack)

    return x_stack, g_stack, f_stack


def generate_trajectories(f, g, x0, xs, fs, algorithm, params):
    x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)
    x_stack = np.array(x_stack).T
    g_stack = np.array(g_stack).T
    f_stack = np.array(f_stack)

    G_half = np.hstack([x_stack[:,:2], g_stack[:,1:]])
    # F = np.concatenate([f_stack, [f_stack[-1]-f_stack[0]]])
    F = f_stack

    return G_half.T@G_half, F


# def generate_trajectories(params, x0, algorithm, matrix_generation, traj_seed=1):
#     N = params['N']
#     d = params['d']
#     mu = params['mu']
#     L = params['L']
#     K_max = params['K_max']
    
#     np.random.seed(traj_seed)
    
#     samples = []
#     for n in range(N):
#         if np.mod(n, 1000) == 0 and n > 0 :
#             print(f'Generated {n} samples.')
#         P = matrix_generation(d, mu, L)

#         def f(x):
#             return .5 * x.T @ P @ x
        
#         def g(x):
#             return P @ x

#         xs = 0.0 * x0
#         x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)

#         x_stack = np.array(x_stack).T
#         g_stack = np.array(g_stack).T
#         f_stack = np.array(f_stack)
    
#         G_half = np.hstack([x_stack[:,:2], g_stack[:,1:]])
#         # F = np.concatenate([f_stack, [f_stack[-1]-f_stack[0]]])
#         F = f_stack
#         samples.append((G_half.T@G_half, F))
    
#     return samples

def sample_x0_centered_disk(n, R):
    x = np.random.normal(0, 1, n)
    x /= np.linalg.norm(x)
    dist = np.random.uniform(0, 1) ** (1 / n)
    return R * dist * x
