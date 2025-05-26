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
        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))
    x_stack = np.array(x_stack)
    g_stack = np.array(g_stack)
    f_stack = np.array(f_stack)

    return x_stack, g_stack, f_stack