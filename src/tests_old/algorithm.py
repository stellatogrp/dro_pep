import numpy as np
import cvxpy as cp

def gradient_descent(f, g, x0, xs, params) :
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

    is_scalar = np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0)

    for k in range(1, K_max+1):
        # algorithm: GD
        t_k = t if is_scalar else t[k-1]
        x = x - t_k * g(x)
        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))

    return x_stack, g_stack, f_stack
    
def nesterov_accelerated_gradient(f, g, x0, xs, params) :
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