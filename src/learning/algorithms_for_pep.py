import numpy as np

def gradient_descent(f, g, x0, xs, params) :
    t = params['stepsizes'][0]
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

def nesterov_fgm(f, g, x0, xs, params):
    '''
        Algorithm 14 from https://arxiv.org/pdf/2101.09545
        Specific equivalent form implemented is Algorithm 28 (Section B.1.3)
        
        t can be scalar or vector of length K_max
        beta is a vector of length K_max
    '''
    t, beta = params['stepsizes'][0], params['stepsizes'][1]
    K_max = params['K_max']

    x_stack = []
    g_stack = []
    f_stack = []

    x = x0
    y = x0
    x_stack = [xs, x0]
    g_stack = [g(xs), g(x0)]
    f_stack = [f(xs), f(x0)]

    # Handle scalar or vector t
    is_scalar_t = np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0)

    for k in range(K_max):
        t_k = t if is_scalar_t else t[k]
        beta_k = beta[k]

        y_prev = y
        y = x - t_k * g(x)
        x = y + beta_k * (y - y_prev)

        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))
    x_stack = np.array(x_stack)
    g_stack = np.array(g_stack)
    f_stack = np.array(f_stack)

    return x_stack, g_stack, f_stack
