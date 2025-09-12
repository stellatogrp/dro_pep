import numpy as np

def marchenko_pastur(d, mu, L):
    assert mu == 0
    sigma = np.sqrt(L)/2
    X = np.random.normal(0, sigma, (d, d))
    H = X.T@X/d
    while not all(np.linalg.eigvals(H) <= L) :
        X = np.random.normal(0, sigma, (d, d))
        H = X.T@X/d
    return H

def gradient_descent(x, g, mu=0.0, L=1.0):
    return x - (1/L)*g

def sample_generation(iter_K=1, N=1, mu=0.0, L=np.inf, algorithm=gradient_descent) :
    d = N
    np.random.seed(42)

    x0 = np.zeros(d)
    x0[0] = 1.0  # Initial point

    hatG, hatF = [], []
    for _ in range(N) :
        Pi = marchenko_pastur(d, mu, L)
        def f(x):
            return .5 * x.T @ Pi @ x, Pi @ x
        
        PT = np.zeros((iter_K+2, d))    # P = [x0, g0, ..., gK]; PT = P.T
        F = np.zeros(iter_K+1)    # F = [f0, ..., fK]
        PT[0,:] = x0

        xk = x0.copy()
        for k in range(iter_K+1) :
            fk, gk = f(xk)
            PT[k+1,:] = gk
            F[k] = fk
            # xk = xk - (1/L) * gk  # Gradient descent step
            xk = algorithm(xk, gk, mu=mu, L=L)
        G = PT@PT.T

        hatG.append(G)
        hatF.append(F)
    
    avgG = np.average(hatG, axis=0)
    avgF = np.average(hatF, axis=0)

    return hatG, hatF, avgG, avgF

def sample_generation_agd(iter_K=1, N=1, mu=0.0, L=np.inf) :
    d = N
    np.random.seed(42)

    x0 = np.zeros(d)
    x0[0] = 1.0  # Initial point

    hatG, hatF = [], []
    for _ in range(N) :
        Pi = marchenko_pastur(d, mu, L)
        def f(x):
            return .5 * x.T @ Pi @ x, Pi @ x
        
        PT = np.zeros((iter_K+2, d))    # P = [x0, g0, ..., gK]; PT = P.T
        F = np.zeros(iter_K+1)    # F = [f0, ..., fK]
        PT[0,:] = x0

        xk = x0.copy()
        yk = x0.copy()
        for k in range(iter_K+1) :
            fk, gk = f(xk)
            PT[k+1,:] = gk
            F[k] = fk
            yk1 = xk - (1/L) * gk  # Gradient descent step (x{k+1}, y{k+1})
            xk = yk1 + ( k/(k+3) ) * (yk1 - yk)
            yk = yk1
        G = PT@PT.T

        hatG.append(G)
        hatF.append(F)
    
    avgG = np.average(hatG, axis=0)
    avgF = np.average(hatF, axis=0)

    return hatG, hatF, avgG, avgF


def sample_generation_agd(iter_K=1, N=1, mu=0.0, L=np.inf) :
    d = N
    np.random.seed(42)

    x0 = np.zeros(d)
    x0[0] = 1.0  # Initial point

    hatG, hatF = [], []
    for _ in range(N) :
        Pi = marchenko_pastur(d, mu, L)
        def f(x):
            return .5 * x.T @ Pi @ x, Pi @ x
        
        PT = np.zeros((iter_K+2, d))    # P = [x0, g0, ..., gK]; PT = P.T
        F = np.zeros(iter_K+1)    # F = [f0, ..., fK]
        PT[0,:] = x0

        xk = x0.copy()
        yk = x0.copy()
        for k in range(iter_K+1) :
            fk, gk = f(xk)
            PT[k+1,:] = gk
            F[k] = fk
            yk1 = xk - (1/L) * gk  # Gradient descent step (x{k+1}, y{k+1})
            xk = yk1 + ( k/(k+3) ) * (yk1 - yk)
            yk = yk1
        G = PT@PT.T

        hatG.append(G)
        hatF.append(F)
    
    avgG = np.average(hatG, axis=0)
    avgF = np.average(hatF, axis=0)

    return hatG, hatF, avgG, avgF