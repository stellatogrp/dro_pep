##### Generates matrices for linear matrix inequality at SDP formulation #####
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import ortho_group
from math import sqrt

def outer_product(x, y):
    z = x@y.T
    z = 1/2 * (z+z.T)
    return z

def generate_sample(d, K_max, mu, L, gamma, seed=0):
    np.random.seed(seed)

    # sample quadratic function in \mathcal{Q}_{0, L}
    eigvals = np.random.randn(d)
    eigvals = mu + L * (eigvals - np.min(eigvals)) / (np.max(eigvals) - np.min(eigvals))
    eigvals.sort()
    U = ortho_group.rvs(d)
    P = U.T @ np.diag(eigvals) @ U

    # q = np.random.randn(d)
    q = np.zeros(d)        # just for now, x* = 0 as q = 0
    sx_star = np.zeros(d)

    def sample_f(x):
        return 1/2 * x.T @ P @ x + q.T @ x

    def grad_sample_f(x):
        return P @ x + q

    # apply gradient descent to obtain sample trajectories
    sxk = np.zeros(d)
    sxk[0] = 1  # start from x0=(1,0,..,0)
    sgk = grad_sample_f(sxk)
    sfk = sample_f(sxk)
    sample_xks = [sxk]
    sample_gks = [sgk]
    sample_fks = [sfk]
    for k in range(1, K_max):
        sxk = sxk - gamma * sgk
        sgk = grad_sample_f(sxk)
        sfk = sample_f(sxk)
        sample_xks.append(sxk)
        sample_gks.append(sgk)
        sample_fks.append(sfk)
    sample_xks.append(np.zeros(d))
    sample_gks.append(np.zeros(d))
    sample_fks.append(0.0)

    return sample_xks, sample_gks, sample_fks

def smooth_convex_gd_close_to_sample(d, K, mu, L, gamma, radius):

    ## define iterates
    x = {}
    g = {}
    f = {}
    E = np.eye(K+1)
    x[0] = E[:,0:1]
    g[0] = E[:,1:2]
    f[0] = E[:,0:1]
    xstar = 0*x[0]
    gstar = 0*g[0]
    fstar = 0*f[0]

    # gradient descent with stepsize gamma
    for k in range(K-1):
        x[k+1] = x[k] - gamma*g[k]
        g[k+1] = E[:,k+2:k+3]
        f[k+1] = E[:,k+1:k+2]
    x[K] = xstar
    g[K] = gstar
    f[K] = fstar

    ## generate matrices for interpolation conditions

    # interpolation with (i, j) : <A(i,j), RR.T> = 0 = b(i,j)
    A = []
    b = []
    for i in range(K+1):
        for j in range(K+1):
            # if i == j-1 or i == j+1:
            if True:
                Aij = 1/2/(1-mu/L) * (
                    1/L * outer_product(g[i]-g[j], g[i]-g[j])
                    + mu * outer_product(x[i]-x[j], x[i]-x[j])
                    - 2*mu/L * outer_product(g[i]-g[j], x[i]-x[j])
                )
                Aij += outer_product(g[j], x[i]-x[j])
                fij = - f[i]@f[i].T + f[j]@f[j].T
                A.append(block_diag(Aij, np.zeros((d,d)), fij))
                b.append(0.0)

    # interpolation with (i, star) and (star, i)
    for i in range(K):
        # j = K
        Aistar = 1/2/(1-mu/L) * (
            1/L * outer_product(g[i]-gstar, g[i]-gstar)
            + mu * outer_product(x[i]-xstar, x[i]-xstar)
            - 2*mu/L * outer_product(g[i]-gstar, x[i]-xstar)
        )
        Aistar += outer_product(gstar, x[i]-xstar)
        fistar = - f[i]@f[i].T + fstar@fstar.T
        A.append(block_diag(Aistar, np.zeros((d,d)), fistar))
        b.append(0.0)

        # i = K
        Astari = 1/2/(1-mu/L) * (
            1/L * outer_product(gstar-g[i], gstar-g[i])
            + mu * outer_product(xstar-x[i], xstar-x[i])
            - 2*mu/L * outer_product(gstar-g[i], xstar-x[i])
        )
        Astari += outer_product(g[i], xstar-x[i])
        fstari = - fstar@fstar.T + f[i]@f[i].T
        A.append(block_diag(Astari, np.zeros((d,d)), fstari))
        b.append(0.0)

    # initial condition
    A0 = outer_product(x[0], x[0])
    f0 = 0*E
    A.append(block_diag(A0, np.zeros((d,d)), f0))
    b.append(1.0)

    # Additional condition: close to sample trajectory
    gamma = 1/L
    sample_xks, sample_gks, sample_fks = generate_sample(d, K, mu, L, gamma, seed=0)
    for k in range(K):
        Ak = block_diag(outer_product(x[k], x[k]), np.zeros((d, d)), 0*E)
        sample_xks[k] = sample_xks[k].reshape((d, 1))
        Ak[K+1:K+1+d,:K+1] = -sample_xks[k]@x[k].T
        Ak[:K+1,K+1:K+1+d] = -x[k]@sample_xks[k].T
        bk = radius**2 - np.sum(sample_xks[k]**2)
        A.append(Ak)
        b.append(bk)

    # # Additional condition: close to sample trajectory - summation
    # gamma = 1/L
    # sample_xks, sample_gks, sample_fks = generate_sample(d, K, mu, L, gamma, seed=0)
    # Ak = np.zeros(A[-1].shape)
    # bk = radius**2
    # for k in range(K):
    #     Ak += block_diag(outer_product(x[k], x[k]), np.zeros((d, d)), 0*E)
    #     sample_xks[k] = sample_xks[k].reshape((d, 1))
    #     Ak[K+1:K+1+d,:K+1] += -sample_xks[k]@x[k].T
    #     Ak[:K+1,K+1:K+1+d] += -x[k]@sample_xks[k].T
    #     bk += - np.sum(sample_xks[k]**2)
    # A.append(Ak)
    # b.append(bk)
    
    

    # Aij trimming: inequality to equality
    m = len(b)
    Em = np.eye(m)
    for ij, (Aij, bij) in enumerate(zip(A, b)):
        Aij = block_diag(Aij, np.diag(Em[:, ij]))
        A[ij] = Aij

    # objective function: <C, RR.T>
    C = block_diag(0*A0, np.zeros((d,d)), - fstar@fstar.T + f[K-1]@f[K-1].T, 0*Em)
    C = -C # maximize --> minimize
    
    return A, b, C


# problem dimension d, iteration number K
def smooth_convex_gradient_descent(d, K):
    ## problem spec
    mu = 0.0
    L = 1.0

    ## define iterates
    x = {}
    g = {}
    f = {}
    E = np.eye(K+1)
    x[0] = E[:,0:1]
    g[0] = E[:,1:2]
    f[0] = E[:,0:1]
    xstar = 0*x[0]
    gstar = 0*g[0]
    fstar = 0*f[0]

    # gradient descent with stepsize gamma
    for k in range(K-1):
        x[k+1] = x[k] - (1/L)*g[k]
        g[k+1] = E[:,k+2:k+3]
        f[k+1] = E[:,k+1:k+2]

    ## generate matrices for interpolation conditions

    # interpolation with (i, j) : <A(i,j), RR.T> = 0 = b(i,j)
    A = {}
    b = {}
    for i in range(K):
        for j in range(K):
        # for j in range(i+1, K):
            Aij = 1/2/(1-mu/L) * (
                1/L * outer_product(g[i]-g[j], g[i]-g[j])
                + mu * outer_product(x[i]-x[j], x[i]-x[j])
                - 2*mu/L * outer_product(g[i]-g[j], x[i]-x[j])
            )
            Aij += outer_product(g[j], x[i]-x[j])
            fij = - f[i]@f[i].T + f[j]@f[j].T
            sij = np.zeros(((K+1)**2, (K+1)**2))
            sij[(K+1)*i+j, (K+1)*i+j] = 1
            A[(K+1)*i+j] = block_diag(Aij, fij, sij)
            b[(K+1)*i+j] = 0.0

    # interpolation with (i, star) and (star, i)
    for i in range(K):
        # j = K
        Aistar = 1/2/(1-mu/L) * (
            1/L * outer_product(g[i]-gstar, g[i]-gstar)
            + mu * outer_product(x[i]-xstar, x[i]-xstar)
            - 2*mu/L * outer_product(g[i]-gstar, x[i]-xstar)
        )
        Aistar += outer_product(gstar, x[i]-xstar)
        fistar = - f[i]@f[i].T + fstar@fstar.T
        sistar = np.zeros(((K+1)**2, (K+1)**2))
        sistar[(K+1)*i+K, (K+1)*i+K] = 1
        A[(K+1)*i+K] = block_diag(Aistar, fistar, sistar)
        b[(K+1)*i+K] = 0.0

        # i = K
        Astari = 1/2/(1-mu/L) * (
            1/L * outer_product(gstar-g[i], gstar-g[i])
            + mu * outer_product(xstar-x[i], xstar-x[i])
            - 2*mu/L * outer_product(gstar-g[i], xstar-x[i])
        )
        Astari += outer_product(g[i], xstar-x[i])
        fstari = - fstar@fstar.T + f[i]@f[i].T
        sstari = np.zeros(((K+1)**2, (K+1)**2))
        sstari[(K+1)*K+i, (K+1)*K+i] = 1
        A[(K+1)*K+i] = block_diag(Astari, fstari, sstari)
        b[(K+1)*K+i] = 0.0

    # initial condition
    A0 = outer_product(x[0], x[0])
    f0 = 0.0*E
    s0 = np.zeros(((K+1)**2, (K+1)**2))
    s0[(K+1)*K+K, (K+1)*K+K] = 1
    A[(K+1)*K+K] = block_diag(A0, f0, s0)
    b[(K+1)*K+K] = 1.0

    # objective function: <C, RR.T>
    C = block_diag(0*A0, - fstar@fstar.T + f[K-1]@f[K-1].T, s0*0)
    C = -C # maximize --> minimize
    
    return A, b, C