import cvxpy as cp
import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from scipy.stats import ortho_group
from sdp_data_matrix_scalable import smooth_convex_gd, generate_sample

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def generate_P(d, mu, L):
    U = ortho_group.rvs(d)
    sigma = np.zeros(d)
    sigma[0] = mu
    sigma[-1] = L
    sigma[1:d-1] = np.random.uniform(low=mu, high=L, size=(d-2, ))

    return U @ np.diag(sigma) @ U.T


def generate_trajectories(N, d, mu, L, R, t, K, x0):
    '''
    Ghalf = [x^star, x0, g0, g1]
    and
    F = [f^star, f0, f1, tau] where tau = f1 - f^star
    '''
    out = []
    for _ in range(N):
        P = generate_P(d, mu, L)

        def func(x):
            return .5 * x.T @ P @ x
        
        def grad(x):
            return P @ x

        Ghalf = np.zeros((d, K + 3))
        F = np.zeros(K + 3)

        Ghalf[:, 1] = x0
        F[1] = func(x0)
        
        Ghalf[:, 2] = grad(x0)

        x = x0
        for k in range(K):
            x = x - t * grad(x)
            F[k + 2] = func(x)
            Ghalf[:, k + 3] = grad(x)

        F[-1] = F[K + 1] - F[0]
        out.append((Ghalf.T @ Ghalf, F))

    return out


def generate_and_solve_dual(eps, N, d, mu, L, R, t, K, obj_consts, pep_data, c=None):
    x0 = np.zeros(d)
    x0[0] = R

    # trajectories = generate_trajectories(N, d, mu, L, R, t, K, x0)
    trajectories = [(sample_G, sample_F)]
    cp_dual(trajectories, eps, obj_consts, pep_data, c=c)


def cp_dual(trajectories, eps, obj_consts, pep_data, c=None):
    N = len(trajectories)
    M = len(pep_data.keys())
    mat_dim = obj_consts['Gobj'].shape[0]
    vec_dim = obj_consts['Fobj'].shape[0]
    Aobj = obj_consts['Gobj']
    bobj = obj_consts['Fobj']

    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    # Gz = cp.Variable((N, mat_dim, mat_dim))
    Gz = [cp.Variable((mat_dim, mat_dim), symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(vec_dim) for _ in range(N)]

    if c is None:
        c = extract_c(pep_data)

    obj = lambd * eps + 1 / N * cp.sum(s)
    constraints = [y >= 0]
    for i in range(N):
        G_sample, F_sample = trajectories[i]
        constraints += [-c.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]

        # LstarG = 0
        # LstarF = 0
        for m in range(M):
            pep_data_idx = pep_data[m]
            Am = pep_data_idx['Gcons']
            bm = pep_data_idx['Fcons']
            # print(y[i,m] * Am)
            LstarG = y[i, m] * Am
            LstarF = y[i, m] * bm
        constraints += [LstarG - Gz[i] - Aobj >> 0]
        constraints += [LstarF - Fz[i] - bobj == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve(solver=cp.CLARABEL)
    print('dro dual:', res)


def extract_c(pep_data):
    out = []
    for idx in pep_data:
        out.append(pep_data[idx]['ccons'])
    # print(out)
    return np.array(out)


def main():
    seed = 0
    mu = 1
    L = 10
    R = 1
    t = 0.1
    K = 1

    d = 5
    N = 1

    np.random.seed(seed)

    problem = PEP()
    # could do SmoothStronglyConvexQuadraticFunction if we want
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    x = x0
    for _ in range(K):
        x = x - t * func.gradient(x)

    problem.set_performance_metric(func(x) - fs)

    pepit_tau = problem.solve(solver=cp.MOSEK)

    # testing sdp
    print('----testing with cvxpy----')
    n = Point.counter
    G = cp.Variable((n, n), symmetric=True)
    F = cp.Variable(n)

    constraints = [G >> 0]

    Gobj, Fobj, cobj = expression_to_matrices(problem.objective)
    print('for obj:', Gobj, Fobj, cobj)
    # cp_obj = cp.sum(cp.multiply(G, Gobj)) + Fobj @ F + cobj
    cp_obj = cp.trace(Gobj @ G) + Fobj @ F + cobj

    obj_consts = {}
    obj_consts['Gobj'] = Gobj
    obj_consts['Fobj'] = Fobj
    obj_consts['cobj'] = cobj

    # for constr in problem.wrapper._list_of_constraints_sent_to_solver:
    pep_data = {}
    for idx, constr in enumerate(problem._list_of_constraints_sent_to_wrapper):
        pep_data_idx = {}

        Gcons, Fcons, ccons = expression_to_matrices(constr.expression)
        # print(Gcons, Fcons, ccons)
        expr = cp.trace(Gcons @ G) + Fcons @ F + ccons
        pep_data_idx['Gcons'] = Gcons
        pep_data_idx['Fcons'] = Fcons
        pep_data_idx['ccons'] = ccons
        # c_vals.append(ccons)

        if constr.equality_or_inequality == 'inequality':
            constraints += [expr <= 0]
        else:
            # TODO: replace with both double sided inequalities to fit our linear operator
            constraints += [expr == 0]
            raise NotImplementedError

        pep_data[idx] = pep_data_idx
    
    cp_prob = cp.Problem(cp.Maximize(cp_obj), constraints)
    cp_res = cp_prob.solve()
    # c_vals = np.array(c_vals)
    print(cp_res)
    print(F.value)
    print(G.value)

    eps = 10

    generate_and_solve_dual(eps, N, d, mu, L, R, t, K, obj_consts, pep_data)


def main_alt():
    d = 5
    K = 1
    mu = 0.0
    L = 1.0
    gamma = 1/L
    R = 1.0

    A, b, C = smooth_convex_gd(d, K, mu, L, gamma)
    b = np.array(b)

    sample_n = 1
    sample_xks, sample_gks, sample_fks = generate_sample(d, K, mu, L, gamma, seed=0)
    sample_P = sample_xks[0]
    for g in sample_gks[:-1]:
        sample_P = np.concatenate([sample_P, g], axis=1)

    global sample_G
    global sample_F

    sample_G = sample_P.T @ sample_P
    sample_F = np.concatenate(sample_fks, axis=1)[0]

    obj_consts = {}
    obj_consts['Gobj'] = -C.G
    obj_consts['Fobj'] = -C.F
    c = -b

    pep_data = {}
    for idx, val in enumerate(A):
        pep_data_idx = {}
        # A[i].G
        # A[i].F
        pep_data_idx['Gcons'] = A[idx].G
        pep_data_idx['Fcons'] = A[idx].F
        pep_data[idx] = pep_data_idx
    eps = 0.

    generate_and_solve_dual(eps, sample_n, d, mu, L, R, gamma, K, obj_consts, pep_data, c=c)


if __name__ == '__main__':
    # main()
    main_alt()
