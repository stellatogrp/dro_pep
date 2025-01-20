import cvxpy as cp
import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from scipy.stats import ortho_group

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


def generate_and_solve_dual(eps, N, d, mu, L, R, t, K):
    x0 = np.zeros(d)
    x0[0] = R

    trajectories = generate_trajectories(N, d, mu, L, R, t, K, x0)


def main():
    seed = 0
    mu = 1
    L = 10
    R = 1
    t = 0.1
    K = 1

    d = 5
    N = 10

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

    # for constr in problem.wrapper._list_of_constraints_sent_to_solver:
    for constr in problem._list_of_constraints_sent_to_wrapper:
        Gcons, Fcons, ccons = expression_to_matrices(constr.expression)
        print(Gcons, Fcons, ccons)
        expr = cp.trace(Gcons @ G) + Fcons @ F + ccons

        if constr.equality_or_inequality == 'inequality':
            constraints += [expr <= 0]
        else:
            # TODO: replace with both double sided inequalities to fit our linear operator
            constraints += [expr == 0]
    
    cp_prob = cp.Problem(cp.Maximize(cp_obj), constraints)
    cp_res = cp_prob.solve()
    print(cp_res)
    print(F.value)
    print(G.value)

    eps = 0.1

    generate_and_solve_dual(eps, N, d, mu, L, R, t, K)


if __name__ == '__main__':
    main()
