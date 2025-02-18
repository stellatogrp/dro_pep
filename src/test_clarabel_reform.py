import cvxpy as cp
import numpy as np
import scipy.sparse as spa

import clarabel


def clarabel_example():
    n = 3
    nvec = int(n*(n+1)/2) 

    # Define problem data
    P = spa.csc_matrix((nvec, nvec))
    P = P.tocsc()

    q = np.array([1., 0., 1., 0., 0., 1.])
    sqrt2 = np.sqrt(2.)

    A = spa.csc_matrix(
        [[-1., 0., 0., 0., 0., 0.],
        [0., -sqrt2, 0., 0., 0., 0.],
        [0., 0., -1., 0., 0., 0.],
        [0., 0., 0., -sqrt2, 0., 0.],
        [0., 0., 0., 0., -sqrt2, 0.],
        [0., 0., 0., 0., 0., -1.],
        [1., 4., 3., 8., 10., 6.]])   

    b = np.append(np.zeros(nvec), 1.)

    cones = [clarabel.PSDTriangleConeT(n),
            clarabel.ZeroConeT(1)] 

    settings = clarabel.DefaultSettings()

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

    solution = solver.solve()
    print(solution.x)
    print(solution.z)
    print(solution.s)

    print('optimal obj:', q @ np.array(solution.x))


def cvxpy_mixed_example():
    print('----testing socp mixed with sdp----')
    n = 3
    A = np.array([
        [1, 2, 4],
        [2, 3, 5],
        [4, 5, 6],
    ])

    X = cp.Variable((n, n))
    lambd = cp.Variable()
    obj = cp.trace(X) + lambd
    # obj = cp.trace(X)
    constraints = [
        cp.trace(A @ X) == 1,
        X >> 0,
        cp.SOC(lambd, cp.vec(X)),
    ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve(solver=cp.CLARABEL)
    print('cvxpy obj:', res)

    # probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
    # A_cp = probdata['A']
    # print(A_cp.shape)
    nvec = 6
    P = spa.csc_matrix((nvec+1, nvec+1))
    P = P.tocsc()

    q = np.array([1., 0., 1., 0., 0., 1., 1.])
    sqrt2 = np.sqrt(2.)

    A = spa.csc_matrix([
        [-1., 0., 0., 0., 0., 0., 0.],
        [0., -sqrt2, 0., 0., 0., 0., 0.],
        [0., 0., -1., 0., 0., 0., 0.],
        [0., 0., 0., -sqrt2, 0., 0., 0.],
        [0., 0., 0., 0., -sqrt2, 0., 0.],
        [0., 0., 0., 0., 0., -1., 0.],
        [1., 4., 3., 8., 10., 6., 0.],
        [0., 0., 0., 0., 0., 0., -1.],
        [-1., 0., 0., 0., 0., 0., 0.],
        [0., -sqrt2, 0., 0., 0., 0., 0.],
        [0., 0., -1., 0., 0., 0., 0.],
        [0., 0., 0., -sqrt2, 0., 0., 0.],
        [0., 0., 0., 0., -sqrt2, 0., 0.],
        [0., 0., 0., 0., 0., -1., 0.],
    ])

    b = np.append(np.zeros(nvec), 1.)
    b = np.append(b, np.zeros(nvec+1))

    cones = [clarabel.PSDTriangleConeT(n),
            clarabel.ZeroConeT(1),
            clarabel.SecondOrderConeT(7)] 

    settings = clarabel.DefaultSettings()

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

    solution = solver.solve()
    print(solution.x)
    print(solution.z)
    print(solution.s)

    print('optimal obj:', q @ np.array(solution.x))
    print(A.shape)


def sdp_lmi():
    print('----testing socp mixed with sdp with lmi----')
    n = 3
    A = np.array([
        [1, 2, 4],
        [2, 3, 5],
        [4, 5, 6],
    ])

    B = 0.01 * A

    X = cp.Variable((n, n))
    lambd = cp.Variable()
    obj = cp.trace(X) + lambd
    # obj = cp.trace(X)
    constraints = [
        cp.trace(A @ X) == 1,
        X - B >> 0,
        cp.SOC(lambd, cp.vec(X)),
    ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve(solver=cp.CLARABEL)
    print('cvxpy obj:', res)


if __name__ == '__main__':
    clarabel_example()
    cvxpy_mixed_example()
    sdp_lmi()
