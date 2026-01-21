import clarabel
import cvxpy as cp
import diffcp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sparse
np.set_printoptions(precision=5, suppress=True)


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def main():
    A = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ])
    B = np.array([
        [7, 8, 9],
        [8, 10, 11],
        [9, 11, 12],
    ])

    X = cp.Variable((3, 3), symmetric=True)
    y = cp.Variable(2)

    constraints = [y[0] * A + y[1] * B >> 0, X >> 0]
    constraints += [
        cp.trace(A @ X) == 1,
        y >= 0,
    ]

    obj = cp.Minimize(cp.trace(X) + np.ones(2) @ y)
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.CLARABEL)
    print('from cvxpy:', res)
    # print(X.value, y.value)

    scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

    x, y, s, D, DT = diffcp.solve_and_derivative(sparse.csc_matrix(scs_A), scs_b, scs_c,
        scs_cones,
        solve_method='SCS',
        verbose=False,
    )

    print('values from SCS pipeline')
    print('x=', x)
    print('obj val=', scs_c.T @ x)
    # print(DT(x, y, s))

    print('--------')
    print('even with the clarabel solver, diffcp expects SCS canonicalization ordering')
    print('the solve_and_derivative function reorders the rows of A internally')
    x, y, s, D, DT = diffcp.solve_and_derivative(sparse.csc_matrix(scs_A), scs_b, scs_c,
        scs_cones,
        solve_method='CLARABEL',
        verbose=False,
    )

    print('values from clarabel pipeline')
    print('x=', x)
    print('obj val=', scs_c.T @ x)
    # print(DT(x, y, s))

    print('optimal objective values should match')


if __name__ == '__main__':
    main()
