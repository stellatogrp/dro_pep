import numpy as np
from PEPit import PEP
from PEPit import Point, BlockPartition, Function
from PEPit.functions import (
    SmoothStronglyConvexFunction,
    SmoothStronglyConvexQuadraticFunction,
    ConvexFunction,
    ConvexLipschitzFunction,
)
from PEPit.primitive_steps import proximal_step
from PEPit.tools.expressions_to_matrices import expression_to_matrices


def test_pep():
    problem = PEP()
    mu = 0.1
    L = 1
    gamma = 0.5 / L
    K = 2
    lambd = 0.1

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L, reuse_gradient=True)
    # f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L, reuse_gradient=True)
    # f2 = problem.declare_function(ConvexFunction)
    f2 = problem.declare_function(ConvexLipschitzFunction, M=lambd, reuse_gradient=True)
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # g0 = f1.gradient(x0)
    # f0 = func(x0)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 4)

    # Run the proximal gradient method starting from x0

    x = [x0 for _ in range(K+1)]
    g = [None for _ in range(K+1)]
    f = [None for _ in range(K+1)]

    # x = x0
    for k in range(K):
        # y = x - gamma * f1.gradient(x)
        # x, gx, fx = proximal_step(y, f2, gamma)

        y = x[k] - gamma * f1.gradient(x[k])

        x[k+1], g[k+1], f[k+1] = proximal_step(y, f2, gamma)

    # Set the performance metric to the distance between x and xs
    # problem.set_performance_metric((x - xs) ** 2)
    # problem.set_performance_metric(func(x0) - func(xs))
    # problem.set_performance_metric(f[-1] - func(xs))
    problem.set_performance_metric(func(x[-1]) - func(xs))


    # Solve the PEP
    # pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # problem.set_initial_condition(x[1] ** 2 <= 9)
    # problem.set_initial_condition(func(x[2]) <= 7)
    problem.set_initial_condition(f1.gradient(xs) ** 2 <= 7)

    print(Point.list_of_leaf_points)

    # pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    pepit_tau = problem.solve()

    # print(Point.list_of_leaf_points)
    # print(f1.list_of_stationary_points[0][0])
    # print(Point.list_of_leaf_points)
    # exit(0)

    print(pepit_tau)

    A_obj, b_obj, _ = expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression)
    print(A_obj, b_obj)
    print(A_obj.shape, b_obj.shape)

    print(len(problem._list_of_constraints_sent_to_wrapper))
    for constr in problem._list_of_constraints_sent_to_wrapper[1:4]:
        A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)
        print(constr.equality_or_inequality)

        print(A_cons, b_cons, c_cons)



if __name__ == '__main__':
    test_pep()
