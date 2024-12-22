from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
import cvxpy as cp
import numpy as np

# algorithm parameters
# Example from "PEPIT: computer-assisted worst-case analyses of first-order optimization methods in Python" https://doi.org/10.1007/s12532-024-00259-7
gamma = 0.4
mu = 0.2
L = 3


# Write problem in pepit
problem = PEP()
func = problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=mu, L=L)
x0 = problem.set_initial_point()
x1 = x0 - gamma * func.gradient(x0)
y0 = problem.set_initial_point()
y1 = y0 - gamma * func.gradient(y0)
problem.set_initial_condition((x0 - y0) ** 2 <= 1)
problem.set_performance_metric((x1 - y1) ** 2)
tau = problem.solve(
    wrapper="cvxpy", solver="CLARABEL", dimension_reduction_heuristic="trace", verbose=1
)
print("Contraction factor = ", tau)

# Write problem directly with Gram matrix
G = cp.Variable((3, 3), symmetric=True)
fx0 = cp.Variable()
fy0 = cp.Variable()
objective = cp.Maximize(
    G[0, 0] - 2 * gamma * (G[0, 1] - G[0, 2]) + gamma**2 * (G[1, 1] + G[2, 2] - 2 * G[1, 2])
)
constraints = [
    G >> 0,
    G[0, 0] <= 1,
    fy0
    >= fx0
    + 1
    / (L - mu)
    * (
        mu * L / 2 * G[0, 0]
        - L * G[0, 1]
        + mu * G[0, 2]
        + (1 / 2) * G[1, 1]
        - G[1, 2]
        + (1 / 2) * G[2, 2]
    ),
    fx0
    >= fy0
    + 1
    / (L - mu)
    * (
        mu * L / 2 * G[0, 0]
        - mu * G[0, 1]
        + L * G[0, 2]
        + (1 / 2) * G[1, 1]
        - G[1, 2]
        + (1 / 2) * G[2, 2]
    ),
]
prob_cvxpy = cp.Problem(objective, constraints)
prob_cvxpy.solve(solver=cp.CLARABEL, verbose=False)
tau_cvxpy = prob_cvxpy.value
print("Contraction factor cvxpy = ", tau_cvxpy)


# Write gram matrix constraints directly
G = cp.Variable((3, 3), symmetric=True)
fx0 = cp.Variable()
fy0 = cp.Variable()

# Objective
C = np.zeros((3, 3))
C[0, 0] = 1
C[0, 1] = -2 * gamma
C[0, 2] = 2 * gamma
C[1, 1] = gamma**2
C[2, 2] = gamma**2
C[1, 2] = -2 * gamma**2
objective = cp.Maximize(cp.trace(C @ G))

# Constraints
A = np.zeros((3, 3, 3))
A[0, 0, 0] = 1

A[1, 0, 0] = mu * L / 2
A[1, 0, 1] = -L
A[1, 0, 2] = mu
A[1, 1, 1] = 1 / 2
A[1, 1, 2] = -1
A[1, 2, 2] = 1 / 2
A[1, :, :] *= 1 / (L - mu)

A[2, 0, 0] = mu * L / 2
A[2, 0, 1] = -mu
A[2, 0, 2] = L
A[2, 1, 1] = 1 / 2
A[2, 1, 2] = -1
A[2, 2, 2] = 1 / 2
A[2, :, :] *= 1 / (L - mu)

constraints += [
    cp.trace(A[0, :, :] @ G) <= 1,
    cp.trace(A[1, :, :] @ G) + fx0 - fy0 <= 0,
    cp.trace(A[2, :, :] @ G) + fy0 - fx0 <= 0,
    G >> 0,
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=False)
tau_sdp = problem.value

print("Contraction factor sdp = ", tau_sdp)
print("Difference cvxpy and pepit = ", np.abs(tau - tau_cvxpy))
print("Difference cvxpy and sdp = ", np.abs(tau_cvxpy - tau_sdp))

# Schur complement to represent points
dim = 10
I = np.eye(dim)
P = cp.Variable((dim, 3))
G = cp.Variable((3, 3), symmetric=True)
fx0 = cp.Variable()
fy0 = cp.Variable()
objective = cp.Maximize(cp.trace(C @ G))
constraints += [
    cp.trace(A[0, :, :] @ G) <= 1,
    cp.trace(A[1, :, :] @ G) + fx0 - fy0 <= 0,
    cp.trace(A[2, :, :] @ G) + fy0 - fx0 <= 0,
    cp.bmat([[G, P.T], [P, I]]) >> 0,
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=True)
tau_schur = problem.value
print("Contraction factor schur = ", tau_schur)
