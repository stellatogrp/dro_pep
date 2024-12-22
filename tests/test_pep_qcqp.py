import gurobipy as grb
import numpy as np


# Problem data
# Set seed
np.random.seed(0)
n = 5
K = 3
v = np.random.randn(n)
mu = 0.2
L = 3
gamma = 1 / L
r = 1.0


# Create gurobi model
model = grb.Model()

# Maximize the squared 2-norm of x[1]
x = {}
f = {}
g = {}
for k in range(K):
    if k == 0:
        x[k] = model.addMVar(n, ub=v + r, lb=v - r)
    else:
        x[k] = model.addMVar(n, ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)

    g[k] = model.addMVar(n, ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)
    f[k] = model.addVar(ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)


# objective
model.setObjective(x[K - 1] @ x[K - 1], grb.GRB.MAXIMIZE)

# initial condition
model.addConstr((x[0] - v) @ (x[0] - v) <= r**2)

# add iterations
for k in range(K - 1):
    model.addConstr(x[k + 1] == x[k] - gamma * g[k])

# interpolation conditions
for k1 in range(K):
    for k2 in range(K):
        model.addConstr(
            f[k1] - f[k2] - g[k2] @ (x[k1] - x[k2])
            >= 1
            / (2 * (1 - mu / L))
            * (
                1 / L * (g[k1] - g[k2]) @ (g[k1] - g[k2])
                + mu * (x[k1] - x[k2]) @ (x[k1] - x[k2])
                - 2 * mu / L * (g[k2] - g[k1]) @ (x[k2] - x[k1])
            )
        )

model.optimize()
grb_val = model.objVal

# extract x g and f values
x_val = [x[k].X for k in range(K)]
g_val = [g[k].X for k in range(K)]
f_val = [f[k].X for k in range(K)]
