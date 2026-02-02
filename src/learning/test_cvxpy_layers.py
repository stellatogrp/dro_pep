import os

# 1. Get the number of available cores
try:
    # Linux/Unix specific: respects container limits (Docker/Slurm)
    # This is the "safe" way on a cluster
    num_cores = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for Mac/Windows (or if sched_getaffinity is missing)
    # os.cpu_count() returns None if undetermined, so we default to 1
    num_cores = os.cpu_count() or 1

# 2. Set the XLA flag dynamically
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"

import cvxpy as cp
import numpy as np

# 3. NOW import JAX
import jax

print(f"JAX sees {jax.device_count()} devices on CPU.")

from cvxpylayers.jax import CvxpyLayer

np.set_printoptions(suppress=True, precision=5)

n, m = 10, 15
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
A_jax = jax.random.normal(k1, shape=(m, n))
b_jax = jax.random.normal(k2, shape=(m,))

(solution,) = layer(A_jax, b_jax)

# compute the gradient of the summed solution with respect to A, b
dlayer = jax.grad(lambda A, b: sum(layer(A, b)[0]), argnums=[0, 1])
gradA, gradb = dlayer(A_jax, b_jax)

print(gradA, gradb)