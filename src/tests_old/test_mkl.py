import clarabel
import numpy as np
from scipy.sparse import csc_matrix

# 1. Define the problem data for a simple quadratic program (QP)
# minimize    (1/2)x'Px + q'x
# subject to  Ax + s = b
#             s in K (where K is a cone)

P = csc_matrix([[3, 0], [0, 4]], dtype=np.float64)
q = np.array([2, -5], dtype=np.float64)

A = csc_matrix([[1, 1]], dtype=np.float64)
b = np.array([1], dtype=np.float64)

# Define the cone constraints
cones = [clarabel.ZeroConeT(1)]

# 2. Create and configure a settings object for Pardiso
settings = clarabel.DefaultSettings()
# settings.direct_solve_method = 'qdldl' # or 'panua' if using Panua Pardiso
settings.direct_solve_method = 'mkl'
settings.verbose = True # Optional: turn on verbose output to see solver progress

# 3. Create the solver object
solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

# 4. Solve the problem
solution = solver.solve()

# 5. Print the results
print("Solution x:", solution.x)
print("Dual solution z:", solution.z)
