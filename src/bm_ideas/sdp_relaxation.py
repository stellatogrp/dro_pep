##### Generates matrices for linear matrix inequality at SDP formulation #####
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from math import sqrt
from sdp_data_matrix_scalable import smooth_convex_gd_close_to_sample, smooth_convex_agm_close_to_sample, smooth_convex_hbm_close_to_sample
import cvxpy as cp

d = 20
K = 10
mu = 0.1
L = 1.0
gamma = 1.0/L
radius = 0.01
# A, b, C = smooth_convex_gd_close_to_sample(d, K, mu, L, gamma, radius)
# A, b, C = smooth_convex_agm_close_to_sample(d, K, 0.0, L, 1.0/L, radius)
A, b, C = smooth_convex_hbm_close_to_sample(d, K, 0.0, L, 1.0/L, radius)

## Optimization variable:
# R @ R.T = [ P.T@P  P.T    O       O   ]   (P.T@P with each element being inner product values)
#           [   P     I     O       O   ]   (P as stack of x0 and gradients)
#           [   O     O   diag(f)   O   ]   (function values as diagonal components)
#           [   O     O     O    diag(s)]   (slack variable for inequalities)
# Here, G = R @ R.T and G >> 0 is a relaxation for above structure.
G = cp.Variable((K+1+d, K+1+d), symmetric=True)

constraints = [
    G[K+1:,K+1:] == np.eye(d),
    G >> 0
]

F = cp.Variable((K+1))

constraints += [
    F >= 0.0
]

m = len(b)
for i in range(m):
    constraints += [
        cp.trace(G@A[i].G) + cp.sum(F*A[i].F) <= b[i]
    ]

objective = cp.Minimize(cp.trace(G@C.G) + cp.sum(F@C.F))

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=True)


print('\n\n[[Result]]\n')

tau = problem.value
print('[1] Objective value:', tau)

PTP = G[:K+1,:K+1].value
P = G[K+1:,:K+1].value
_, V, _ = np.linalg.svd(PTP - P.T@P)
print('\n[2] Eigenvalues of G - P.T@P:', V)
print('(If the value above is all close to 0, then G ~ P.T@P)\n')

dual_values = []
slack_values = []
for i in range(m):
    dual_values.append(constraints[-m+i].dual_value)

## Store the results
data_dict = {}
data_dict['dimension'] = d
data_dict['iter_num'] = K
data_dict['trajectory_info'] = P
data_dict['inner_product'] = PTP
data_dict['objective_values'] = (F.value)[:-1]
data_dict['dual_var'] = dual_values
np.save(f'result/sdp_data_d={d}_K={K}_radius={radius}.npy', data_dict)