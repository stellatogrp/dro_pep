import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from PEPit.tools.expressions_to_matrices import expression_to_matrices

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


VALID_MEASURES = [
    'expectation',
    'cvar',
]

VALID_WRAPPERS = [
    'cvxpy',
]

class DROReformulator(object):

    def __init__(self, pep_problem, samples, measure, wrapper):
        self.pep_problem = pep_problem
        self.samples = samples
        if measure not in VALID_MEASURES:
            raise NotImplementedError('not a valid measure')

        if pep_problem.objective is None:
            raise AssertionError('pep problem needs to be solved to extract data')

        self.measure = measure

        self.extract_pep_data()
        
        if wrapper == 'cvxpy':
            self.setup_cvxpy_problem()
        else:
            raise NotImplementedError(f'wrapper {wrapper} not implemented')

    def extract_pep_data(self):
        problem = self.pep_problem
        A_obj, b_obj, _ = expression_to_matrices(problem.objective)
        self.A_obj = A_obj
        self.b_obj = b_obj

        A_vals = []
        b_vals = []
        c_vals = []
        for constr in problem._list_of_constraints_sent_to_wrapper:
            A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)

            A_vals.append(A_cons)
            b_vals.append(b_cons)
            c_vals.append(c_cons)

            if constr.equality_or_inequality == 'equality':
                raise NotImplementedError # TODO add the extra constraints for the double sided inequalities
        self.A_vals = np.array(A_vals)
        self.b_vals = np.array(b_vals)
        self.c_vals = np.array(c_vals)

        assert self.A_vals.shape[0] == self.b_vals.shape[0] == self.c_vals.shape[0]

    def setup_cvxpy_problem(self):
        if self.measure == 'expectation':
            self.setup_cvxpy_expectation_problem()

    def setup_cvxpy_expectation_problem(self):
        N = len(self.samples)
        M = len(self.A_vals)
        mat_shape = self.A_obj.shape
        vec_shape = self.b_obj.shape

        lambd = cp.Variable()
        s = cp.Variable(N)
        y = cp.Variable((N, M))
        Gz = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
        Fz = [cp.Variable(vec_shape) for _ in range(N)]

        eps = cp.Parameter()

        obj = lambd * eps + 1 / N * cp.sum(s)
        constraints = [y >= 0]

        for i in range(N):
            G_sample, F_sample = self.samples[i]
            constraints += [-self.c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]

            LstarG = 0
            LstarF = 0
            for m in range(M):
                Am = self.A_vals[m]
                bm = self.b_vals[m]
                LstarG = LstarG + y[i, m] * Am
                LstarF = LstarF + y[i, m] * bm
            constraints += [LstarG - Gz[i] - self.A_obj >> 0]
            constraints += [LstarF - Fz[i] - self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        self.cp_problem = prob
        self.eps_param = eps

    def solve_eps_vals(self, eps_vals):
        out = []
        for eps in eps_vals:
            print(f'solving eps={eps}')
            res = self.solve_single_eps_val(eps)
            out.append(res)
        return np.array(out)

    def solve_single_eps_val(self, eps):
        self.eps_param.value = eps
        res = self.cp_problem.solve(solver=cp.CLARABEL)
        return res
