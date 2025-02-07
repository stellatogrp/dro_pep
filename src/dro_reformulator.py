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
                # raise NotImplementedError # TODO add the extra constraints for the double sided inequalities
                A_vals.append(-A_cons)
                b_vals.append(-b_cons)
                c_vals.append(-c_cons)
        self.A_vals = np.array(A_vals)
        self.b_vals = np.array(b_vals)
        self.c_vals = np.array(c_vals)

        assert self.A_vals.shape[0] == self.b_vals.shape[0] == self.c_vals.shape[0]

    def setup_cvxpy_problem(self):
        if self.measure == 'expectation':
            self.setup_cvxpy_expectation_problem()
        if self.measure == 'cvar':
            self.setup_cvxpy_cvar_problem()

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
        # TODO: check DPP just in case is_dpp 

    def setup_cvxpy_cvar_problem(self):
        N = len(self.samples)
        M = len(self.A_vals)
        mat_shape = self.A_obj.shape
        vec_shape = self.b_obj.shape

        lambd = cp.Variable()
        s = cp.Variable(N)
        y1 = cp.Variable((N, M))
        y2 = cp.Variable((N, M))
        t = cp.Variable()

        Gz1 = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
        Fz1 = [cp.Variable(vec_shape) for _ in range(N)]
        Gz2 = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
        Fz2 = [cp.Variable(vec_shape) for _ in range(N)]

        eps = cp.Parameter()
        alpha_inv = cp.Parameter()

        obj = lambd * eps + 1 / N * cp.sum(s)
        constraints = [y1 >= 0, y2 >= 0]

        for i in range(N):
            G_sample, F_sample = self.samples[i]
            # constraints += [-self.c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]
            constraints += [t -self.c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
            constraints += [-(alpha_inv - 1) * t -self.c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]

            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz1[i]), Fz1[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz2[i]), Fz2[i]]))]

            y1A_adj = 0
            y2A_adj = 0
            y1b_adj = 0
            y2b_adj = 0

            for m in range(M):
                Am = self.A_vals[m]
                bm = self.b_vals[m]

                y1A_adj = y1A_adj + y1[i, m] * Am
                y2A_adj = y2A_adj + y2[i, m] * Am

                y1b_adj = y1b_adj + y1[i, m] * bm
                y2b_adj = y2b_adj + y2[i, m] * bm

            constraints += [y1A_adj - Gz1[i] >> 0]
            constraints += [y1b_adj - Fz1[i] == 0]
            constraints += [y2A_adj - Gz2[i] - alpha_inv * self.A_obj >> 0]
            constraints += [y2b_adj - Fz2[i] - alpha_inv * self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        self.cp_problem = prob
        self.eps_param = eps
        self.alpha_inv_param = alpha_inv

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

    def solve_fixed_alpha_eps_vals(self, alpha, eps_vals):
        self.alpha_inv_param.value = 1 / alpha
        out = []
        for eps in eps_vals:
            print(f'solving eps={eps}')
            res = self.solve_single_eps_val(eps)
            out.append(res)
        return np.array(out)
