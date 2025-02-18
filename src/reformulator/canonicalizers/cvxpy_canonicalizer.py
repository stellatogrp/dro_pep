import cvxpy as cp

from .canonicalizer import Canonicalizer


class CvxpyCanonicalizer(Canonicalizer):

    def __init__(self, pep_problem, samples, measure, wrapper):
        super().__init__(pep_problem, samples, measure, wrapper)

    def setup_problem(self):
        if self.measure == 'expectation':
            self.setup_expectation_problem()
        if self.measure == 'cvar':
            self.setup_cvar_problem()

    def setup_expectation_problem(self):
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
        # obj = lambd * eps
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

        # probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        # A_cp = probdata['A']
        # print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.eps_param = eps

        assert self.cp_problem.is_dpp('dcp')

    def set_eps(self, eps):
        self.eps_param.value = eps

    def setup_cvar_problem(self):
        raise NotImplementedError

    def solve(self):
        res = self.cp_problem.solve(solver=cp.CLARABEL)
        out = {
            'obj': res,
            'solvetime': self.cp_problem.solver_stats.solve_time,
        }
        return out
