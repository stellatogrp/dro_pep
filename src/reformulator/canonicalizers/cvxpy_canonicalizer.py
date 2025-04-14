import cvxpy as cp
import numpy as np

from .canonicalizer import Canonicalizer


class CvxpyCanonicalizer(Canonicalizer):

    def __init__(self, pep_problem, samples, measure, wrapper, precond=True):
        super().__init__(pep_problem, samples, measure, wrapper, precond=precond)

    def setup_problem(self):
        if self.measure == 'expectation':
            self.setup_expectation_problem()
        if self.measure == 'cvar':
            self.setup_cvar_problem()

    def setup_expectation_problem(self):
        samples_to_use = self.samples_to_use

        N = len(samples_to_use)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)
        mat_shape = self.A_obj.shape
        vec_shape = self.b_obj.shape

        lambd = cp.Variable()
        s = cp.Variable(N)
        y = cp.Variable((N, M))
        Gz = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
        Fz = [cp.Variable(vec_shape) for _ in range(N)]

        Gz_psd = [[cp.Variable(self.PSD_shapes[m_psd], PSD=True) for m_psd in range(M_psd)] for _ in range(N)]
        eps = cp.Parameter()

        obj = lambd * eps + 1 / N * cp.sum(s)
        # obj = lambd * eps
        constraints = [y >= 0]

        G_preconditioner = np.diag(self.preconditioner[0])
        F_preconditioner = self.preconditioner[1]

        for i in range(N):
            G_sample, F_sample = samples_to_use[i]
            constraints += [- self.c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz[i]@G_preconditioner ), cp.multiply(F_preconditioner**2, Fz[i])]))]

            LstarG = 0
            LstarF = 0
            for m in range(M):
                Am = self.A_vals[m]
                bm = self.b_vals[m]
                LstarG = LstarG + y[i, m] * Am
                LstarF = LstarF + y[i, m] * bm

            for m_psd in range(M_psd):
                Am_psd = self.PSD_A_vals[m_psd]
                bm_psd = self.PSD_b_vals[m_psd]
                # cm_psd = self.PSD_c_vals[m_psd] # TODO: deal with cm_psd (s[i] constraint)
                for j in range(self.PSD_shapes[m_psd][0]) :
                    for k in range(self.PSD_shapes[m_psd][1]) :
                        LstarG = LstarG - Gz_psd[i][m_psd][j,k] * Am_psd[j,k]
                        LstarF = LstarF - Gz_psd[i][m_psd][j,k] * bm_psd[j,k]
                        
            constraints += [LstarG - Gz[i] - self.A_obj >> 0]
            constraints += [LstarF - Fz[i] - self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        A_cp = probdata['A']
        # print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = None
        self.eps_param = eps

        assert self.cp_problem.is_dpp('dcp')

    def set_eps(self, eps):
        self.eps_param.value = eps

    def setup_cvar_problem(self):
        samples_to_use = self.samples_to_use

        N = len(samples_to_use)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)
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

        Gz1_psd = [[cp.Variable(self.PSD_shapes[m_psd], PSD=True) for m_psd in range(M_psd)] for _ in range(N)]
        Gz2_psd = [[cp.Variable(self.PSD_shapes[m_psd], PSD=True) for m_psd in range(M_psd)] for _ in range(N)]
        
        eps = cp.Parameter()
        alpha_inv = cp.Parameter()

        G_preconditioner = np.diag(self.preconditioner[0])
        F_preconditioner = self.preconditioner[1]

        obj = lambd * eps + 1 / N * cp.sum(s)
        constraints = [y1 >= 0, y2 >= 0]

        for i in range(N):
            G_sample, F_sample = samples_to_use[i]
            constraints += [t - self.c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
            # constraints += [t <= s[i]]
            constraints += [-(alpha_inv - 1) * t - self.c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]

            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz2[i]), Fz2[i]]))]
            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz1[i]), Fz1[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz1[i]@G_preconditioner ), cp.multiply(F_preconditioner**2, Fz1[i])]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz2[i]@G_preconditioner ), cp.multiply(F_preconditioner**2, Fz2[i])]))]

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

            for m_psd in range(M_psd):
                Am_psd = self.PSD_A_vals[m_psd]
                bm_psd = self.PSD_b_vals[m_psd]
                # cm_psd = self.PSD_c_vals[m_psd] # TODO: deal with cm_psd (s[i] constraint)
                for j in range(self.PSD_shapes[m_psd][0]) :
                    for k in range(self.PSD_shapes[m_psd][1]) :
                        y1A_adj = y1A_adj - Gz1_psd[i][m_psd][j,k] * Am_psd[j,k]
                        y1b_adj = y1b_adj - Gz1_psd[i][m_psd][j,k] * bm_psd[j,k]
                        y2A_adj = y2A_adj - Gz2_psd[i][m_psd][j,k] * Am_psd[j,k]
                        y2b_adj = y2b_adj - Gz2_psd[i][m_psd][j,k] * bm_psd[j,k]

            constraints += [y1A_adj - Gz1[i] >> 0]
            constraints += [y1b_adj - Fz1[i] == 0]
            constraints += [y2A_adj - Gz2[i] - alpha_inv * self.A_obj >> 0]
            constraints += [y2b_adj - Fz2[i] - alpha_inv * self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        A_cp = probdata['A']
        # print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = t
        self.eps_param = eps
        self.alpha_inv_param = alpha_inv

    def set_eps_alpha_value(self, eps, alpha):
        self.eps_param.value = eps
        self.alpha_inv_param.value = 1 / alpha

    def set_params(self, eps=0.1, alpha=0.1):
        if self.measure == 'expectation':
            self.set_eps(eps)
        elif self.measure == 'cvar':
            self.set_eps_alpha_value(eps, alpha)

    def solve(self):
        res = self.cp_problem.solve(solver=cp.CLARABEL)
        out = {
            'obj': res,
            'solvetime': self.cp_problem.solver_stats.solve_time,
        }
        return out
