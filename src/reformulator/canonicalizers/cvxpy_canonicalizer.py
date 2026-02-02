import cvxpy as cp
import numpy as np

from .custom_interp_canonicalizer import CustomInterpCanonicalizer


class CvxpyCanonicalizer(CustomInterpCanonicalizer):

    def __init__(self, pep_data, samples, measure, wrapper, precond=True, precond_type='average', mro_clusters=None):
        super().__init__(pep_data, samples, measure, wrapper, precond=precond, precond_type=precond_type, mro_clusters=mro_clusters)

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

        # G_preconditioner = np.diag(self.preconditioner[0])
        # F_preconditioner = self.preconditioner[1]

        G_preconditioner = np.diag(self.precond_inv[0])
        F_preconditioner = self.precond_inv[1]

        for i in range(N):
            G_sample, F_sample = samples_to_use[i]
            constraints += [- self.c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz[i])]))]

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

        # probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        # A_cp = probdata['A']
        # print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = None
        self.eps_param = eps

        self.lambda_var = lambd
        self.s_var = s
        self.y_var = y
        self.Gz_var = Gz
        self.Fz_var = Fz
        self.H_var = Gz_psd

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

        self.lambda_var = lambd
        self.t_var = t
        self.s_var = s
        self.y1_var = y1
        self.Gz1_var = Gz1
        self.Fz1_var = Fz1

        self.y2_var = y1
        self.Gz2_var = Gz2
        self.Fz2_var = Fz2

        self.H1_var = Gz1_psd
        self.H2_var = Gz2_psd
        
        eps = cp.Parameter()
        alpha_inv = cp.Parameter()

        # G_preconditioner = np.diag(self.preconditioner[0])
        # F_preconditioner = self.preconditioner[1]

        G_preconditioner = np.diag(self.precond_inv[0])
        F_preconditioner = self.precond_inv[1]

        obj = lambd * eps + 1 / N * cp.sum(s)
        constraints = [y1 >= 0, y2 >= 0]

        for i in range(N):
            G_sample, F_sample = samples_to_use[i]
            constraints += [t - self.c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
            # constraints += [t <= s[i]]
            constraints += [-(alpha_inv - 1) * t - self.c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]

            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz2[i]), Fz2[i]]))]
            # constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz1[i]), Fz1[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz1[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz1[i])]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz2[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz2[i])]))]

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

    def extract_solution(self):
        if self.measure == 'expectation':
            return {
                'lambda': self.lambda_var.value,
                's': self.s_var.value,
                'y': self.y_var.value,
                'Gz': [G.value for G in self.Gz_var],
                'Fz': [F.value for F in self.Fz_var],
                # 'H': [[cp.Variable(self.PSD_shapes[m_psd], PSD=True) for m_psd in range(M_psd)] for _ in range(N)]
                'H': [[Hi_m.value for Hi_m in Hi]for Hi in self.H_var],
            }
        elif self.measure == 'cvar':
            return {
                'lambda': self.lambda_var.value,
                's': self.s_var.value,
                't': self.t_var.value,
                'y1': self.y1_var.value,
                'y2': self.y2_var.value,
                'Gz1': [G.value for G in self.Gz1_var],
                'Gz2': [G.value for G in self.Gz2_var],
                'Fz1': [F.value for F in self.Fz1_var],
                'Fz2': [F.value for F in self.Fz2_var],
                'H1': [[Hi_m.value for Hi_m in Hi]for Hi in self.H1_var],
                'H2': [[Hi_m.value for Hi_m in Hi]for Hi in self.H2_var],
            }
