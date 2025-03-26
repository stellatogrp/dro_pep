import clarabel
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

from PEPit.tools.expressions_to_matrices import expression_to_matrices

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


VALID_MEASURES = [
    'expectation',
    'cvar',
    'expectation_mro',
    'cvar_mro',
    'expectation_reduced'
]

VALID_WRAPPERS = [
    'cvxpy',
    'clarabel',
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
        self.wrapper = wrapper
        self.mro_diff = None

        self.extract_pep_data()
        
        if wrapper == 'cvxpy':
            self.setup_cvxpy_problem()
        elif wrapper == 'clarabel':
            self.setup_clarabel_problem()
        else:
            raise NotImplementedError(f'wrapper {wrapper} not implemented')

    def extract_pep_data(self):
        problem = self.pep_problem
        A_obj, b_obj, _ = expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression)
        self.A_obj = - A_obj
        self.b_obj = - b_obj[:-1]

        A_vals = []
        b_vals = []
        c_vals = []
        for constr in problem._list_of_constraints_sent_to_wrapper[1:]:
            A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)

            A_vals.append(A_cons)
            b_vals.append(b_cons[:-1])
            c_vals.append(c_cons)

            if constr.equality_or_inequality == 'equality':
                # raise NotImplementedError # TODO add the extra constraints for the double sided inequalities
                A_vals.append(-A_cons)
                b_vals.append(-b_cons[:-1])
                c_vals.append(-c_cons)
        self.A_vals = np.array(A_vals)
        self.b_vals = np.array(b_vals)
        self.c_vals = np.array(c_vals)

        # TODO: PSD constraint for quadratic interpolation
        PSD_A_vals = []
        PSD_b_vals = []
        PSD_c_vals = []
        PSD_shapes = []

        for psd in problem._list_of_psd_sent_to_wrapper:
            psd_expr = psd.matrix_of_expressions
            psd_shape = psd_expr.shape
            PSD_shapes.append(psd_shape)

            PSD_A_val, PSD_b_val, PSD_c_val = [], [], []
            for i in range(psd_shape[0]) :
                PSD_A_row, PSD_b_row, PSD_c_row = [], [], []
                for j in range(psd_shape[1]) :
                    PSD_A_cons, PSD_b_cons, PSD_c_cons = expression_to_matrices(psd_expr[i,j])
                    PSD_A_row.append(PSD_A_cons)
                    PSD_b_row.append(PSD_b_cons[:-1])
                    PSD_c_row.append(PSD_c_cons)
                PSD_A_val.append(PSD_A_row)
                PSD_b_val.append(PSD_b_row)
                PSD_c_val.append(PSD_c_row)
            PSD_A_vals.append(np.array(PSD_A_val))
            PSD_b_vals.append(np.array(PSD_b_val))
            PSD_c_vals.append(np.array(PSD_c_val))
        self.PSD_A_vals = PSD_A_vals
        self.PSD_b_vals = PSD_b_vals
        self.PSD_c_vals = PSD_c_vals
        self.PSD_shapes = PSD_shapes

        assert self.A_vals.shape[0] == self.b_vals.shape[0] == self.c_vals.shape[0]

    def setup_cvxpy_problem(self):
        if self.measure == 'expectation':
            self.setup_cvxpy_expectation_problem()
        if self.measure == 'expectation_mro':
            self.setup_cvxpy_expectation_mro_problem()
        if self.measure == 'cvar':
            self.setup_cvxpy_cvar_problem()
        if self.measure == 'cvar_mro':
            self.setup_cvxpy_cvar_mro_problem()

    def setup_cvxpy_expectation_problem(self):
        N = len(self.samples)
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

        for i in range(N):
            G_sample, F_sample = self.samples[i]
            constraints += [- self.c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz[i]), Fz[i]]))]

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
        print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = None
        self.eps_param = eps
        # TODO: check DPP just in case is_dpp 

    def setup_cvxpy_expectation_mro_problem(self):
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)
        mat_shape = self.A_obj.shape
        vec_shape = self.b_obj.shape
        avg_G = np.average([sample[0] for sample in self.samples], axis=0)
        avg_F = np.average([sample[1] for sample in self.samples], axis=0)

        lambd = cp.Variable()
        s = cp.Variable()
        y = cp.Variable(M)
        Gz = cp.Variable(mat_shape, symmetric=True)
        Fz = cp.Variable(vec_shape)
        Gz_psd = [cp.Variable(self.PSD_shapes[m], PSD=True) for m in range(M_psd)]

        eps = cp.Parameter()

        obj = lambd * eps + s
        # obj = lambd * eps
        constraints = [y >= 0]

        constraints += [- self.c_vals.T @ y - cp.trace(avg_G @ Gz) - avg_F.T @ Fz <= s]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz), Fz]))]

        LstarG = 0
        LstarF = 0
        for m in range(M):
            Am = self.A_vals[m]
            bm = self.b_vals[m]
            LstarG = LstarG + y[m] * Am
            LstarF = LstarF + y[m] * bm

        for m_psd in range(M_psd):
            Am_psd = self.PSD_A_vals[m_psd]
            bm_psd = self.PSD_b_vals[m_psd]
            # cm_psd = self.PSD_c_vals[m_psd] # TODO: deal with cm_psd (s[i] constraint)
            for j in range(self.PSD_shapes[m_psd][0]) :
                for k in range(self.PSD_shapes[m_psd][1]) :
                    LstarG = LstarG - Gz_psd[m_psd][j,k] * Am_psd[j,k]
                    LstarF = LstarF - Gz_psd[m_psd][j,k] * bm_psd[j,k]
                    
        constraints += [LstarG - Gz - self.A_obj >> 0]
        constraints += [LstarF - Fz - self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        A_cp = probdata['A']
        print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = None
        self.eps_param = eps
        # TODO: check DPP just in case is_dpp 

        self.mro_diff = self.calculate_mro_diff([y], [Gz], [Fz], [Gz_psd])
    
    def setup_cvxpy_cvar_problem(self):
        N = len(self.samples)
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

        obj = lambd * eps + 1 / N * cp.sum(s)
        constraints = [y1 >= 0, y2 >= 0]

        for i in range(N):
            G_sample, F_sample = self.samples[i]
            constraints += [t - self.c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
            # constraints += [t <= s[i]]
            constraints += [-(alpha_inv - 1) * t - self.c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]

            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz2[i]), Fz2[i]]))]
            constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz1[i]), Fz1[i]]))]

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
        print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = t
        self.eps_param = eps
        self.alpha_inv_param = alpha_inv
        

    def setup_cvxpy_cvar_mro_problem(self):
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)
        mat_shape = self.A_obj.shape
        vec_shape = self.b_obj.shape
        avg_G = np.average([sample[0] for sample in self.samples], axis=0)
        avg_F = np.average([sample[1] for sample in self.samples], axis=0)

        lambd = cp.Variable()
        s = cp.Variable()
        y1 = cp.Variable(M)
        y2 = cp.Variable(M)
        t = cp.Variable()

        Gz1 = cp.Variable(mat_shape, symmetric=True)
        Fz1 = cp.Variable(vec_shape)
        Gz2 = cp.Variable(mat_shape, symmetric=True)
        Fz2 = cp.Variable(vec_shape)

        Gz1_psd = [cp.Variable(self.PSD_shapes[m], PSD=True) for m in range(M_psd)]
        Gz2_psd = [cp.Variable(self.PSD_shapes[m], PSD=True) for m in range(M_psd)]
        

        eps = cp.Parameter()
        alpha_inv = cp.Parameter()

        obj = lambd * eps + s
        constraints = [y1 >= 0, y2 >= 0]

        constraints += [t - self.c_vals.T @ y1 - cp.trace(avg_G @ Gz1) - avg_F.T @ Fz1 <= s]
        constraints += [-(alpha_inv - 1) * t - self.c_vals.T @ y2 - cp.trace(avg_G @ Gz2) - avg_F.T @ Fz2 <= s]

        constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz1), Fz1]))]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec(Gz2), Fz2]))]

        y1A_adj = 0
        y2A_adj = 0
        y1b_adj = 0
        y2b_adj = 0

        for m in range(M):
            Am = self.A_vals[m]
            bm = self.b_vals[m]

            y1A_adj = y1A_adj + y1[m] * Am
            y2A_adj = y2A_adj + y2[m] * Am

            y1b_adj = y1b_adj + y1[m] * bm
            y2b_adj = y2b_adj + y2[m] * bm

        for m_psd in range(M_psd):
            Am_psd = self.PSD_A_vals[m_psd]
            bm_psd = self.PSD_b_vals[m_psd]
            # cm_psd = self.PSD_c_vals[m_psd] # TODO: deal with cm_psd (s[i] constraint)
            for j in range(self.PSD_shapes[m_psd][0]) :
                for k in range(self.PSD_shapes[m_psd][1]) :
                    y1A_adj = y1A_adj - Gz1_psd[m_psd][j,k] * Am_psd[j,k]
                    y1b_adj = y1b_adj - Gz1_psd[m_psd][j,k] * bm_psd[j,k]
                    y2A_adj = y2A_adj - Gz2_psd[m_psd][j,k] * Am_psd[j,k]
                    y2b_adj = y2b_adj - Gz2_psd[m_psd][j,k] * bm_psd[j,k]


        constraints += [y1A_adj - Gz1 >> 0]
        constraints += [y1b_adj - Fz1 == 0]
        constraints += [y2A_adj - Gz2 - alpha_inv * self.A_obj >> 0]
        constraints += [y2b_adj - Fz2 - alpha_inv * self.b_obj == 0]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        A_cp = probdata['A']
        print('A shape from cvxpy:', A_cp.shape)

        self.cp_problem = prob
        self.quantile_estimate = t
        self.eps_param = eps
        self.alpha_inv_param = alpha_inv
        
        self.mro_diff = self.calculate_mro_diff([y1, y2], [Gz1, Gz2], [Fz1, Fz2], [Gz1_psd, Gz2_psd])
    

    def calculate_mro_diff(self, y_list, Gz_list, Fz_list, Gz_psd_list) :        
        avg_G = np.average([sample[0] for sample in self.samples], axis=0)
        avg_F = np.average([sample[1] for sample in self.samples], axis=0)

        slacks = [( (sample[0]-avg_G, sample[1]-avg_F), \
                    np.array([-np.trace(Am@(sample[0]-avg_G))-bm.T@(sample[1]-avg_F) for (Am, bm) in zip(self.A_vals, self.b_vals)]), \
                    [np.array([[np.trace(Ap[i,j]@(sample[0]-avg_G))+bp[i,j].T@(sample[1]-avg_F) for j in range(Ap.shape[1])] for i in range(Ap.shape[0])]) for (Ap, bp) in zip(self.PSD_A_vals, self.PSD_b_vals)] ) \
                      for sample in self.samples]
        
        out = 0
        for (diff, LGF, H_list) in slacks :
            temp = [ - cp.trace(Gz@diff[0]) - Fz.T@diff[1] - LGF.T@y - cp.sum([cp.trace(G_psd@H) for (G_psd, H) in zip(Gz_psd, H_list)]) for (y, Gz, Fz, Gz_psd) in zip(y_list, Gz_list, Fz_list, Gz_psd_list) ]
            out += cp.max(cp.vstack(temp))
        
        return 1/len(self.samples) * out


    def solve_eps_vals(self, eps_vals):
        out = []
        for eps in eps_vals:
            print(f'solving eps={eps}')
            res = self.solve_single_eps_val(eps)
            if self.mro_diff is not None:
                res += self.mro_diff.value
            out.append(res)
        return np.array(out)

    def solve_single_eps_val(self, eps):
        self.eps_param.value = eps
        res = self.cp_problem.solve(solver=cp.CLARABEL)
        if self.cp_problem.status != 'optimal' :
            print(self.cp_problem.status)
        if self.mro_diff is not None:
            res += self.mro_diff.value
        return res

    def solve_single_alpha_eps_val(self, alpha, eps):
        self.alpha_inv_param.value = 1 / alpha
        self.eps_param.value = eps
        res = self.cp_problem.solve(solver=cp.CLARABEL)
        if self.cp_problem.status != 'optimal' :
            print(self.cp_problem.status)
        if self.mro_diff is not None:
            res += self.mro_diff.value
        return res

    def solve_fixed_alpha_eps_vals(self, alpha, eps_vals):
        self.alpha_inv_param.value = 1 / alpha
        out = []
        t_value = []
        for eps in eps_vals:
            print(f'solving alpha={alpha} eps={eps}')
            res = self.solve_single_eps_val(eps)
            out.append(res)
            t_value.append(self.quantile_estimate.value)
            if self.mro_diff is not None:
                t_value[-1] = t_value[-1] + self.mro_diff.value
        return np.array(out), np.array(t_value)
    
    def solve_fixed_eps_alpha_vals(self, alpha_vals, eps) :
        self.eps_param.value = eps
        out = []
        t_value = []
        for alpha in alpha_vals:
            print(f'solving alpha={alpha} eps={eps}')
            res = self.solve_single_alpha_eps_val(alpha, eps)
            out.append(res)
            t_value.append(self.quantile_estimate.value)
            if self.mro_diff is not None:
                t_value[-1] = t_value[-1] + self.mro_diff.value
        return np.array(out), np.array(t_value)
        

    def setup_clarabel_problem(self):
        if self.measure == 'expectation':
            self.setup_clarabel_expectation_problem()
        if self.measure == 'cvar':
            self.setup_clarabel_cvar_problem()

    def setup_clarabel_expectation_problem(self, eps=0.1):
        '''
            x var structure: [lambd, s, y, Fz, Gz]
            s in Rn, y in Rm, Fz in RV, Gz in RS_vec
        '''
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)
        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        print('N, M, V, S_vec:', N, M, V, S_vec)

        x_dim = 1 + N * (1 + M + V + S_vec)
        print('x_dim:', x_dim)

        lambd_idx = 0
        lambd_offset = 1

        s_start = lambd_offset
        def s_idx(i):
            return lambd_offset + i
        s_offset = s_idx(N)
        # print(s_offset)

        y_start = s_offset
        def y_idx(i, j):
            return s_offset + i * M + j 
        y_offset = y_idx(N, 0)
        # print(y_offset)

        Fz_start = y_offset
        def Fz_idx(i, j):
            return y_offset + i * V + j
        Fz_offset = Fz_idx(N, 0)
        # print(Fz_offset)

        Gz_start = Fz_offset
        def Gz_idx(i, j):
            return Fz_offset + i * S_vec + j
        Gz_offset = Gz_idx(N, 0)
        # print(Gz_offset)

        c = self.c_vals

        A = []
        b = []
        cones = []
        # constraints: -c^T yi - Tr(G_sample @ Gz_i) - F_sample @ Fz_i - s_i <= 0
        epi_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            epi_constr[i, y_start: y_end] = -c
            epi_constr[i, s_idx(i)] = -1

            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            epi_constr[i, Fz_start: Fz_end] = -F_sample

            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)
            epi_constr[i, Gz_start: Gz_end] = -symm_vectorize(G_sample, 2)

        # print(epi_constr)
        
        A.append(spa.csc_matrix(epi_constr))
        b.append(np.zeros(N))
        # cones.append(clarabel.NonnegativeConeT(N)) # coalescing with next nonneg cone

        # constraints: yi >= 0
        y_nonneg = np.zeros((N * M, x_dim))
        y_start = y_idx(0, 0)
        y_end = y_idx(N-1, M)
        y_nonneg[0: N*M, y_start: y_end] = -np.eye(N * M)

        A.append(spa.csc_matrix(y_nonneg))
        b.append(np.zeros(N * M))
        # cones.append(clarabel.NonnegativeConeT(N * M))
        cones.append(clarabel.NonnegativeConeT(N + N * M)) # coalesce from above ineq constraints

        # constraints: -B^Ty_i + Fz_i = -b_obj
        Bm_full = np.array(self.b_vals)
        Bm_T = Bm_full.T

        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            curr_lhs[:, y_start: y_end] = -Bm_T

            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            curr_lhs[:, Fz_start: Fz_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(-self.b_obj)
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # PSD constraints
        Aobj_svec = symm_vectorize(self.A_obj, np.sqrt(2.))
        Am_svec = [symm_vectorize(self.A_vals[m], np.sqrt(2.)) for m in range(M)]

        # print(Aobj_svec)
        # print(Am_svec)
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T
        # print(Am_T)

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            curr_lhs[:, y_start: y_end] = -Am_T

            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)
            scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
            curr_lhs[:, Gz_start: Gz_end] = scaledI # need to scale the off-triangles

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(-Aobj_svec)

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        # SOCP constraints
        socp_lhs = []
        socp_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)

            scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
            curr_lhs[0, lambd_idx] = -1
            curr_lhs[1: V + 1, Fz_start: Fz_end] = -np.eye(V)
            curr_lhs[V + 1:, Gz_start: Gz_end] = -scaledI

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))
        
        socp_lhs = spa.vstack(socp_lhs)
        socp_rhs = np.hstack(socp_rhs)

        A.append(socp_lhs)
        b.append(socp_rhs)
        cones += [clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(N)]

        A = spa.vstack(A)
        b = np.hstack(b)
        print('A shape from custom:', A.shape)

        P = spa.csc_matrix((x_dim, x_dim))
        q = np.zeros(x_dim)
        q[0] = eps
        q[s_idx(0): s_idx(N)] = 1 / N

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

        solution = solver.solve()
        # print(solution.x)
        # print(q @ solution.x)

        return solution.obj_val

    def setup_clarabel_cvar_problem(self, eps=0.1, alpha=0.1):
        '''
            x var structure: [lambd, t, s, y1, y1, Fz1, Fz2, Gz1, Gz2]
            s in Rn, y in Rm, Fz in RV, Gz in RS_vec
        '''
        alpha_inv = 1 / alpha
        N = len(self.samples)
        M = len(self.A_vals)
        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        print('N, M, V, S_vec:', N, M, V, S_vec)

        x_dim = 2 + N * (1 + 2 * (M + V + S_vec))
        print('x_dim:', x_dim)

        lambd_idx = 0
        # lambd_offset = 1

        t_idx = 1
        t_offset = 2

        s_start = t_offset
        def s_idx(i):
            return s_start + i
        s_offset = s_idx(N)

        y1_start = s_offset
        def y1_idx(i, j):
            return y1_start + i * M + j
        y1_offset = y1_idx(N, 0)

        y2_start = y1_offset
        def y2_idx(i, j):
            return y2_start + i * M + j
        y2_offset = y2_idx(N, 0)

        fz1_start = y2_offset
        def fz1_idx(i, j):
            return fz1_start + i * V + j
        fz1_offset = fz1_idx(N, 0)

        fz2_start = fz1_offset
        def fz2_idx(i, j):
            return fz2_start + i * V + j
        fz2_offset = fz2_idx(N, 0)

        Gz1_start = fz2_offset
        def Gz1_idx(i, j):
            return Gz1_start + i * S_vec + j
        Gz1_offset = Gz1_idx(N, 0)

        Gz2_start = Gz1_offset
        def Gz2_idx(i, j):
            return Gz2_start + i * S_vec + j

        c = self.c_vals
        A = []
        b = []
        cones = []

        # constraints: y1 >= 0, y2 >= 0
        y1_nonneg = np.zeros((N * M, x_dim))
        y1_idx_start = y1_idx(0, 0)
        y1_idx_end = y1_idx(N-1, M)
        y1_nonneg[0: N*M, y1_idx_start: y1_idx_end] = -np.eye(N * M)

        y2_nonneg = np.zeros((N * M, x_dim))
        y2_idx_start = y2_idx(0, 0)
        y2_idx_end = y2_idx(N-1, M)
        y2_nonneg[0: N*M, y2_idx_start: y2_idx_end] = -np.eye(N * M)

        A += [spa.csc_matrix(y1_nonneg), spa.csc_matrix(y2_nonneg)]
        b += [np.zeros(2 * N * M)]
        cones += [clarabel.NonnegativeConeT(2 * N * M)]

        # SOCP constraints
        socp_lhs = []
        socp_rhs = []
        scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
        for i in range(N):
            # Gz1, fz1
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)

            curr_lhs[0, lambd_idx] = -1
            curr_lhs[1: V+1, fz1_idx_start: fz1_idx_end] = -np.eye(V)
            curr_lhs[V+1:, Gz1_idx_start: Gz1_idx_end] = -scaledI

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))

            # Gz2, fz2
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)

            curr_lhs[0, lambd_idx] = -1
            curr_lhs[1:V+1, fz2_idx_start: fz2_idx_end] = -np.eye(V)
            curr_lhs[V+1:, Gz2_idx_start: Gz2_idx_end] = -scaledI

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))

        socp_lhs = spa.vstack(socp_lhs)
        socp_rhs = np.hstack(socp_rhs)

        A.append(socp_lhs)
        b.append(socp_rhs)
        cones += [clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(2 * N)]

        # constraints: -B^Ty_i + Fz1_i = 0
        Bm_full = np.array(self.b_vals)
        Bm_T = Bm_full.T

        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            curr_lhs[:, y1_idx_start: y1_idx_end] = -Bm_T

            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            curr_lhs[:, fz1_idx_start: fz1_idx_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(np.zeros(V))
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # constraints: -A^\star y_1i + Gz1_i << 0
        Am_svec = [symm_vectorize(self.A_vals[m], np.sqrt(2.)) for m in range(M)]
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            curr_lhs[:, y1_idx_start: y1_idx_end] = -Am_T

            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
            curr_lhs[:, Gz1_idx_start: Gz1_idx_end] = scaledI

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(np.zeros(S_vec))

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        # constraints: t - c^T y1 - Tr(G_sample @ Gz1_i) - F_sample @ Fz1_i - s_i <= 0
        epi1_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            epi1_constr[i, t_idx] = 1
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            epi1_constr[i, y1_idx_start: y1_idx_end] = -c
            epi1_constr[i, s_idx(i)] = -1

            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            epi1_constr[i, fz1_idx_start: fz1_idx_end] = -F_sample

            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
            epi1_constr[i, Gz1_idx_start: Gz1_idx_end] = -symm_vectorize(G_sample, 2)

        A.append(spa.csc_matrix(epi1_constr))
        b.append(np.zeros(N))
        cones.append(clarabel.NonnegativeConeT(N))

        # everything from here on out depends on alpha
        # constraints: -(alpha_inv - 1) t - c^T y2 - Tr(G_sample @ Gz2_i) - F_sample @ Fz2_i - s_i <= 0
        epi2_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            epi2_constr[i, t_idx] = -(alpha_inv - 1)
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            epi2_constr[i, y2_idx_start: y2_idx_end] = -c
            epi2_constr[i, s_idx(i)] = -1

            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            epi2_constr[i, fz2_idx_start: fz2_idx_end] = -F_sample

            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
            epi2_constr[i, Gz2_idx_start: Gz2_idx_end] = -symm_vectorize(G_sample, 2)

        A.append(spa.csc_matrix(epi2_constr))
        b.append(np.zeros(N))
        cones.append(clarabel.NonnegativeConeT(N))

        # constraints: -B^Ty_2i + Fz2_i = -alpha_inv * b_obj
        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            curr_lhs[:, y2_idx_start: y2_idx_end] = -Bm_T

            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            curr_lhs[:, fz2_idx_start: fz2_idx_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(-alpha_inv * self.b_obj)
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # constraints: -A^\star y_2i + Gz2+i << -alpha_inv * A_obj
        Aobj_svec = symm_vectorize(self.A_obj, np.sqrt(2.))

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            curr_lhs[:, y2_idx_start: y2_idx_end] = -Am_T

            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
            curr_lhs[:, Gz2_idx_start: Gz2_idx_end] = scaledI

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(-alpha_inv * Aobj_svec)

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        # solving

        P = spa.csc_matrix((x_dim, x_dim))
        q = np.zeros(x_dim)
        q[0] = eps
        q[s_idx(0): s_idx(N)] = 1 / N

        A = spa.vstack(A)
        b = np.hstack(b)
        print('A shape from custom:', A.shape)

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = solver.solve()
        
        return solution.obj_val


def symm_vectorize(A, scale_factor):
    n = A.shape[0]
    rows, cols = np.tril_indices(n)  # tril is correct, need upper triangle in column order

    A_vec = A[rows, cols]
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= scale_factor

    return A_vec


def scaled_off_triangles(A, scale_factor):
    Avec = symm_vectorize(A, scale_factor)
    return np.diag(Avec)
