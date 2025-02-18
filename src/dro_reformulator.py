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

        self.extract_pep_data()
        
        if wrapper == 'cvxpy':
            self.setup_cvxpy_problem()
        elif wrapper == 'clarabel':
            self.setup_clarabel_problem()
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

        probdata, _, _ = prob.get_problem_data(cp.CLARABEL)
        A_cp = probdata['A']
        print('A shape from cvxpy:', A_cp.shape)

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

    def setup_clarabel_problem(self):
        if self.measure == 'expectation':
            self.setup_clarabel_expectation_problem()

    def setup_clarabel_expectation_problem(self, eps=0.1):
        '''
            x var structure: [lambd, s, y, Fz, Gz]
            s in Rn, y in Rm, Fz in RV, Gz in RS_vec
        '''
        N = len(self.samples)
        M = len(self.A_vals)
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

        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

        solution = solver.solve()
        # print(solution.x)
        print(q @ solution.x)


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
