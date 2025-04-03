import clarabel
import numpy as np
import scipy.sparse as spa

from .canonicalizer import Canonicalizer


class ClarabelCanonicalizer(Canonicalizer):

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
        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        # print('N, M, V, S_vec:', N, M, V, S_vec)

        x_dim = 1 + N * (1 + M + V + S_vec)
        # print('x_dim:', x_dim)

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
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T

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

        self.A = A
        self.b = b
        self.cones = cones
        self.P = spa.csc_matrix((x_dim, x_dim))
        self.s_idx_func = s_idx

    def setup_cvar_problem(self):
        # sets up the part of the cvar problem that does not depend on alpha
        pass

    def set_params(self, eps=0.1, alpha=0.1):
        if self.measure == 'expectation':
            self.set_expectation_eps(eps)

    def set_expectation_eps(self, eps):
        x_dim = self.P.shape[0]
        N = len(self.samples)
        q = np.zeros(x_dim)
        q[0] = eps
        q[self.s_idx_func(0): self.s_idx_func(N)] = 1 / N

        self.q = q

    def solve(self):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(self.P, self.q, self.A, self.b, self.cones, settings)
        solution = solver.solve()
        out = {
            'obj': solution.obj_val,
            'solvetime': solution.solve_time,
        }
        return out

    def setup_cvar_problem(self):
        raise NotImplementedError


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
